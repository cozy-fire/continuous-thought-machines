"""Full fixed-panel teacher audit, independent of any student training runner."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .config import REPO_ROOT, load_config, resolved_dict, config_hash
from .contracts import CTMState
from .data import build_manifest
from .envs import MazeEnv, FourRoomsEnv
from .teachers import MazeTeacher, FourRoomsTeacher, map_probabilities, parse_observation
from .teachers.fourrooms import file_sha256


def oracle_distances(walls, goal):
    """Independent simultaneous wavefront expansion; no BFS queue/parent/tie-break."""
    distances = np.full(walls.shape, -1, dtype=np.int64)
    frontier = np.zeros_like(walls)
    frontier[goal] = True
    depth = 0
    while frontier.any():
        distances[frontier] = depth
        adjacent = np.zeros_like(walls)
        adjacent[1:] |= frontier[:-1]
        adjacent[:-1] |= frontier[1:]
        adjacent[:, 1:] |= frontier[:, :-1]
        adjacent[:, :-1] |= frontier[:, 1:]
        frontier = adjacent & ~walls & (distances < 0)
        depth += 1
    return distances


def summarize(results, elapsed, query_seconds):
    return {"episodes": len(results), "successes": sum(r["success"] for r in results),
            "success_rate": float(np.mean([r["success"] for r in results])),
            "mean_return": float(np.mean([r["return"] for r in results])),
            "mean_length": float(np.mean([r["length"] for r in results])),
            "action_counts": np.sum([r["action_counts"] for r in results], axis=0).tolist(),
            "environment_steps": sum(r["length"] for r in results),
            "total_seconds": elapsed, "expert_query_seconds": query_seconds,
            "episodes_per_second": len(results) / elapsed, "results": results}


def inspect_maze(config, manifest, teacher):
    started, query_seconds = time.perf_counter(), 0.0
    root = REPO_ROOT / config.environment.maze_root
    results = []
    for index, entry in enumerate(manifest.validation_panel):
        env = MazeEnv(root, (entry,), seed=config.training.seed)
        try:
            obs, info = env.reset()
            walls, start, goal = parse_observation(obs.teacher_obs)
            reference = oracle_distances(walls, goal)
            initial_distance = int(reference[start])
            if initial_distance < 1:
                raise AssertionError("invalid independent initial distance")
            distances, counts, total = [], [0] * 5, 0.0
            while True:
                before_pos = env.agent_pos
                begin = time.perf_counter()
                target = teacher.predict(obs.teacher_obs[None], [info])
                query_seconds += time.perf_counter() - begin
                distance = int(target.distance_to_goal[0])
                if distance != int(reference[before_pos]):
                    raise AssertionError(f"map {entry.sha256}: BFS distance differs from oracle")
                distances.append(distance)
                action = int(target.probabilities[0].argmax())
                counts[action] += 1
                obs, reward, term, trunc, info = env.step(action)
                total += reward
                if env.agent_pos == before_pos or reference[env.agent_pos] != distance - 1:
                    raise AssertionError(f"map {entry.sha256}: first edge did not reduce distance by one")
                if term or trunc:
                    if initial_distance <= env.max_steps and not (term and env.step_count == initial_distance):
                        raise AssertionError("reachable within budget must succeed in exactly shortest distance")
                    if initial_distance > env.max_steps and not trunc:
                        raise AssertionError("long paths must respect unchanged time limit")
                    results.append({"panel_index": index, "map_path": entry.path, "map_sha256": entry.sha256,
                                    "success": term, "length": env.step_count, "return": total,
                                    "initial_distance": initial_distance, "distances_before_action": distances,
                                    "action_counts": counts, "over_time_limit": initial_distance > env.max_steps})
                    break
        finally:
            env.close()
    report = summarize(results, time.perf_counter() - started, query_seconds)
    report.update(distance_decrement_checks=sum(r["length"] for r in results),
                  over_time_limit_episodes=sum(r["over_time_limit"] for r in results),
                  source_snapshot_id=teacher.source_snapshot_id)
    return report


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def inspect_fourrooms(config, teacher):
    """Main-process batched inference, environment-only slot refill, ordered results."""
    count = config.evaluation.validation_episodes
    seeds = tuple(range(config.evaluation.fourrooms_validation_seed_start,
                        config.evaluation.fourrooms_validation_seed_start + count))
    result = [None] * count
    slots, next_index = [], 0
    started, query_seconds, env_seconds, max_difference, disagreements = time.perf_counter(), 0., 0., 0., 0
    before = teacher.parameter_hash()

    def assign(slot, index):
        slot["env"].close()
        slot["env"] = FourRoomsEnv(split="validation", evaluation_seeds=(seeds[index],))
        slot.update(index=index, obs=slot["env"].reset()[0], state=teacher.initial_state(1),
                    reference=teacher.initial_state(1), reset=True, length=0, total=0., counts=[0] * 5,
                    movements=0, rotations=0)

    try:
        for _ in range(min(config.evaluation.num_envs, count)):
            slot = {"env": FourRoomsEnv(split="validation", evaluation_seeds=(seeds[next_index],))}
            assign(slot, next_index)
            next_index += 1
            slots.append(slot)
        while slots:
            observations = np.stack([slot["obs"].teacher_obs for slot in slots])
            state = CTMState(torch.cat([s["state"].pre for s in slots]),
                             torch.cat([s["state"].post for s in slots]))
            starts = np.array([s["reset"] for s in slots], dtype=bool)
            synchronize(teacher.device)
            begin = time.perf_counter()
            output = teacher.predict(observations, state, starts)
            synchronize(teacher.device)
            query_seconds += time.perf_counter() - begin
            # Native direct calls advance an independent state along the same observation history.
            # This is audit overhead, excluded from expert_query_seconds but included in total_seconds.
            with torch.no_grad():
                initial = teacher.initial_state(len(slots))
                mask = torch.as_tensor(starts, device=teacher.device).view(-1, 1, 1)
                pre = torch.where(mask, initial.pre, torch.cat([s["reference"].pre for s in slots]))
                post = torch.where(mask, initial.post, torch.cat([s["reference"].post for s in slots]))
                synch, native_state = teacher.model.recurrent_model(
                    torch.as_tensor(observations, device=teacher.device), (pre, post))
                native = torch.softmax(teacher.model.actor(synch), -1)
                mapped = map_probabilities(native)
                max_difference = max(max_difference, float((native-output.native_probabilities).abs().max()))
                torch.testing.assert_close(native, output.native_probabilities, atol=1e-5, rtol=1e-4)
                disagreements += int((mapped.argmax(-1) != output.probabilities.argmax(-1)).sum())
            actions = output.probabilities.argmax(-1).cpu().numpy()
            remaining = []
            for i, slot in enumerate(slots):
                slot["state"] = CTMState(output.state.pre[i:i+1].clone(), output.state.post[i:i+1].clone())
                slot["reference"] = CTMState(native_state[0][i:i+1].clone(), native_state[1][i:i+1].clone())
                slot["reset"] = False
                action = int(actions[i])
                begin = time.perf_counter()
                obs, reward, term, trunc, info = slot["env"].step(action)
                env_seconds += time.perf_counter() - begin
                slot["obs"], slot["length"] = obs, slot["length"] + 1
                slot["total"] += reward
                slot["counts"][action] += 1
                slot["movements"] += int(info["agent_pos_before"] != info["agent_pos_after"])
                slot["rotations"] += int(info["agent_dir_before"] != info["agent_dir_after"])
                if term or trunc:
                    index = slot["index"]
                    if result[index] is not None:
                        raise AssertionError("duplicate panel result")
                    result[index] = {"panel_index": index, "episode_seed": seeds[index], "success": term,
                                     "length": slot["length"], "return": slot["total"],
                                     "action_counts": slot["counts"], "movements": slot["movements"],
                                     "rotations": slot["rotations"]}
                    if next_index < count:
                        assign(slot, next_index)
                        next_index += 1
                        remaining.append(slot)
                    else:
                        slot["env"].close()
                else:
                    remaining.append(slot)
            slots = remaining
    finally:
        for slot in slots:
            slot["env"].close()
    if any(r is None for r in result) or next_index != count:
        raise AssertionError("incomplete fixed panel")
    after = teacher.parameter_hash()
    if after != before:
        raise AssertionError("frozen teacher weights changed during evaluation")
    report = summarize(result, time.perf_counter()-started, query_seconds)
    report.update(source_snapshot_id=teacher.source_snapshot_id, weights_unchanged=True,
                  parameter_sha256_before=before, parameter_sha256_after=after,
                  max_native_probability_difference=max_difference, argmax_disagreements=disagreements,
                  native_parity_atol=1e-5, native_parity_rtol=1e-4, environment_step_seconds=env_seconds,
                  max_inference_batch=min(config.evaluation.num_envs, count))
    return report


def random_proxy_probe(root, entry, teacher):
    """Labels follow a random five-action proxy; this is not a student success result."""
    rng = np.random.default_rng(17)
    maze = MazeEnv(root, (entry,), seed=0)
    four = FourRoomsEnv(split="validation", evaluation_seeds=(2000000, 2000001))
    maze_queries, four_queries = 0, 0
    try:
        mt = MazeTeacher()
        obs, info = maze.reset()
        initial = maze.agent_pos
        off_shortest_edge = False
        for _ in range(30):
            target = mt.predict(obs.teacher_obs[None], [info])
            action = int(rng.integers(5))
            previous = maze.agent_pos
            obs, _, term, trunc, info = maze.step(action)
            off_shortest_edge |= maze.agent_pos != previous and action != int(target.action[0])
            maze_queries += 1
            if term or trunc:
                obs, info = maze.reset()
        # Force a legal out-and-back detour to prove query uses current position, not an old route.
        obs, info = maze.reset()
        target = mt.predict(obs.teacher_obs[None], [info])
        obs, _, term, trunc, info = maze.step(int(target.action[0]))
        if term or trunc:
            raise AssertionError("probe map needs at least two shortest-path edges")
        reverse = {0: 1, 1: 0, 2: 3, 3: 2}[int(target.action[0])]
        obs, _, _, _, info = maze.step(reverse)
        fresh = mt.predict(obs.teacher_obs[None], [info])
        if maze.agent_pos != initial or fresh.distance_to_goal[0] != target.distance_to_goal[0]:
            raise AssertionError("detour was not replanned from actual location")
        observation, _ = four.reset()
        state = teacher.initial_state(1)
        for step in range(52):
            # Step 26 manually ends a short probe episode; the next real reset is immediately labelled.
            starts = np.array([step in (0, 26)], dtype=bool)
            if step == 26:
                observation, _ = four.reset()
            output = teacher.predict(observation.teacher_obs[None], state, starts)
            if not output.valid_target.all():
                raise AssertionError("a real observation lacked its target")
            state = output.state
            observation, _, term, trunc, _ = four.step(int(rng.integers(5)))
            if term or trunc:
                raise AssertionError("unexpected proxy termination; probe setup must be explicit")
            four_queries += 1
        return {"policy_type": "random_five_action_proxy_not_student",
                "maze_queries": maze_queries + 2, "maze_out_and_back_replanned": True,
                "random_maze_nonexpert_moves_observed": off_shortest_edge,
                "fourrooms_queries": four_queries, "fourrooms_valid_targets": four_queries,
                "first_step_label": True, "reset_first_step_label": True,
                "short_episode_all_labels": True, "across_50_observation_window_continuous_state": True,
                "probe_reset_after_actions": 26}
    finally:
        maze.close()
        four.close()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    root = REPO_ROOT / config.environment.maze_root
    try:
        manifest = build_manifest(root, validation_count=config.evaluation.maze_validation_hashes,
                                  validation_episodes=config.evaluation.validation_episodes,
                                  test_episodes=config.evaluation.test_episodes,
                                  drift_episodes=min(32, config.evaluation.validation_episodes))
        manifest.save(output / "maze_manifest.json")
        write_json(output / "resolved_config.json", resolved_dict(config))
        mt = MazeTeacher()
        mt.save_descriptor(output / "maze_teacher.json")
        teacher = FourRoomsTeacher(REPO_ROOT / config.teachers.fourrooms.checkpoint, args.device)
        write_json(output / "fourrooms_teacher.json", teacher.metadata)
        print("Evaluating Maze fixed validation panel...", flush=True)
        maze = inspect_maze(config, manifest, mt)
        write_json(output / "maze_validation.json", maze)
        print(f"Maze {maze['successes']}/{maze['episodes']}; evaluating FourRooms...", flush=True)
        four = inspect_fourrooms(config, teacher)
        write_json(output / "fourrooms_validation.json", four)
        probe = random_proxy_probe(root, manifest.validation_panel[0], teacher)
        write_json(output / "random_proxy_probe.json", probe)
        count = config.evaluation.validation_episodes
        if maze["episodes"] != count or four["episodes"] != count:
            raise AssertionError("full configured panel was not completed")
        source_paths = sorted((REPO_ROOT / "tasks/continual_nav_opd").rglob("*.py"))
        source_paths += [REPO_ROOT / p for p in (
            "models/ctm_rl.py", "models/ctm.py", "models/modules.py", "models/utils.py", "models/constants.py",
            "models/resnet.py", "tasks/rl/train.py",
            "tasks/continual_nav/envs/maze.py", "tasks/continual_nav/envs/fourrooms.py",
            "tasks/continual_nav/envs/common.py", "tasks/continual_nav/data/manifest.py")]
        write_json(output / "source_manifest.json", {p.relative_to(REPO_ROOT).as_posix(): file_sha256(p) for p in source_paths})
        summary = {"status": "complete", "method": config.method, "schema_version": config.schema_version,
                   "sequence_protocol": config.sequence_protocol, "config_hash": config_hash(config),
                   "device": str(teacher.device), "torch": torch.__version__,
                   "hardware": torch.cuda.get_device_name(teacher.device) if teacher.device.type == "cuda" else "CPU",
                   "maze": {k: v for k, v in maze.items() if k != "results"},
                   "fourrooms": {k: v for k, v in four.items() if k != "results"},
                   "limits": "Teacher/interface audit only; no student training or RTX5090 performance proof."}
        write_json(output / "report.json", summary)
        write_json(output / "sha256.json", {p.name: file_sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
        print(json.dumps({"output": str(output), "maze": maze["success_rate"],
                          "fourrooms": four["success_rate"]}), flush=True)
    except Exception as exc:
        # A failed/incomplete panel never publishes a successful aggregate report.
        write_json(output / "failure.json", {"status": "failed", "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
