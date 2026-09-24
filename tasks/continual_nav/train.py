"""Explicit method runs with atomic stage-boundary recovery; no implicit seed sweep."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import uuid
from weakref import WeakSet

import torch

from . import checkpoint as ck
from .analysis.metrics import forgetting
from .config import Config, budget_summary, config_hash, load_config, resolved_dict, save_config
from .contracts import FisherState, TASKS
from .data.manifest import MazeManifest, build_manifest
from .data.replay import ReplayBank, ShardedWriter, TransitionStore
from .data.rollout import collect_fisher
from .envs import VectorEnvAdapter, build_env
from .evaluate import evaluate_policy
from .learning.common import freeze, prepare_kb, prepare_ppo
from .learning.distill import run_compress_stage
from .learning.fisher import estimate_fisher, update_online_fisher
from .learning.ppo import run_ppo_stage
from .learning.world import SnapshotCollector, fit_world_model, make_world_optimizer
from .models import Controller, DualPolicy, SIGReg, SingleActorCritic, StandalonePolicy, VisionEncoder, WorldModel, frozen_copy
from .schedule import METHODS, RandomStreams, expand_stages, ends_visit, visit_identity, isolated_rng
from .wandb_logging import WandbLogger


_PRE_WANDB_SOURCES = {
    "tasks/continual_nav/config.py": "c56ae3d2da03ae63bffc3f5657be6c7478347afd37c52438d5b25ae0a7d97db3",
    "tasks/continual_nav/train.py": "0d3e21b7e5340a9567158b8ef81d6d0ba5248b03493938df2300555df2ea1e22",
}


def _validate_resume_sources(saved: dict[str, str], current: dict[str, str]) -> bool:
    if saved == current:
        return False
    # Permit only the exact pre-W&B checkout to cross this logging-only boundary.
    if (all(saved.get(path) == digest for path, digest in _PRE_WANDB_SOURCES.items())
            and set(current) - set(saved) == {"tasks/continual_nav/wandb_logging.py"}
            and set(saved) - set(current) == set()
            and all(saved[path] == current[path] for path in saved if path not in _PRE_WANDB_SOURCES)):
        return True
    raise ValueError("checkpoint source manifest mismatch")


class CountedVector(VectorEnvAdapter):
    def __init__(self, envs, callback):
        super().__init__(envs)
        self.callback = callback

    def step(self, actions):
        result = super().step(actions)
        self.callback(self.num_envs)
        return result


def prepare_manifest(config: Config, supplied: Path | None) -> MazeManifest:
    ev = config.evaluation
    if supplied is None:
        return build_manifest(config.environment.maze_root, validation_count=ev.maze_validation_hashes,
            validation_episodes=ev.validation_episodes, test_episodes=ev.test_episodes, drift_episodes=ev.drift_episodes)
    manifest = MazeManifest.load(supplied, config.environment.maze_root)
    if len({entry.sha256 for entry in manifest.validation}) != ev.maze_validation_hashes:
        raise ValueError("supplied manifest validation hash count differs from configuration")
    if len(manifest.validation_panel) < ev.validation_episodes or len(manifest.test_panel) < ev.test_episodes or len(manifest.drift_panel) < ev.drift_episodes:
        raise ValueError("supplied manifest has insufficient evaluation panels")
    return replace(manifest, validation_panel=manifest.validation_panel[:ev.validation_episodes],
                   test_panel=manifest.test_panel[:ev.test_episodes], drift_panel=manifest.drift_panel[:ev.drift_episodes])


class Runner:
    def __init__(self, config: Config, method: str, seed: int, root: Path, *, task: str | None = None,
                 manifest_path: Path | None = None, vision_checkpoint: Path | None = None, resume: bool = False,
                 wandb_mode: str | None = None):
        self.config, self.method, self.seed, self.task, self.root = config, method, seed, task, root.resolve()
        self.stages = expand_stages(config, method, task)
        if seed not in config.training.seeds:
            raise ValueError("seed must be explicitly listed in the run configuration")
        self.device = torch.device(config.training.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("configured CUDA device is unavailable; no silent device fallback")
        self.rng = RandomStreams(seed, method, task)
        self.sources = ck.source_manifest()
        self.single = method in (METHODS[1], METHODS[2])
        self.encoder = self.world = self.kb = self.policy = self.world_optimizer = None
        self.fisher, self.kb_ready, self.fresh, self.pools = None, False, None, {}
        self.vision_source = self.vision_export = None
        self.completed, self.evaluations, self.next_index = [], [], 1
        self.pending_active = {}
        self.finalized, self._logger = False, None
        self.counters = {name: 0 for name in ("global_env_steps", "world_collect_steps", "explore_steps", "progress_steps",
            "compress_steps", "fisher_steps", "eval_steps", "world_optimizer_updates", "ppo_optimizer_updates",
            "distill_optimizer_updates", "policy_internal_ticks", "eval_internal_ticks")}
        self._hooked, self._evaluating = WeakSet(), False
        logging_migration = False
        if resume:
            payload = ck.load(self.root, expected_config_hash=config_hash(config))
            logging_migration = _validate_resume_sources(payload["sources"], self.sources)
            if (payload["method"], payload["seed"], payload["task"]) != (method, seed, task):
                raise ValueError("resume method/seed/task mismatch")
            if payload["stages"] != [s.record() for s in self.stages] or payload["policy_kind"] != ("single" if self.single else "dual"):
                raise ValueError("resume schedule/policy kind mismatch")
            self.manifests = payload["manifests"]
            self.manifest = MazeManifest.load(ck.verify_reference(self.root, self.manifests["maze"]), config.environment.maze_root)
            self._construct()
            self._restore(payload)
        else:
            if method != METHODS[0] and vision_checkpoint is None:
                raise ValueError("shared-vision baseline requires --vision-checkpoint")
            if method == METHODS[0] and vision_checkpoint is not None:
                raise ValueError("main method cannot import a baseline visual checkpoint")
            self.manifest = prepare_manifest(config, manifest_path)
            self.root.mkdir(parents=True, exist_ok=False)
            save_config(config, self.root / "resolved_config.yaml")
            self.manifest.save(self.root / "manifests/maze_splits.json")
            panels = dict(maze_validation=[asdict(e) for e in self.manifest.validation_panel],
                maze_test=[asdict(e) for e in self.manifest.test_panel], maze_drift=[asdict(e) for e in self.manifest.drift_panel],
                fourrooms_validation=list(range(config.evaluation.fourrooms_validation_seed_start,
                    config.evaluation.fourrooms_validation_seed_start+config.evaluation.validation_episodes)),
                fourrooms_test=list(range(config.evaluation.fourrooms_test_seed_start,
                    config.evaluation.fourrooms_test_seed_start+config.evaluation.test_episodes)))
            ck.atomic_json(self.root / "manifests/eval_panels.json", panels)
            self.manifests = {name: ck.reference(self.root, self.root / path) for name, path in
                             (("maze", "manifests/maze_splits.json"), ("evaluation", "manifests/eval_panels.json"))}
            self._construct()
            if vision_checkpoint is not None:
                self._import_vision(vision_checkpoint)
            ck.atomic_json(self.root / "provenance.json", dict(method=method, seed=seed, task=task,
                sources=self.sources, config_hash=config_hash(config), device=str(self.device), rng_seeds=self.rng.seeds,
                stages=[s.record() for s in self.stages], python=platform.python_version(),
                cuda_runtime=torch.version.cuda,
                gpu=torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
                versions={name: importlib.metadata.version(name)
                for name in ("torch", "torchvision", "numpy", "gymnasium", "minigrid", "PyYAML", "wandb")}))
            self.completed = ["init"]
        self.bank = ReplayBank(self.root / "replay", capacity=config.replay.high_error_capacity_per_task,
                                shard_size=config.replay.shard_size) if method == METHODS[0] else None
        if resume and self.bank is not None:
            # The global checkpoint, not a possibly uncommitted latest pool, is authoritative.
            for current_task in TASKS:
                if current_task in self.pools:
                    self.bank._publish(current_task, ck.verify_reference(self.root, self.pools[current_task], replay=True))
                else:
                    (self.root / "replay" / current_task / "latest.json").unlink(missing_ok=True)
        if not resume:
            self._commit("init")
        mode = wandb_mode or ("online" if config.training.remote_logging else "disabled")
        self._logger = WandbLogger(self.root, config, method, seed, task, mode=mode, resume=resume)
        if logging_migration:
            self._event(dict(type="logging_migration", stage=self.completed[-1], backend="wandb"))

    def _construct(self):
        with self.rng.model_initialization(0):
            self.encoder = VisionEncoder(self.config).to(self.device)
            if self.method == METHODS[0]:
                self.world = WorldModel(self.config, self.encoder).to(self.device)
                self.world_optimizer = make_world_optimizer(self.world, self.config)
            if self.single:
                self.policy = SingleActorCritic(self.config).to(self.device)
                self._track_ticks(self.policy)
            else:
                self.kb = StandalonePolicy(self.config).to(self.device)
                self._track_ticks(self.kb)

    def _track_ticks(self, module):
        for child in module.modules():
            if isinstance(child, Controller) and child not in self._hooked:
                self._hooked.add(child)
                def count(_, args):
                    name = "eval_internal_ticks" if self._evaluating else "policy_internal_ticks"
                    self.counters[name] += args[0].shape[0]
                child.synapse.register_forward_pre_hook(count)

    def _import_vision(self, path):
        artifact = ck.load_artifact(path)
        if artifact["kind"] != "vision" or artifact["method"] != METHODS[0] or artifact["seed"] != self.seed:
            raise ValueError("vision artifact must come from the same seed's main method TA")
        structural = ("schema_version", "observation", "vision", "ctm", "attention")
        if any(artifact["config"][k] != resolved_dict(self.config)[k] for k in structural):
            raise ValueError("vision architecture mismatch")
        if artifact["maze_manifest_hash"] != self.manifests["maze"]["sha256"]:
            raise ValueError("shared vision belongs to different fixed data/evaluation panels")
        self.encoder.load_state_dict(artifact["encoder"])
        if not bool(self.encoder.permanently_frozen):
            raise ValueError("shared vision must already be permanently frozen")
        self.encoder.freeze(permanent=True)
        destination = self.root / "exports/shared_vision.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        shutil.copyfile(path.with_suffix(path.suffix+".sha256.json"), destination.with_suffix(destination.suffix+".sha256.json"))
        self.vision_source = ck.reference(self.root, destination)

    def _restore(self, payload):
        self.encoder.load_state_dict(payload["models"]["encoder"])
        if self.world is not None:
            self.world.load_state_dict(payload["models"]["world"])
            self.world_optimizer.load_state_dict(payload["world_optimizer"])
            self.world.freeze()
        elif payload["models"]["world"] is not None or payload["world_optimizer"] is not None:
            raise ValueError("baseline checkpoint contains forbidden world state")
        if self.single:
            if any(payload["models"][name] is not None for name in ("kb", "dual")) or payload["fisher"] is not None:
                raise ValueError("single-column checkpoint contains KB/dual/Fisher state")
            self.policy.load_state_dict(payload["models"]["single"])
        else:
            self.kb.load_state_dict(payload["models"]["kb"])
            if payload["models"]["dual"] is not None:
                with self.rng.model_initialization(payload["next_index"]):
                    self.policy = DualPolicy(self.config, self.kb).to(self.device)
                self.policy.load_state_dict(payload["models"]["dual"])
                self._track_ticks(self.policy)
        self.fisher = FisherState(**payload["fisher"]) if payload["fisher"] is not None else None
        self.pending_active = payload.get("pending_active", {})
        if self.fisher is not None:
            from .learning.fisher import validate_fisher
            validate_fisher(self.kb, self.fisher)
        for key in ("kb_ready", "fresh", "pools", "vision_source", "vision_export", "completed", "evaluations", "next_index", "counters", "finalized"):
            setattr(self, key, payload[key])
        if self.completed != [str(s.key) for s in self.stages[:self.next_index]]:
            raise ValueError("completed stage prefix does not match next stage")
        self.rng.load_state_dict(payload["rng"])
        self.encoder.freeze()

    def costs(self):
        shared = None
        vision_ref = self.vision_source or self.vision_export
        if vision_ref is not None:
            artifact = torch.load(ck.verify_reference(self.root, vision_ref), map_location="cpu", weights_only=True)
            shared = artifact["shared_ta_steps"]
        return dict(shared_visual_TA_generation_steps=shared, shared_cost_paid_in_this_run=self.method == METHODS[0],
                    planned_shared_TA_steps=budget_summary(self.config)["shared_ta_steps"],
                    actual_committed_training_steps=self.counters["global_env_steps"],
                    additional_downstream_training_steps=(max(0, self.counters["global_env_steps"]-(shared or self.counters["global_env_steps"]))
                        if self.method == METHODS[0] else self.counters["global_env_steps"]),
                    downstream_progress_steps=self.counters["progress_steps"],
                    planned_run_training_steps=sum(s.transitions for s in self.stages),
                    evaluation_steps=self.counters["eval_steps"])

    def _payload(self, stage_key):
        return dict(schema_version=1, config=resolved_dict(self.config), config_hash=config_hash(self.config),
            sources=self.sources, method=self.method, policy_kind="single" if self.single else "dual", seed=self.seed,
            task=self.task, current_stage=stage_key, next_index=self.next_index, completed=self.completed,
            stages=[s.record() for s in self.stages], counters=self.counters, rng=self.rng.state_dict(),
            models=dict(encoder=self.encoder.state_dict(), world=self.world.state_dict() if self.world is not None else None,
                kb=self.kb.state_dict() if self.kb is not None else None,
                single=self.policy.state_dict() if self.single else None,
                dual=self.policy.state_dict() if not self.single and self.policy is not None else None),
            world_optimizer=self.world_optimizer.state_dict() if self.world_optimizer is not None else None,
            fisher=asdict(self.fisher) if self.fisher is not None else None, kb_ready=self.kb_ready,
            fresh=self.fresh, pools=self.pools, manifests=self.manifests, vision_source=self.vision_source,
            vision_export=self.vision_export, pending_active=self.pending_active,
            evaluations=self.evaluations, costs=self.costs(), finalized=self.finalized)

    def _commit(self, key):
        self.marker = ck.commit(self.root, key, self._payload(key))

    def _event(self, value):
        # Keep both axes: total committed interactions and task-reward PPO progress.
        progress = value.get("consumed", value.get("stage_training_steps", 0)) if value.get("stage", "").endswith("/P") else 0
        value["downstream_progress_steps"] = self.counters["progress_steps"]+progress
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, allow_nan=False)+"\n")
        if value["type"] in ("ppo_update", "world_update", "distill_update", "learner_update",
                             "evaluation", "visual_drift", "stage_complete"):
            with (self.root / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(value, allow_nan=False)+"\n")
        if self._logger is not None:
            self._logger.log(value, self.counters)

    def close_logging(self, *, exit_code: int = 0) -> None:
        if self._logger is not None:
            self._logger.finish(exit_code=exit_code)
            self._logger = None

    def _evaluate(self, policy, stage, event, *, policy_kind="kb", source_task=None, encoder=None):
        self._evaluating = True
        try:
            report = evaluate_policy(policy, encoder if encoder is not None else self.encoder, self.manifest, self.config)
        finally:
            self._evaluating = False
        family, visit = visit_identity(stage)
        return dict(stage=str(stage.key), event=event, family=family, visit=visit,
                    policy_kind=policy_kind, source_task=source_task, split="validation",
                    evaluation_id=f"{family}/v{visit}/{policy_kind}/{source_task or 'all'}/validation",
                    stage_training_steps=0, global_training_steps=self.counters["global_env_steps"], report=report)

    def _save_active(self, stage):
        vision = self.vision_export or self.vision_source
        if vision is None or not self.encoder.permanently_frozen:
            raise RuntimeError("P Active snapshot requires the shared frozen vision artifact")
        path = self.root / "exports/active" / f"v{stage.key.visit}_{stage.key.task}.pt"
        # The DualPolicy includes its original KB and adapter, not the future C student.
        ck.save_artifact(path, dict(kind="active_snapshot", family=stage.key.family,
            visit=stage.key.visit, task=stage.key.task, policy=self.policy.state_dict(),
            vision=vision, encoder_version=int(self.encoder.encoder_version)))
        self.pending_active[stage.key.task] = ck.reference(self.root, path)

    def _load_active(self, task, visit):
        path = ck.verify_reference(self.root, self.pending_active[task])
        payload = ck.load_artifact(path)
        if (payload["kind"], payload["family"], payload["visit"], payload["task"]) != (
                "active_snapshot", "pnc", visit, task):
            raise ValueError("Active snapshot identity mismatch")
        # Never reconstruct a historical teacher on self.kb: load_state_dict would
        # overwrite the live, already compressed KB through shared parameter storage.
        with isolated_rng():
            policy = DualPolicy(self.config, StandalonePolicy(self.config))
            policy.load_state_dict(payload["policy"])
            encoder = VisionEncoder(self.config)
            vision = ck.load_artifact(ck.verify_reference(self.root, payload["vision"]))
            encoder.load_state_dict(vision["encoder"])
            encoder.freeze(permanent=True)
        if int(encoder.encoder_version) != payload["encoder_version"]:
            raise ValueError("Active snapshot encoder version mismatch")
        self._track_ticks(policy)
        return policy.to(self.device), encoder.to(self.device)

    def _evaluate_visit(self, stage):
        rows = [self._evaluate(self.policy if self.single else self.kb, stage, "visit_end",
                              policy_kind="single" if self.single else "kb")]
        if stage.key.family == "pnc":
            for task in self.config.task_order:
                policy, encoder = self._load_active(task, stage.key.visit)
                try:
                    rows.append(self._evaluate(policy, stage, "visit_end", policy_kind="active",
                                               source_task=task, encoder=encoder))
                finally:
                    del policy, encoder
        # A partial panel/strategy set must not be published as a complete visit.
        for item in rows:
            self.counters["eval_steps"] += item["report"]["transitions"]
            self.evaluations.append(item)
            self._event(dict(type="evaluation", **item))
        self.pending_active = {}

    def _phase_modes(self):
        return {name: dict(training=module.training, trainable=[n for n, p in module.named_parameters() if p.requires_grad])
                for name, module in (("encoder", self.encoder), ("world", self.world), ("kb", self.kb), ("policy", self.policy)) if module is not None}

    def _assert_phase_modes(self, stage):
        world_fit = stage.key.subphase == "fit"
        if self.encoder.training != world_fit or any(p.requires_grad != world_fit for p in self.encoder.parameters()):
            raise RuntimeError("visual phase ownership mismatch")
        if self.world is not None and any(p.requires_grad != world_fit for p in self.world.parameters()):
            raise RuntimeError("world phase ownership mismatch")
        if self.kb is not None and any(p.requires_grad != (stage.key.phase in ("C", "F")) for p in self.kb.parameters()):
            raise RuntimeError("KB phase ownership mismatch")
        if self.policy is not None:
            for name, p in self.policy.named_parameters():
                expected = stage.key.phase in ("X", "P") and (self.single or not name.startswith("kb."))
                if p.requires_grad != expected:
                    raise RuntimeError(f"policy phase ownership mismatch: {name}")

    def run_next(self):
        if self.next_index == len(self.stages):
            return False
        stage = self.stages[self.next_index]
        attempt_path = self.root / "attempts" / (str(stage.key).replace("/", "__")+"__"+uuid.uuid4().hex+".json")
        attempt = dict(stage=str(stage.key), confirmed_env_steps=0, status="running", committed=False)
        ck.atomic_json(attempt_path, attempt)
        def observed(count):
            attempt["confirmed_env_steps"] += count
            ck.atomic_json(attempt_path, attempt)
        phase, sub = stage.key.phase, stage.key.subphase
        envs, builder = None, None
        metrics = {}
        try:
            if phase in ("X", "P") and not self.single:
                with self.rng.model_initialization(stage.ordinal):
                    self.policy = DualPolicy(self.config, self.kb, kb_ready=self.kb_ready).to(self.device)
                self._track_ticks(self.policy)
            if phase == "P":
                self.encoder.freeze(permanent=True)
            if phase != "W" and self.world is not None:
                self.world.freeze()
            if phase == "W":
                freeze(self.kb)
                if sub == "fit":
                    self.world.set_fit_mode()
                else:
                    self.world.freeze()
            elif phase in ("X", "P"):
                if self.world is not None:
                    self.world.freeze()
                prepare_ppo(self.policy, self.encoder)
            elif phase == "C":
                self.policy = frozen_copy(self.policy)  # Separate teacher old-KB from live student before enabling gradients.
                prepare_kb(self.kb, self.encoder)
            else:
                prepare_kb(self.kb, self.encoder)
                self.kb.eval()
            self._assert_phase_modes(stage)
            self._event(dict(type="stage_enter", stage=str(stage.key), modes=self._phase_modes()))
            if phase != "W" or sub != "fit":
                seeds = self.rng.numpy["env"].integers(0, 2**63-1, size=self.config.training.num_envs)
                envs = CountedVector([build_env(stage.key.task, "train", int(seed), config=self.config,
                                       maze_manifest=self.manifest) for seed in seeds], observed)
            if phase == "W" and sub == "collect":
                freeze(self.kb)
                self.world.freeze()
                collector = SnapshotCollector(self.encoder, self.kb, self.config, world_version=int(self.world.world_model_version))
                directory = self.root / "world_fresh" / uuid.uuid4().hex
                writer = ShardedWriter(directory, task=stage.key.task, encoder_version=int(self.encoder.encoder_version),
                    world_version=int(self.world.world_model_version), source="W", source_stage=str(stage.key),
                    expected_count=stage.transitions, shard_size=self.config.replay.shard_size, snapshot_id=collector.snapshot_id)
                result = collector.collect(envs, writer, steps=stage.transitions,
                    action_rng=self.rng.torch["policy_action"], start_transition_id=self.counters["global_env_steps"])
                self.fresh = ck.reference(self.root, Path(result.manifest))
                self.counters["world_collect_steps"] += stage.transitions
            elif phase == "W":
                fresh = TransitionStore(ck.verify_reference(self.root, self.fresh, replay=True))
                high_ref = self.pools.get(stage.key.task)
                high = TransitionStore(ck.verify_reference(self.root, high_ref, replay=True)) if high_ref else None
                def world_progress(updates, metrics):
                    self._event(dict(type="world_update", stage=str(stage.key),
                        world_optimizer_updates=self.counters["world_optimizer_updates"]+updates, metrics=metrics))
                metrics = fit_world_model(self.world, SIGReg(self.config.sigreg), fresh, high, self.world_optimizer, self.config,
                    task=stage.key.task, replay_rng=self.rng.numpy["replay"], sigreg_rng=self.rng.torch["sigreg"],
                    on_update=world_progress)
                self.world.commit_fit()
                self.counters["world_optimizer_updates"] += stage.world_updates
                self.fresh = None
            elif phase in ("X", "P"):
                if phase == "X":
                    builder = self.bank.begin(stage.key.task, encoder_version=int(self.encoder.encoder_version),
                        world_version=int(self.world.world_model_version), source_stage=str(stage.key), expected_count=stage.transitions)
                def progress(consumed, metrics):
                    self._event(dict(type="ppo_update", stage=str(stage.key), consumed=consumed, metrics=metrics))
                result = run_ppo_stage(envs, self.policy, self.encoder, self.config, phase=phase, steps=stage.transitions,
                    action_rng=self.rng.torch["policy_action"], minibatch_rng=self.rng.torch["ppo_shuffle"],
                    world=self.world if phase == "X" else None, high_error=builder,
                    start_transition_id=self.counters["global_env_steps"], on_rollout=progress)
                if phase == "P" and not self.single:
                    self._save_active(stage)
                self.counters["ppo_optimizer_updates"] += result["updates"]
                self.counters["explore_steps" if phase == "X" else "progress_steps"] += stage.transitions
                if builder is not None:
                    self.pools[stage.key.task] = ck.reference(self.root, Path(result["pool_manifest"]))
            elif phase == "C":
                def distill_progress(consumed, updates, metrics):
                    self._event(dict(type="distill_update", stage=str(stage.key), consumed=consumed,
                        distill_optimizer_updates=self.counters["distill_optimizer_updates"]+updates, metrics=metrics))
                result = run_compress_stage(envs, self.policy, self.kb, self.encoder, self.config, self.fisher,
                    steps=stage.transitions, action_rng=self.rng.torch["policy_action"], minibatch_rng=self.rng.torch["ppo_shuffle"],
                    start_transition_id=self.counters["global_env_steps"], on_window=distill_progress)
                self.counters["compress_steps"] += stage.transitions
                self.counters["distill_optimizer_updates"] += result["updates"]
                metrics = result["last"]
                self.kb_ready = True
                self.policy = None  # The teacher is no longer needed at a C boundary.
            else:
                sequences = collect_fisher(envs, self.kb, self.encoder, self.config, action_rng=self.rng.torch["policy_action"],
                    fisher_rng=self.rng.torch["fisher"], start_transition_id=self.counters["global_env_steps"])
                current = estimate_fisher(self.kb, self.encoder, sequences, self.config)
                self.fisher = update_online_fisher(self.kb, current, self.fisher,
                    decay=self.config.ewc.agnostic_decay if stage.key.family == "ta" else self.config.ewc.pnc_decay,
                    sample_count=self.config.fisher.scored_samples, stage_key=str(stage.key), encoder_version=int(self.encoder.encoder_version))
                self.counters["fisher_steps"] += stage.transitions
                metrics = dict(fisher_sum=sum(x.sum().item() for x in current.values()), scored_samples=self.config.fisher.scored_samples)
            if attempt["confirmed_env_steps"] != stage.transitions:
                raise RuntimeError("actual stage interactions do not match its exact budget")
            self.counters["global_env_steps"] += stage.transitions
            if ends_visit(self.stages, self.next_index):
                self._evaluate_visit(stage)
            self.completed.append(str(stage.key)); self.next_index += 1
            if metrics:
                self._event(dict(type="learner_update", stage=str(stage.key), metrics=metrics))
            if stage.key.family == "ta" and (self.next_index == len(self.stages) or self.stages[self.next_index].key.family != "ta"):
                self.encoder.freeze(permanent=True)
                path = self.root / "exports/vision_final.pt"
                ck.save_artifact(path, dict(kind="vision", method=self.method, seed=self.seed, config=resolved_dict(self.config),
                    encoder=self.encoder.state_dict(), shared_ta_steps=self.counters["global_env_steps"],
                    maze_manifest_hash=self.manifests["maze"]["sha256"]))
                self.vision_export = ck.reference(self.root, path)
            self._event(dict(type="stage_complete", stage=str(stage.key), modes=self._phase_modes(), counters=self.counters))
            self._commit(str(stage.key))
            attempt.update(status="complete", committed=True, checkpoint=self.marker.name)
            ck.atomic_json(attempt_path, attempt)
            self._cleanup()
            if self.next_index == len(self.stages):
                self.finalize()
            return True
        except BaseException as error:
            attempt.update(status="failed", error=type(error).__name__+": "+str(error))
            ck.atomic_json(attempt_path, attempt)
            raise
        finally:
            if envs is not None:
                envs.close()
            if builder is not None:
                builder.abort()

    def _cleanup(self):
        # Only committed dependencies survive; old markers may intentionally become
        # non-resumable and will fail dependency validation rather than resample data.
        parent = self.root / "world_fresh"
        keep = ck.contained(self.root, self.fresh["path"]).parent if self.fresh else None
        if parent.exists():
            for child in parent.iterdir():
                if child.resolve().parent != parent.resolve() or not child.is_dir():
                    raise ValueError("unsafe fresh cleanup target")
                if child != keep:
                    manifest = child / "manifest.json"
                    if manifest.exists():
                        audit = self.root / "manifests/fresh_audit" / (child.name+".json")
                        audit.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(manifest, audit)
                    shutil.rmtree(child)
        if self.bank is not None:
            for task in self.pools:
                for manifest in (self.root / "replay" / task / "versions").glob("*/manifest.json"):
                    audit = self.root / "manifests/replay_audit" / (task+"_"+manifest.parent.name+".json")
                    audit.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(manifest, audit)
                self.bank.prune(task)

    def finalize(self):
        output = self.root / "exports/final.pt"
        if self.finalized:
            return
        if self.next_index != len(self.stages):
            raise RuntimeError("cannot finalize an incomplete schedule")
        policy = self.policy if self.single else self.kb
        self._evaluating = True
        try:
            result = evaluate_policy(policy, self.encoder, self.manifest, self.config, split="test")
        finally:
            self._evaluating = False
        self.counters["eval_steps"] += result["transitions"]
        result.update(method=self.method, seed=self.seed, training_task=self.task, costs=self.costs())
        ck.atomic_json(self.root / "evaluation/final_test.json", result)
        family, visit = visit_identity(self.stages[-1])
        item = dict(stage=self.completed[-1], event="final_test", family=family, visit=visit,
            policy_kind="single" if self.single else "kb", source_task=None, split="test",
            evaluation_id=f"{family}/final/{'single' if self.single else 'kb'}/test",
            global_training_steps=self.counters["global_env_steps"], stage_training_steps=0, report=result)
        self.evaluations.append(item)
        self._event(dict(type="evaluation", **item))
        matrices = {}
        for family in ("ta", "pnc"):
            rows = [item for item in self.evaluations if item.get("family") == family
                    and item.get("policy_kind") == "kb" and item["event"] == "visit_end"]
            matrix = [{task: item["report"]["tasks"][task]["success_rate"] for task in TASKS} for item in rows]
            matrices[family] = dict(visits=[row["visit"] for row in rows],
                                    success_matrix=matrix, forgetting=forgetting(matrix))
        ck.atomic_json(self.root / "evaluation/forgetting.json", matrices)
        ck.save_artifact(output, dict(kind="inference", method=self.method, seed=self.seed, task=self.task,
            policy_kind="single" if self.single else "dual", config=resolved_dict(self.config),
            encoder=self.encoder.state_dict(), policy=policy.state_dict(), maze_manifest=asdict(self.manifest), costs=self.costs()))
        self.finalized = True
        self._commit(self.completed[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, default=METHODS[0])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--maze-manifest", type=Path)
    parser.add_argument("--vision-checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        help="Override W&B mode; full runs default to online")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-stages", type=int, help="Stop after N additional complete stages; never truncate a stage")
    args = parser.parse_args()
    config = load_config(args.config)
    stages = expand_stages(config, args.method, args.task)
    if args.dry_run:
        manifest = prepare_manifest(config, args.maze_manifest)
        print(json.dumps(dict(config_hash=config_hash(config), stages=[s.record() for s in stages],
            transitions=sum(s.transitions for s in stages), world_updates=sum(s.world_updates for s in stages),
            data_counts={name: len(manifest.entries(name)) for name in ("train", "validation", "test")}), indent=2))
        return
    if args.seed is None or args.run_dir is None or (args.max_stages is not None and args.max_stages < 1):
        parser.error("training requires explicit --seed and --run-dir; --max-stages must be positive")
    torch.set_num_threads(2)
    runner = Runner(config, args.method, args.seed, args.run_dir, task=args.task, manifest_path=args.maze_manifest,
                    vision_checkpoint=args.vision_checkpoint, resume=args.resume, wandb_mode=args.wandb_mode)
    try:
        count = 0
        while runner.next_index < len(stages) and (args.max_stages is None or count < args.max_stages):
            runner.run_next(); count += 1
            print(f"committed {runner.completed[-1]}: {runner.counters['global_env_steps']} training transitions", flush=True)
        if runner.next_index == len(stages):
            runner.finalize()
    except BaseException:
        runner.close_logging(exit_code=1)
        raise
    else:
        runner.close_logging()


if __name__ == "__main__":
    main()
