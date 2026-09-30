"""Deterministic, uncached shortest-path labels on the actual agent position."""
from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))


def shortest_path(walls, start, goal):
    walls = np.asarray(walls)
    if walls.ndim != 2 or walls.dtype != np.bool_:
        raise ValueError("walls must be a boolean 2D grid")
    height, width = walls.shape
    for pos in (start, goal):
        if len(pos) != 2 or not all(isinstance(p, (int, np.integer)) for p in pos):
            raise ValueError("positions must be integer pairs")
        if not (0 <= pos[0] < height and 0 <= pos[1] < width) or walls[pos]:
            raise ValueError("start/goal must be inside a traversable cell")
    if start == goal:
        return []
    queue, parents = deque([start]), {start: None}
    while queue:
        pos = queue.popleft()
        for action, (dr, dc) in enumerate(MOVES):
            nxt = (pos[0] + dr, pos[1] + dc)
            if not (0 <= nxt[0] < height and 0 <= nxt[1] < width):
                continue
            if walls[nxt] or nxt in parents:
                continue
            # Mark on enqueue: the first parent fixes the FIFO/tie-break shortest path.
            parents[nxt] = (pos, action)
            if nxt == goal:
                path = []
                while parents[nxt] is not None:
                    nxt, edge = parents[nxt]
                    path.append(edge)
                return path[::-1]
            queue.append(nxt)
    raise ValueError("goal is unreachable")


def parse_observation(observation):
    if not isinstance(observation, np.ndarray) or observation.dtype != np.uint8 or observation.shape != (3, 19, 19):
        raise ValueError("Maze teacher expects uint8 [3,19,19]")
    rgb = observation.transpose(1, 2, 0)
    allowed = {(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0)}
    if any(tuple(c) not in allowed for c in np.unique(rgb.reshape(-1, 3), axis=0)):
        raise ValueError("unknown color (answer-path blue is forbidden)")
    start = np.argwhere(np.all(rgb == (255, 0, 0), axis=-1))
    goal = np.argwhere(np.all(rgb == (0, 255, 0), axis=-1))
    if len(start) != 1 or len(goal) != 1:
        raise ValueError("live observation requires exactly one agent and goal")
    return np.all(rgb == 0, axis=-1), tuple(map(int, start[0])), tuple(map(int, goal[0]))


@dataclass(frozen=True)
class MazeTarget:
    action: np.ndarray
    probabilities: np.ndarray
    valid_target: np.ndarray
    distance_to_goal: np.ndarray


class MazeTeacher:
    def predict(self, observations, contexts):
        if not isinstance(observations, np.ndarray) or observations.shape[1:] != (3, 19, 19) or observations.ndim != 4 or len(observations) == 0:
            raise ValueError("Maze batch must be nonempty [B,3,19,19]")
        if len(contexts) != len(observations):
            raise ValueError("each observation needs map/episode context")
        actions, distances = [], []
        for image, context in zip(observations, contexts):
            try:
                walls, start, goal = parse_observation(image)
                path = shortest_path(walls, start, goal)
                if not path:
                    raise ValueError("terminal state cannot supply an action label")
            except ValueError as exc:
                raise ValueError(f"map={context.get('map_sha256', 'unknown')} "
                                 f"episode={context.get('episode_id', 'unknown')}: {exc}") from exc
            actions.append(path[0])
            distances.append(len(path))
        action = np.asarray(actions, dtype=np.int64)
        return MazeTarget(action, np.eye(5, dtype=np.float32)[action],
                          np.ones(len(action), dtype=bool), np.asarray(distances, dtype=np.int64))

    def descriptor(self):
        return {"type": "shortest_path_bfs", "algorithm_version": "bfs_grid_v1",
                "connectivity": 4, "edge_cost": 1,
                "actions": ["up", "down", "left", "right", "wait"],
                "tie_break_order": [0, 1, 2, 3], "target_distribution": "one_hot",
                "input_protocol": "uint8_B3_19_19_black_wall_white_floor_red_agent_green_goal",
                "implementation": {"path": "tasks/continual_nav_opd/teachers/maze.py",
                                   "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}

    def descriptor_bytes(self):
        return json.dumps(self.descriptor(), sort_keys=True, separators=(",", ":")).encode("utf-8")

    @property
    def source_snapshot_id(self):
        return hashlib.sha256(self.descriptor_bytes()).hexdigest()

    def save_descriptor(self, path):
        with Path(path).open("xb") as stream:
            stream.write(self.descriptor_bytes())
