"""Medium Maze and partial-RGB FourRooms with explicit portable episode state."""
from collections import deque
from pathlib import Path
import hashlib
import json

import gymnasium as gym
import minigrid  # Environment registration.
import numpy as np
from PIL import Image
from minigrid.wrappers import RGBImgPartialObsWrapper

MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))


def file_sha(file: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(file).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pixels(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(Image.fromarray(rgb).resize((84, 84), Image.Resampling.NEAREST))
    return np.ascontiguousarray(image.transpose(2, 0, 1))


def action_value(action: int) -> int:
    if not isinstance(action, (int, np.integer)) or not 0 <= action < 5:
        raise ValueError('action must be an integer in [0,4]')
    return int(action)


def training_entries(root: Path, count: int) -> list[dict]:
    paths = sorted((root/'train').rglob('*.png'), key=lambda p: tuple(
        int(t) if t.isdigit() else t for t in p.relative_to(root).with_suffix('').parts))
    entries, seen = [], set()
    for p in paths:
        rel = p.relative_to(root)
        if len(rel.parts) != 3 or not rel.parts[1].isdigit() or not p.stem.isdigit():
            continue
        digest = file_sha(p)
        if digest in seen:
            continue
        seen.add(digest)
        entries.append({'path': rel.as_posix(), 'sha256': digest})
        if len(entries) == count:
            return entries
    raise ValueError(f'only {len(entries)}/{count} unique training maps available')


def evaluation_entries(file: Path, count: int, train: list[dict]) -> list[dict]:
    entries = json.loads(file.read_text(encoding='utf-8'))
    if len(entries) < count or any(not e['path'].startswith('test/') for e in entries):
        raise ValueError('test manifest must contain the fixed test panel')
    hashes = [e['sha256'] for e in entries]
    if len(hashes) != len(set(hashes)) or set(hashes) & {e['sha256'] for e in train}:
        raise ValueError('duplicate test map or train/test overlap')
    return entries[:count]


class MapCache:
    def __init__(self, root: Path, entries: list[dict]):
        self.root, self.entries = root, entries
        self.maps = {}
        for entry in entries:
            p = (root/entry['path']).resolve()
            if not p.is_relative_to(root.resolve()) or file_sha(p) != entry['sha256']:
                raise ValueError('invalid map identity: ' + entry['path'])
            with Image.open(p) as image:
                rgb = np.array(image.convert('RGB'))
            if rgb.shape != (19, 19, 3):
                raise ValueError('Maze medium requires 19x19 maps')
            allowed = {(0,0,0), (255,255,255), (255,0,0), (0,255,0), (0,0,255)}
            if any(tuple(c) not in allowed for c in np.unique(rgb.reshape(-1,3), axis=0)):
                raise ValueError('unknown Maze color')
            starts, goals = (np.argwhere(np.all(rgb == c, axis=-1)) for c in ((255,0,0), (0,255,0)))
            if len(starts) != 1 or len(goals) != 1:
                raise ValueError('map needs exactly one start and goal')
            start, goal = tuple(map(int, starts[0])), tuple(map(int, goals[0]))
            rgb[np.all(rgb == (0,0,255), axis=-1)] = (255,255,255)
            rgb[start] = (255,255,255)
            walls = np.all(rgb == 0, axis=-1)
            distance = np.full((19,19), -1, dtype=np.int64)
            distance[goal] = 0
            queue = deque([goal])
            while queue:
                row, col = queue.popleft()
                for dr, dc in MOVES[:4]:
                    nxt = row+dr, col+dc
                    if 0 <= nxt[0] < 19 and 0 <= nxt[1] < 19 and not walls[nxt] and distance[nxt] < 0:
                        distance[nxt] = distance[row,col]+1
                        queue.append(nxt)
            if (distance[~walls] < 0).any():
                raise ValueError('unreachable traversable state')
            self.maps[entry['sha256']] = (rgb, walls, start, goal, distance)

    def view(self, entry: dict, position: tuple[int,int]) -> np.ndarray:
        image = self.maps[entry['sha256']][0].copy()
        image[position] = (255,0,0)
        return pixels(image)

    def tables(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Exact student-driven state transitions and deterministic BFS labels."""
        images, labels, map_ids, cells = [], [], [], []
        for index, entry in enumerate(self.entries):
            _, walls, _, goal, distance = self.maps[entry['sha256']]
            for raw in np.argwhere(~walls):
                pos = tuple(map(int, raw))
                if pos == goal:
                    continue
                # First distance-decreasing neighbor equals FIFO BFS tie breaking.
                target = next(a for a, (dr,dc) in enumerate(MOVES[:4])
                              if 0 <= pos[0]+dr < 19 and 0 <= pos[1]+dc < 19
                              and distance[pos[0]+dr,pos[1]+dc] == distance[pos]-1)
                images.append(self.view(entry, pos)); labels.append(target)
                map_ids.append(index); cells.append(pos)
        lookup = {(m,p): i for i,(m,p) in enumerate(zip(map_ids,cells))}
        transitions = np.empty((len(cells),5), dtype=np.int64)
        for i,(m,pos) in enumerate(zip(map_ids,cells)):
            _, walls, _, goal, _ = self.maps[self.entries[m]['sha256']]
            for a,(dr,dc) in enumerate(MOVES):
                nxt = pos[0]+dr,pos[1]+dc
                if not (0 <= nxt[0] < 19 and 0 <= nxt[1] < 19) or walls[nxt]:
                    nxt = pos
                transitions[i,a] = -1 if nxt == goal else lookup[m,nxt]
        return np.stack(images), np.asarray(labels), transitions, np.asarray(map_ids)


class MazeEnv:
    task = 'maze_medium'

    def __init__(self, cache: MapCache, entries: list[dict], seed: int):
        self.cache, self.entries = cache, entries
        self.rng = np.random.default_rng(seed)
        self.done = True
        self.episode = -1

    def reset(self) -> np.ndarray:
        self.entry = self.entries[int(self.rng.integers(len(self.entries)))]
        _, self.walls, self.position, self.goal, _ = self.cache.maps[self.entry['sha256']]
        self.steps, self.done = 0, False
        self.episode += 1
        return self.observe()

    def observe(self) -> np.ndarray:
        return self.cache.view(self.entry, self.position)

    def step(self, action: int) -> tuple[np.ndarray, bool, bool]:
        a = action_value(action)
        if self.done:
            raise RuntimeError('reset required before step')
        dr, dc = MOVES[a]
        nxt = self.position[0]+dr, self.position[1]+dc
        if 0 <= nxt[0] < 19 and 0 <= nxt[1] < 19 and not self.walls[nxt]:
            self.position = nxt
        self.steps += 1
        success = self.position == self.goal
        self.done = success or self.steps >= 300
        return self.observe(), self.done, success

    def capture(self) -> dict:
        return dict(map_sha=self.entry['sha256'], position=list(self.position), steps=self.steps,
                    done=self.done, episode=self.episode, rng=self.rng.bit_generator.state)

    def restore(self, value: dict) -> np.ndarray:
        self.entry = next(e for e in self.entries if e['sha256'] == value['map_sha'])
        _, self.walls, _, self.goal, _ = self.cache.maps[self.entry['sha256']]
        self.position = tuple(value['position'])
        self.steps, self.done, self.episode = value['steps'], value['done'], value['episode']
        self.rng.bit_generator.state = value['rng']
        return self.observe()

    def close(self) -> None:
        pass


class FourRoomsEnv:
    task = 'fourrooms'

    def __init__(self, seed: int, evaluation_seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.evaluation_seed = evaluation_seed
        self.native = RGBImgPartialObsWrapper(gym.make('MiniGrid-FourRooms-v0', max_steps=300,
                                                       render_mode='rgb_array'), tile_size=8)
        self.done, self.episode = True, -1

    @property
    def position(self) -> tuple[int,int]:
        return tuple(map(int, self.native.unwrapped.agent_pos))

    def reset(self) -> np.ndarray:
        self.episode_seed = self.evaluation_seed if self.evaluation_seed is not None else int(self.rng.integers(1000000))
        raw, _ = self.native.reset(seed=self.episode_seed)
        self.done = False
        self.episode += 1
        return pixels(raw['image'])

    def observe(self) -> np.ndarray:
        raw = self.native.unwrapped.gen_obs()
        return pixels(self.native.observation(raw)['image'])

    def step(self, action: int) -> tuple[np.ndarray,bool,bool]:
        a = action_value(action)
        if self.done:
            raise RuntimeError('reset required before step')
        raw, _, terminated, truncated, _ = self.native.step(a if a < 3 else 6)
        self.done = bool(terminated or truncated)
        return pixels(raw['image']), self.done, bool(terminated)

    def capture(self) -> dict:
        env = self.native.unwrapped
        return dict(seed=self.episode_seed, position=list(self.position), direction=int(env.agent_dir),
                    steps=int(env.step_count), episode=self.episode, done=self.done,
                    rng=self.rng.bit_generator.state, native_rng=env.np_random.bit_generator.state)

    def restore(self, value: dict) -> np.ndarray:
        # Only left/right/forward/no-op are exposed: the layout never changes within an episode.
        self.native.reset(seed=value['seed'])
        env = self.native.unwrapped
        env.agent_pos = tuple(value['position']); env.agent_dir = value['direction']; env.step_count = value['steps']
        env.np_random.bit_generator.state = value['native_rng']
        self.rng.bit_generator.state = value['rng']
        self.episode_seed, self.episode, self.done = value['seed'], value['episode'], value['done']
        return self.observe()

    def close(self) -> None:
        self.native.close()


class BatchEnv:
    def __init__(self, envs: list[MazeEnv | FourRoomsEnv]):
        self.envs = envs
        self.task = envs[0].task
        if any(e.task != self.task for e in envs):
            raise ValueError('one task per environment batch')

    def reset(self) -> np.ndarray:
        return np.stack([e.reset() for e in self.envs])

    def step(self, actions: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        images, starts = [], []
        for env, action in zip(self.envs, actions):
            image, done, _ = env.step(int(action))
            images.append(env.reset() if done else image)
            starts.append(done)
        return np.stack(images), np.asarray(starts, dtype=bool)

    def capture(self) -> list[dict]:
        return [e.capture() for e in self.envs]

    def restore(self, values: list[dict]) -> np.ndarray:
        if len(values) != len(self.envs):
            raise ValueError('environment slot count mismatch')
        return np.stack([e.restore(v) for e,v in zip(self.envs,values)])

    def close(self) -> None:
        for env in self.envs:
            env.close()
