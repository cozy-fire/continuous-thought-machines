"""CPU-only fixed-panel workers; the parent owns all neural inference and traces."""
from __future__ import annotations

from collections import deque
import multiprocessing as mp
import traceback

from .maze import MazeEnv
from .fourrooms import FourRoomsEnv


def shortest_path(env: MazeEnv) -> int:
    queue, seen = deque([(env.agent_pos, 0)]), {env.agent_pos}
    while queue:
        pos, distance = queue.popleft()
        if pos == env.goal_pos:
            return distance
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = pos[0]+dr, pos[1]+dc
            if 0 <= nxt[0] < 19 and 0 <= nxt[1] < 19 and not env.walls[nxt] and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, distance+1))
    raise ValueError("unreachable evaluation maze")


def create_panel_env(spec: tuple):
    task, split, root, entry = spec
    if task == "maze_medium":
        return MazeEnv(root, (entry,))
    if task == "fourrooms":
        return FourRoomsEnv(split=split, evaluation_seeds=(entry,))
    raise ValueError("unknown evaluation task")


def _command(env, command: str, value, factory=create_panel_env):
    if command == "reset":
        if env is not None:
            env.close()
        env = factory(value)
        try:
            obs, _ = env.reset()
            distance = shortest_path(env) if value[0] == "maze_medium" else None
        except BaseException:
            env.close()
            raise
        return env, (obs, distance)
    if command == "step" and env is not None:
        return env, env.step(value)
    raise ValueError("invalid evaluation worker command")


def _worker(connection, factory) -> None:
    env = None
    try:
        while True:
            command, value = connection.recv()
            if command == "close":
                break
            env, result = _command(env, command, value, factory)
            connection.send((True, result))
    except EOFError:
        pass
    except BaseException:
        connection.send((False, traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        connection.close()


class EvaluationPool:
    """One persistent process per slot, with no autoreset or speculative env steps."""
    def __init__(self, size: int, backend: str, *, factory=create_panel_env):
        if size < 1 or backend not in ("serial", "subprocess"):
            raise ValueError("invalid evaluation pool")
        self.backend = backend
        self.factory = factory
        self.envs = [None] * size
        self.connections, self.processes = [], []
        try:
            if backend == "subprocess":
                context = mp.get_context("spawn")
                for _ in range(size):
                    parent, child = context.Pipe()
                    process = context.Process(target=_worker, args=(child, factory), daemon=True)
                    try:
                        process.start()
                    except BaseException:
                        parent.close()
                        raise
                    finally:
                        child.close()
                    self.connections.append(parent)
                    self.processes.append(process)
        except BaseException:
            self.close()
            raise

    def exchange(self, command: str, requests: dict[int, object]) -> dict:
        if self.backend == "serial":
            results = {}
            for slot, value in requests.items():
                self.envs[slot], results[slot] = _command(self.envs[slot], command, value, self.factory)
            return results
        try:
            # Dispatch to all workers before waiting: CPU environment steps overlap.
            for slot, value in requests.items():
                self.connections[slot].send((command, value))
            results = {}
            for slot in requests:
                if not self.connections[slot].poll(60):
                    raise RuntimeError(f"evaluation worker {slot} timed out")
                ok, result = self.connections[slot].recv()
                if not ok:
                    raise RuntimeError(f"evaluation worker {slot} failed:\n{result}")
                results[slot] = result
            return results
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
            connection.close()
        for process in self.processes:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        for env in self.envs:
            if env is not None:
                env.close()
        self.connections.clear()
        self.processes.clear()
        self.envs = [None] * len(self.envs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
