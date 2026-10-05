"""Pure stage expansion; boundaries are structural, never name suffixes."""
from dataclasses import dataclass
from .config import Config, validate_config

@dataclass(frozen=True)
class Stage:
    key: str
    family: str
    visit: int
    task: str
    phase: str
    env_steps: int
    ticks: int
    optimizer_updates: int = 0  # Nonzero only for Maze P; env_steps is its capacity.

def expand_stages(config: Config) -> tuple[Stage,...]:
    validate_config(config)
    result=[]
    for visit in range(config.pnc.visits):
        for task in config.task_order:
            budget=getattr(config.pnc.task_budgets,task)
            for phase,steps in (("P",budget.progress_steps),("C",budget.compress_steps),("F",config.fisher.collect_steps)):
                updates = sum(config.maze_progress.pool_updates) if phase == 'P' and task == 'maze_medium' else 0
                if updates:
                    steps = updates*config.maze_progress.sequences_per_update*config.maze_progress.decisions
                result.append(Stage(f"pnc/v{visit}/{task}/{phase}","pnc",visit,task,phase,steps,
                                    config.ctm.ticks_by_task.for_task(task),updates))
    return tuple(result)
