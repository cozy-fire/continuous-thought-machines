"""OPD training followed by independent evaluation by default; pure dry-run."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import subprocess
import sys
from .config import REPO_ROOT, load_config, config_hash, budget_summary
from .schedule import expand_stages

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument('--seed',type=int)
    parser.add_argument('--run-dir')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--wandb-mode',choices=['disabled','offline','online'],default='disabled')
    parser.add_argument('--max-stages',type=int)
    parser.add_argument('--timing-mode',choices=['events','synchronized'],default='events')
    parser.add_argument('--device',help='Explicit runtime/config device override, included in config hash')
    parser.add_argument('--skip-evaluation',action='store_true',help='Train/export only; do not run evaluation after finalization')
    parser.add_argument('--evaluation-dir',help='Independent evaluation output, defaults to <run-dir>_evaluation; matching results are reused')
    args=parser.parse_args()
    config=load_config(args.config)
    if args.device:
        from dataclasses import replace
        from .config import validate_config
        config=replace(config,training=replace(config.training,device=args.device)); validate_config(config)
    if not args.dry_run:
        if args.seed is None or args.run_dir is None: parser.error('--seed and --run-dir are required for training')
        run_root=Path(args.run_dir).resolve()
        evaluation_dir=Path(args.evaluation_dir).resolve() if args.evaluation_dir else run_root.with_name(run_root.name+'_evaluation')
        if not args.skip_evaluation and (evaluation_dir.is_relative_to(run_root) or run_root.is_relative_to(evaluation_dir)):
            parser.error('--evaluation-dir must be outside the training run')
        import torch
        from .runner import run
        torch.set_num_threads(2)
        from .timing import timing_mode
        with timing_mode(args.timing_mode):
            result=run(config,args.seed,args.run_dir,args.resume,args.wandb_mode,args.max_stages)
        print(json.dumps({'next_index':result['next_index'],'finalized':result['finalized'],'global_env_steps':result['global_env_steps'],
                          'optimizer_updates':result['optimizer_updates']}),flush=True)
        if result['finalized'] and not args.skip_evaluation:
            # The evaluator is a separate process. It cannot alter training status or resume state.
            del result; gc.collect()
            if torch.cuda.is_initialized(): torch.cuda.empty_cache()
            subprocess.run([sys.executable,'-m','tasks.continual_nav_opd.evaluate_stages',
                            '--stage-index',str(run_root/'exports/stages.json'),'--output-dir',str(evaluation_dir),
                            '--device',config.training.device],cwd=REPO_ROOT,check=True)
        return
    print(json.dumps(dict(method=config.method,schema_version=config.schema_version,
                         sequence_protocol=config.sequence_protocol,config_hash=config_hash(config),
                         ticks_by_task=asdict(config.ctm.ticks_by_task),memory_ticks=config.ctm.memory_length,
                         evaluation_in_training=False,evaluation_after_training=not args.skip_evaluation,
                         stage_snapshot_count=len(expand_stages(config)),
                         budget=budget_summary(config),stages=[asdict(s) for s in expand_stages(config)]),indent=2))

if __name__=="__main__":
    main()
