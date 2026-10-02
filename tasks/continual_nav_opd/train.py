"""Independent no-TA OPD training, stage-boundary resume and pure dry-run."""
import argparse
from dataclasses import asdict
import json
from .config import load_config, config_hash, budget_summary
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
    args=parser.parse_args()
    config=load_config(args.config)
    if args.device:
        from dataclasses import replace
        from .config import validate_config
        config=replace(config,training=replace(config.training,device=args.device)); validate_config(config)
    if not args.dry_run:
        if args.seed is None or args.run_dir is None: parser.error('--seed and --run-dir are required for training')
        import torch
        from .runner import run
        torch.set_num_threads(2)
        from .timing import timing_mode
        with timing_mode(args.timing_mode):
            result=run(config,args.seed,args.run_dir,args.resume,args.wandb_mode,args.max_stages)
        print(json.dumps({'next_index':result['next_index'],'finalized':result['finalized'],'global_env_steps':result['global_env_steps'],
                          'optimizer_updates':result['optimizer_updates']}))
        return
    print(json.dumps(dict(method=config.method,schema_version=config.schema_version,
                         sequence_protocol=config.sequence_protocol,config_hash=config_hash(config),
                         ticks_by_task=asdict(config.ctm.ticks_by_task),memory_ticks=config.ctm.memory_length,
                         budget=budget_summary(config),stages=[asdict(s) for s in expand_stages(config)]),indent=2))

if __name__=="__main__":
    main()
