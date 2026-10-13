"""python -m tasks.suffix_dual_expert.visualize [--config switch.yaml]."""
import argparse
import json
from pathlib import Path

import yaml

from ..envs import file_sha

PACKAGE=Path(__file__).parent
ROOT=PACKAGE.parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=PACKAGE/'config.yaml')
    parser.add_argument('--checkpoint',help='Override configured checkpoint path')
    parser.add_argument('--reuse',action='store_true',help='Reuse verified collected data; regenerate page')
    parser.add_argument('--serve-only',action='store_true',help='Serve existing output without inference')
    parser.add_argument('--no-serve',action='store_true',help='Generate files and exit')
    args=parser.parse_args()
    config=yaml.safe_load((PACKAGE/'config.yaml').read_text(encoding='utf-8'))
    config.update(yaml.safe_load(args.config.read_text(encoding='utf-8')))
    if args.checkpoint:
        config['checkpoint']=args.checkpoint
    if config['mode'] not in ('baseline','switch') or not 1<=config['max_actions']<=300:
        parser.error('mode must be baseline/switch; max_actions must be 1..300')
    if not 8<=config['switch_after_actions']<=config['max_actions'] or not 1<=config['probe_actions']<=20:
        parser.error('switch_after_actions must fill 40 ticks and fit max_actions; probe_actions must be 1..20')
    out=ROOT/config['output']
    if not args.serve_only:
        if args.reuse:
            summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
            if summary['config']!=config or summary['checkpoint_sha256']!=file_sha(ROOT/config['checkpoint']) or summary.get('cases_sha256')!=file_sha(PACKAGE/config['cases']):
                parser.error('reuse config/cases/checkpoint differs from saved data')
        else:
            # Never silently overwrite another experiment or leave a stale summary after failed collection.
            out.mkdir(parents=True,exist_ok=False)
            from .collect import collect
            cases_path=PACKAGE/config['cases']
            panel=json.loads(cases_path.read_text(encoding='utf-8'))
            summary=collect(config,ROOT,out,panel)
            summary['cases_sha256']=file_sha(cases_path)
            (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        from .page import build
        build(summary,out)
        print(out/'neuron_response_player.html',flush=True)
    if not args.no_serve:
        from .server import serve
        serve(out,config['port'])


if __name__=='__main__':
    main()
