"""One event stream, with policy/source identities preserved in W&B namespaces."""
import json
from pathlib import Path
import time
from .checkpoint import atomic_json
from .config import resolved_dict,config_hash
from .timing import current_timing_mode


class EventLogger:
    def __init__(self,root,config,seed,mode='disabled',resume=False):
        if mode not in ('disabled','offline','online'): raise ValueError('invalid W&B mode')
        self.root=Path(root); self.run=None
        self.identity={'method':config.method,'schema_version':config.schema_version,'sequence_protocol':config.sequence_protocol,'seed':seed,'timing_mode':current_timing_mode()}
        if mode!='disabled':
            import wandb
            path=self.root/'wandb_run.json'
            old=json.loads(path.read_text()) if resume and path.exists() else {}
            self.run=wandb.init(project='ctm-pnc-opd',dir=str(self.root),mode=mode,
                                id=old.get('id'),resume='must' if old and mode=='online' else None,
                                config={**resolved_dict(config),'run_seed':seed,'config_hash':config_hash(config)})
            atomic_json(path,{'id':self.run.id,'url':self.run.url,'mode':mode,'project':self.run.project})

    def emit(self,event):
        event={'time_unix':time.time(),**self.identity,**event}
        text=json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n'
        for name in ('events.jsonl','metrics.jsonl'):
            with (self.root/name).open('a',encoding='utf-8') as stream: stream.write(text)
        if self.run is not None:
            namespace='/'.join(str(event.get(k,'none')) for k in ('family','phase','task','policy_type','active_source','evaluation_task','split'))
            values={namespace+'/'+k:v for k,v in event.items() if isinstance(v,(int,float)) and not isinstance(v,bool)}
            # Preserve nonnumeric identities as well as chart values; unique IDs remain traceable.
            values.update({namespace+'/'+k:v for k,v in event.items() if isinstance(v,str)})
            for key in ('action_counts',):
                for index,value in enumerate(event.get(key,[])): values[f'{namespace}/{key}/{index}']=value
            self.run.log(values)

    def close(self,failed=False):
        if self.run is not None: self.run.finish(exit_code=1 if failed else 0)
