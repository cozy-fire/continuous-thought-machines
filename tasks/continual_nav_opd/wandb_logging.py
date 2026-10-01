"""One event stream, with policy/source identities preserved in W&B namespaces."""
import json
from pathlib import Path
import time
from .checkpoint import atomic_json
from .config import resolved_dict,config_hash
from .timing import current_timing_mode

TRAIN_METRICS = {'kl', 'total_loss', 'agreement', 'student_entropy', 'grad_norm',
                 'global_env_steps', 'stage_env_steps', 'episode_mean_return', 'episode_success_rate'}
EVALUATION_METRICS = {'episodes', 'success_rate', 'mean_return', 'mean_length', 'environment_steps',
                      'displacement_rate', 'turn_rate', 'evaluation_seconds', 'global_env_steps'}


def chart_values(event: dict) -> dict:
    """Whitelist charts only; the complete event remains available in local JSONL."""
    evaluation = event.get('split') in ('validation', 'test')
    if evaluation:
        keys, list_key = EVALUATION_METRICS, 'action_counts'
    elif event.get('event') == 'training':
        keys, list_key = TRAIN_METRICS, 'action_fractions'
        if event.get('phase') == 'C':
            keys = keys | {'ewc'}
    elif event.get('event') == 'fisher':
        keys, list_key = {'global_env_steps', 'stage_env_steps'}, None
    else:
        return {}
    namespace = '/'.join(str(event.get(k, 'none')) for k in
                         ('family', 'phase', 'task', 'policy_type', 'active_source', 'evaluation_task', 'split'))
    values = {namespace+'/'+k: v for k, v in event.items()
              if k in keys and isinstance(v, (int, float)) and not isinstance(v, bool)}
    if list_key:
        for index, value in enumerate(event.get(list_key, [])):
            values[f'{namespace}/{list_key}/{index}'] = value
    return values


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
            values = chart_values(event)
            if values:
                self.run.log(values)

    def close(self,failed=False):
        if self.run is not None: self.run.finish(exit_code=1 if failed else 0)
