"""One joint neuron layout, fixed colors, and a shared player for both modes."""
import os
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


def build(summary: dict, out: Path) -> None:
    os.environ['NUMBA_CACHE_DIR'] = str(out/'numba_cache')
    os.environ['NUMBA_NUM_THREADS'] = '1'
    import umap

    assets = Path(__file__).parent/'assets'
    references = [np.load(out/(c['id']+'.npz')) for c in summary['reference_cases']]
    samples = min(100, min(len(d['post']) for d in references))
    features = np.concatenate([d['post'][np.linspace(0, len(d['post'])-1, samples).astype(int)] for d in references]).T
    normalized = (features-features.mean(1, keepdims=True))/np.maximum(features.std(1, keepdims=True), 1e-8)
    old = np.load(out/'neuron_umap.npz') if (out/'neuron_umap.npz').exists() else None
    layout = old['coordinates'] if old is not None and np.array_equal(old['features'], features) else umap.UMAP(
        n_neighbors=20, min_dist=.1, metric='cosine', random_state=0, n_jobs=1).fit_transform(normalized)
    np.savez_compressed(out/'neuron_umap.npz', coordinates=layout, features=features, normalized=normalized)
    low, high = np.percentile(features, [1, 99], axis=1)
    span = np.maximum(high-low, 1e-8)

    def png(a):
        stream = io.BytesIO()
        Image.fromarray(a).save(stream, format='PNG')
        return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()

    def responses(d):
        ids = d['attention_top20_indices']
        values = np.take_along_axis(d['attention_head_mean'].reshape(-1, 441), ids, axis=1)
        colors = np.rint(np.clip((d['post']-low)/span, 0, 1)*255).astype(np.uint8)
        return dict(colors=base64.b64encode(colors.tobytes()).decode(),
                    attention_top20=base64.b64encode(ids.astype('<u2').tobytes()).decode(),
                    attention_values=base64.b64encode(values.astype('<f4').tobytes()).decode())

    cases=[]
    for c in summary['cases']:
        d=np.load(out/(c['id']+'.npz'))
        view = dict(bounds=d['view_bounds'].tolist(), visible=d['view_visible'].astype(int).tolist(),
                    directions=d['view_directions'].tolist()) if c['task']=='fourrooms' else None
        cases.append(dict(c, **responses(d), observations=[png(v.transpose(1,2,0)) for v in d['observations']],
                          worlds=[png(v) for v in d['worlds']], actions=d['actions'].tolist(),
                          positions=d['positions'].tolist(), view=view))
    probes=[]
    for meta in summary['probes']:
        d=np.load(out/meta['file'])
        probes.append(dict(observations=[png(v.transpose(1,2,0)) for v in d['observations']],
            **{mode:responses({k:d[mode+'_'+k] for k in ['post','attention_head_mean','attention_top20_indices']})
               for mode in ['fresh','maze_history']}))
    payload=dict(layout=layout.tolist(), cases=cases, probes=probes, results=summary['results'],
                 prefix=summary['prefix'], config=summary['config'])
    switch=summary['config']['mode']=='switch'
    title='Maze历史 → FourRooms · 切换响应对照' if switch else '双专家KB · 神经元响应模式'
    description='来源Maze运行至配置边界后继承完整pre/post历史，分别切换到FourRooms；同seed全新初态为对照。' if switch else '固定Maze/FourRooms案例逐tick响应；更换权重后展示实际成功或失败结果，不重新筛选成功案例。'
    html=(assets/'player.html').read_text(encoding='utf-8')
    for key,value in dict(TITLE=title,DESCRIPTION=description,SHA=summary['checkpoint_sha256'],
                          PAYLOAD=json.dumps(payload, ensure_ascii=False, separators=(',',':'), allow_nan=False)).items():
        html=html.replace('__'+key+'__',value)
    scripts=['live_controls.js','action_summary.js']+(['switch_controls.js'] if switch else [])
    html=html.replace('</script></html>','</script>'+''.join('<script>'+(assets/name).read_text(encoding='utf-8')+'</script>' for name in scripts)+'</html>')
    (out/'neuron_response_player.html').write_text(html,encoding='utf-8')
