"""Loopback-only CPU UMAP refitting over saved responses."""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np


def serve(out, port):
    os.environ['NUMBA_CACHE_DIR']=str(out/'numba_cache')
    os.environ['NUMBA_NUM_THREADS']='1'
    import umap
    data=np.load(out/'neuron_umap.npz')
    default=dict(n_neighbors=20,min_dist=.1,n_components=2,metric='cosine')
    key=lambda params:json.dumps(params,sort_keys=True)
    cache={key(default):dict(coordinates=data['coordinates'].tolist(),params=default,elapsed_seconds=0,cached=True)}
    pool=ThreadPoolExecutor(max_workers=1)
    jobs={}

    def fit(params):
        k=key(params)
        if k not in cache:
            start=time.perf_counter()
            coords=umap.UMAP(**params,random_state=0,n_jobs=1).fit_transform(data['normalized'])
            if not np.isfinite(coords).all():
                raise ValueError('nonfinite UMAP')
            if len(cache)>=64:
                cache.pop(next(iter(cache)))
            cache[k]=dict(coordinates=coords.tolist(),params=params,elapsed_seconds=time.perf_counter()-start,cached=False)
        return cache[k]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass

        def reply(self,value,status=200):
            raw=json.dumps(value,allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(raw)))
            self.send_header('Cache-Control','no-store')
            self.end_headers();self.wfile.write(raw)

        def do_GET(self):
            route=self.path.split('?')[0]
            if route=='/api/health':
                self.reply(dict(status='ready',features=list(data['normalized'].shape)));return
            if route.startswith('/api/job/'):
                future=jobs.get(route.rsplit('/',1)[-1])
                if future is None:
                    self.reply(dict(error='unknown job'),404)
                elif not future.done():
                    self.reply(dict(status='running'))
                else:
                    try:self.reply(dict(status='complete',**future.result()))
                    except Exception as e:self.reply(dict(status='failed',error=str(e)))
                return
            if route not in ('/','/neuron_response_player.html'):
                self.reply(dict(error='not found'),404);return
            raw=(out/'neuron_response_player.html').read_bytes()
            self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store')
            self.end_headers();self.wfile.write(raw)

        def do_POST(self):
            origin=f'127.0.0.1:{self.server.server_port}'
            if self.path!='/api/fit' or self.headers.get('Host')!=origin or self.headers.get('Origin',f'http://{origin}')!=f'http://{origin}':
                self.reply(dict(error='origin or route rejected'),403);return
            try:
                size=int(self.headers.get('Content-Length',0))
                if not 0<size<=4096:raise ValueError('invalid body size')
                p=json.loads(self.rfile.read(size))
                if set(p)!=set(default) or type(p['n_neighbors']) is not int or not 2<=p['n_neighbors']<=511 or type(p['n_components']) is not int or not 1<=p['n_components']<=3 or type(p['min_dist']) not in (int,float) or not np.isfinite(p['min_dist']) or not 0<=p['min_dist']<=1 or p['metric'] not in ('cosine','correlation','euclidean','manhattan'):
                    raise ValueError('invalid UMAP parameters')
            except (ValueError,TypeError,KeyError) as e:
                self.reply(dict(error=str(e)),400);return
            # Only one running fit and one queued request; superseded browser revisions are ignored.
            for k in list(jobs):
                if jobs[k].done():
                    if len(jobs)>64:jobs.pop(k)
                elif not jobs[k].running():jobs[k].cancel()
            ident=str(time.time_ns());jobs[ident]=pool.submit(fit,p)
            self.reply(dict(job=ident),202)

    server=HTTPServer(('127.0.0.1',port),Handler)
    print(f'http://127.0.0.1:{port}/',flush=True)
    try:server.serve_forever()
    finally:server.server_close();pool.shutdown(cancel_futures=True)
