#!/usr/bin/env python3
"""GUI for cyclically permuting one of the original 512 Diff-AE z_sem coordinates."""
import argparse, base64, io, json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import torch
from PIL import Image

HTML = r'''<!doctype html><html><head><meta charset="utf-8"><title>Raw AE feature permutation</title>
<style>body{font-family:system-ui;background:#111;color:#eee;margin:0;padding:20px}header{display:flex;gap:10px;align-items:center;flex-wrap:wrap}button,input{font:inherit;background:#222;color:#eee;border:1px solid #555;padding:7px 10px;border-radius:4px}button{cursor:pointer}#status{color:#aaa}#summary{margin:12px 0;color:#ccc}.grid{display:grid;grid-template-columns:repeat(8,minmax(120px,1fr));gap:10px;max-width:1800px}figure{margin:0;background:#1c1c1c;padding:8px}img{width:100%;display:block}figcaption{font-size:12px;color:#bbb;margin-top:6px}@media(max-width:700px){.grid{grid-template-columns:repeat(2,minmax(140px,1fr))}}</style></head>
<body><header><strong>Raw AE z_sem coordinate permutation</strong><span>importance rank</span><button id="prev">&#8592;</button><input id="feature" type="number" min="0" max="511" value="0"><button id="next">&#8594;</button><button id="load">Load</button><span id="status">ready</span></header><div id="summary"></div><div id="grid" class="grid"></div>
<script>const $=id=>document.getElementById(id);async function load(){let rank=Math.max(0,Math.min(511,parseInt($('feature').value||0)));$('feature').value=rank;$('status').textContent='decoding importance rank '+rank+'...';try{let r=await fetch('/api/feature?rank='+rank),d=await r.json();if(d.error)throw Error(d.error);$('summary').textContent='rank '+d.rank+' | coordinate '+d.feature+' | dataset std '+d.dataset_std.toFixed(5)+' | selected mean '+d.mean.toFixed(5)+' | selected std '+d.std.toFixed(5);$('grid').innerHTML=d.images.map((x,i)=>x.variants.map(v=>'<figure><img src="'+v.image+'"><figcaption>example '+i+' '+v.label+'</figcaption></figure>').join('')+'<figure><img src="'+x.heatmap+'"><figcaption>example '+i+' spatial effect</figcaption></figure>').join('');$('status').textContent='loaded'}catch(e){$('status').textContent='error: '+e.message}}$('load').onclick=load;$('feature').onchange=load;$('prev').onclick=()=>{$('feature').value=Math.max(0,parseInt($('feature').value||0)-1);load()};$('next').onclick=()=>{$('feature').value=Math.min(511,parseInt($('feature').value||0)+1);load()};load();</script></body></html>'''

def png(x):
    x=x.detach().float().cpu().clamp(0,1); b=io.BytesIO(); Image.fromarray((x.permute(1,2,0).numpy()*255).round().astype('uint8')).save(b,format='PNG'); return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()

class Server:
    def __init__(self,a):
        self.a=a; sys.path.insert(0,os.path.abspath(a.repo)); from experiment import LitModel; from templates import ffhq128_autoenc_130M
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); conf=ffhq128_autoenc_130M(); conf.pretrain=None; conf.latent_infer_path=None; self.lit=LitModel(conf)
        state=torch.load(a.checkpoint,map_location='cpu',weights_only=False); self.lit.load_state_dict(state['state_dict'],strict=False); self.lit.eval().to(self.device)
        state=torch.load(a.codes,map_location='cpu',weights_only=False); self.codes=state['conds'].float(); self.dataset_std=self.codes.std(0); self.order=torch.argsort(self.dataset_std,descending=True).tolist(); self.cache={}; self.lock=threading.Lock()

    @torch.no_grad()
    def feature(self,rank):
        rank=int(rank)
        if rank<0 or rank>=512: raise ValueError('rank must be in [0, 511]')
        f=self.order[rank]
        with self.lock:
            if f in self.cache:return self.cache[f]
            # Use examples with the largest magnitude in this raw coordinate.
            _, indices=torch.topk(self.codes[:,f].abs(),self.a.examples); indices=indices.tolist(); z=self.codes[indices].to(self.device)
            gen=torch.Generator().manual_seed(self.a.seed+f); noise=torch.randn(len(indices),3,128,128,generator=gen).to(self.device)
            sigma=z[:,f].std(); multipliers=(-10,-4,-2,0,2,4,10); variants=[]
            for mult in multipliers:
                z2=z.clone(); z2[:,f]=z[:,f]+mult*sigma
                variants.append((mult,self.lit.render(noise,cond=z2,T=20).clamp(0,1)))
            effect=(variants[-1][1]-variants[0][1]).abs(); effect=effect/effect.flatten(1).amax(1).clamp_min(1e-6)[:,None,None,None]
            out={'rank':rank,'feature':f,'dataset_std':float(self.dataset_std[f]),'examples':len(z),'mean':float(z[:,f].mean()),'std':float(sigma),'images':[{'variants':[{'label':f'{mult:+d} std','image':png(decoded[i])} for mult,decoded in variants],'heatmap':png(effect[i])} for i in range(len(z))]}; self.cache[f]=out; return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--repo',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--codes',required=True); p.add_argument('--port',type=int,default=8767); p.add_argument('--examples',type=int,default=12); p.add_argument('--seed',type=int,default=2026); a=p.parse_args(); s=Server(a)
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            q=urlparse(self.path)
            if q.path=='/': out=HTML.encode(); self.send_response(200); self.send_header('Content-Type','text/html')
            elif q.path=='/api/feature':
                try: out=json.dumps(s.feature(int(parse_qs(q.query).get('rank',['0'])[0]))).encode(); self.send_response(200)
                except Exception as e: out=json.dumps({'error':str(e)}).encode(); self.send_response(400)
                self.send_header('Content-Type','application/json')
            else: self.send_response(404); out=b''
            self.end_headers(); self.wfile.write(out)
        def log_message(self,*args): pass
    print(f'GUI ready at http://0.0.0.0:{a.port}',flush=True); ThreadingHTTPServer(('0.0.0.0',a.port),H).serve_forever()
if __name__=='__main__':main()
