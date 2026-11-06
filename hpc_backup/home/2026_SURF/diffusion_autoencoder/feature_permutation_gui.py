#!/usr/bin/env python3
"""On-demand GUI for permuting one z_sem SAE feature across fixed examples."""

import argparse
import base64
import html
import io
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import torch
from PIL import Image
from torchvision.utils import make_grid


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>__LABEL__ z_sem SAE feature permutation</title>
<style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:20px}
header{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:16px}
button,input{font:inherit;background:#222;color:#eee;border:1px solid #555;padding:7px 10px;border-radius:4px}
button{cursor:pointer}button:hover{background:#333} input{width:100px}
#status{color:#aaa} #summary{margin:10px 0;color:#ccc}
.grid{display:grid;grid-template-columns:repeat(8,minmax(120px,1fr));gap:10px;max-width:1800px}
figure{margin:0;background:#1c1c1c;padding:8px;border-radius:5px} img{width:100%;display:block}
figcaption{font-size:12px;color:#bbb;margin-top:6px}
@media(max-width:700px){.grid{grid-template-columns:repeat(2,minmax(140px,1fr))}}
</style></head><body>
<header><strong>__LABEL__ z_sem SAE feature permutation</strong>
<span>importance rank</span><button id="prev">&#8592;</button><input id="feature" type="number" min="0" value="0">
<button id="next">&#8594;</button><button id="load">Load</button>
<span id="status">ready</span></header>
<div id="summary"></div><div id="grid" class="grid"></div>
<script>
const $=id=>document.getElementById(id);
async function load(){let rank=Math.max(0,parseInt($('feature').value||0));$('status').textContent='decoding importance rank '+rank+'...';
 try{let r=await fetch('/api/feature?rank='+rank);let d=await r.json();if(d.error)throw Error(d.error);
 $('summary').textContent='rank '+d.rank+' | feature '+d.feature+' | dataset mean '+d.dataset_mean_activation.toFixed(5)+' | dataset std '+d.dataset_std_activation.toFixed(5)+' | active '+(100*d.dataset_active_fraction).toFixed(2)+'% | selected active '+d.selected_active+' / '+d.examples;
 $('grid').innerHTML=d.images.map((x,i)=>x.variants.map(v=>'<figure><img src="'+v.image+'"><figcaption>example '+i+' '+v.label+'</figcaption></figure>').join('')+'<figure><img src="'+x.heatmap+'"><figcaption>example '+i+' spatial effect (+2σ vs -2σ)</figcaption></figure>').join('');$('status').textContent='loaded';
 }catch(e){$('status').textContent='error: '+e.message}}
$('load').onclick=load;$('feature').onchange=load;$('prev').onclick=()=>{$('feature').value=Math.max(0,parseInt($('feature').value||0)-1);load()};$('next').onclick=()=>{$('feature').value=parseInt($('feature').value||0)+1;load()};load();
</script></body></html>"""


def png_data(image, scale=1.0):
    image = image.detach().float().cpu().clamp(0, 1)
    buf = io.BytesIO()
    Image.fromarray((image.permute(1, 2, 0).numpy() * 255).round().astype("uint8")).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class FeatureServer:
    def __init__(self, args):
        self.args = args
        sys.path.insert(0, os.path.abspath(args.repo))
        from experiment import LitModel
        from templates import ffhq128_autoenc_130M

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        conf = ffhq128_autoenc_130M()
        conf.pretrain = None
        conf.latent_infer_path = None
        self.lit = LitModel(conf)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        self.lit.load_state_dict(state["state_dict"], strict=False)
        self.lit.eval().to(self.device)

        sae_state = torch.load(args.sae, map_location="cpu", weights_only=False)
        model_state = sae_state.get("model", sae_state)
        self.input_mean = sae_state["input_mean"].float().to(self.device)
        self.input_std = sae_state["input_std"].float().to(self.device)
        from train_zsem_sae import TopKSAE
        config = sae_state.get("config", {})
        self.sae = TopKSAE(512, int(config.get("dict_mult", 16)) * 512,
                           int(config.get("k", 32))).to(self.device)
        self.sae.load_state_dict(model_state, strict=True)
        self.sae.eval()
        codes_state = torch.load(args.codes, map_location="cpu", weights_only=False)
        codes = codes_state["conds"].float()
        # Rank by activation mass over the full latent corpus. This is a usage
        # ranking, not a causal ranking; the page still reports the raw ID.
        sums = torch.zeros(self.sae.dict_size, device=self.device)
        squares = torch.zeros(self.sae.dict_size, device=self.device)
        active = torch.zeros(self.sae.dict_size, device=self.device)
        with torch.no_grad():
            for start in range(0, len(codes), args.batch_size):
                batch = ((codes[start:start + args.batch_size].to(self.device)
                          - self.input_mean) / self.input_std)
                pre = self.sae.encoder(batch - self.sae.dec_bias)
                acts = torch.relu(pre)
                vals, idx = torch.topk(acts, self.sae.k, dim=-1)
                sums.scatter_add_(0, idx.reshape(-1), vals.reshape(-1))
                squares.scatter_add_(0, idx.reshape(-1), (vals * vals).reshape(-1))
                active.scatter_add_(0, idx.reshape(-1), torch.ones_like(vals).reshape(-1))
        self.dataset_mean_activation = sums / len(codes)
        self.dataset_std_activation = (squares / len(codes) - self.dataset_mean_activation.square()).clamp_min(0).sqrt()
        self.dataset_active_fraction = active / len(codes)
        self.feature_order = torch.argsort(self.dataset_mean_activation, descending=True).tolist()
        self.codes = codes
        gen = torch.Generator().manual_seed(args.seed)
        self.cache = {}
        self.lock = threading.Lock()
        self.max_feature = self.sae.dict_size - 1

    @torch.no_grad()
    def feature(self, rank, require_inactive=False):
        rank = int(rank)
        if rank < 0 or rank >= len(self.feature_order):
            raise ValueError(f"rank must be in [0, {len(self.feature_order) - 1}]")
        feature = self.feature_order[rank]
        if feature < 0 or feature > self.max_feature:
            raise ValueError(f"feature must be in [0, {self.max_feature}]")
        with self.lock:
            if feature in self.cache:
                return self.cache[feature]
            # Select examples by pre-TopK score, not only by TopK membership.
            # This lets us test an inactive feature by support-augmenting it.
            candidates = []
            for start in range(0, len(self.codes), 4096):
                raw = self.codes[start:start + 4096].to(self.device)
                x_batch = (raw - self.input_mean) / self.input_std
                pre_batch = self.sae.encoder(x_batch - self.sae.dec_bias)
                scores = pre_batch[:, feature]
                if require_inactive:
                    top_idx = torch.topk(torch.relu(pre_batch), self.sae.k, dim=-1).indices
                    valid = ~(top_idx == feature).any(dim=1)
                else:
                    valid = torch.ones(len(scores), dtype=torch.bool, device=scores.device)
                for row in range(len(scores)):
                    if valid[row]:
                        candidates.append((float(scores[row]), int(start + row)))
            candidates.sort(reverse=True)
            selected = [idx for _, idx in candidates[:self.args.examples]]
            z_raw = self.codes[selected].to(self.device)
            gen = torch.Generator().manual_seed(self.args.seed + feature)
            x_T = torch.randn(len(selected), 3, 128, 128, generator=gen).to(self.device)
            x = (z_raw - self.input_mean) / self.input_std
            pre = self.sae.encoder(x - self.sae.dec_bias)
            acts = torch.relu(pre)
            vals, idx = torch.topk(acts, self.sae.k, dim=-1)
            coeff = torch.zeros_like(acts)
            coeff.scatter_(1, idx, vals)
            clean = coeff[:, feature]
            sigma = self.dataset_std_activation[feature]
            naturally_active = (coeff[:, feature] > 0)
            replace_pos = vals.argmin(dim=1)
            replace_feature = idx.gather(1, replace_pos[:, None]).squeeze(1)
            replace_value = vals.gather(1, replace_pos[:, None]).squeeze(1)
            multipliers = (-10, -4, -2, 0, 2, 4, 10)
            variants = []
            for mult in multipliers:
                if mult == 0:
                    z_mod_raw = z_raw
                else:
                    # Apply only the requested coefficient change. If the
                    # feature was outside TopK and is being turned on, replace
                    # the weakest selected feature to keep exactly k slots.
                    target = torch.clamp(clean + mult * sigma, min=0)
                    delta = (target - clean)[:, None] * self.sae.decoder[feature]
                    support_added = (~naturally_active) & (target > 0)
                    if support_added.any():
                        delta[support_added] -= (
                            replace_value[support_added, None]
                            * self.sae.decoder[replace_feature[support_added]]
                        )
                    z_mod_raw = z_raw + delta * self.input_std
                decoded = self.lit.render(x_T, cond=z_mod_raw, T=20).clamp(0, 1)
                variants.append((mult, decoded))
            effect = (variants[-1][1] - variants[0][1]).abs()
            effect = effect / effect.flatten(1).amax(1).clamp_min(1e-6)[:, None, None, None]
            result = {"rank": rank, "feature": feature, "examples": len(z_raw),
                      "dataset_mean_activation": float(self.dataset_mean_activation[feature]),
                      "dataset_active_fraction": float(self.dataset_active_fraction[feature]),
                      "dataset_std_activation": float(sigma),
                      "selected_active": int(naturally_active.sum()),
                      "replacement_features": replace_feature.tolist(),
                      "mean_coeff": float(clean.mean()),
                      "images": [{"variants": [{"label": f"{mult:+d} std", "image": png_data(decoded[i])}
                                                   for mult, decoded in variants],
                                  "heatmap": png_data(effect[i])}
                                 for i in range(len(z_raw))]}
            self.cache[feature] = result
            return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sae", required=True)
    p.add_argument("--codes", required=True)
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--examples", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--label", default="SAE")
    args = p.parse_args()
    server = FeatureServer(args)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                payload = HTML.replace("__LABEL__", html.escape(args.label)).encode()
                self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers(); self.wfile.write(payload)
            elif parsed.path == "/api/feature":
                try:
                    query = parse_qs(parsed.query)
                    if "feature" in query:
                        raw_feature = int(query["feature"][0])
                        rank = server.feature_order.index(raw_feature)
                        inactive = query.get("inactive", ["0"])[0] == "1"
                        payload = json.dumps(server.feature(rank, require_inactive=inactive)).encode()
                    else:
                        f = query.get("rank", ["0"])[0]
                        inactive = query.get("inactive", ["0"])[0] == "1"
                        payload = json.dumps(server.feature(int(f), require_inactive=inactive)).encode()
                    self.send_response(200)
                except Exception as e:
                    payload = json.dumps({"error": str(e)}).encode()
                    self.send_response(400)
                self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(payload)
            else:
                self.send_response(404); self.end_headers()
        def log_message(self, *_):
            return

    print(f"GUI ready at http://0.0.0.0:{args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
