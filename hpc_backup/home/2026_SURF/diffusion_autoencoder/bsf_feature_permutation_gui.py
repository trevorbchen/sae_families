#!/usr/bin/env python3
"""On-demand GUI for perturbing vector blocks in the BSF z_sem representation."""

import argparse
import base64
import io
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import torch
from PIL import Image


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>BSF block permutation</title>
<style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:20px}
header{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:14px}
button,input{font:inherit;background:#222;color:#eee;border:1px solid #555;padding:7px 10px;border-radius:4px}
button{cursor:pointer}button:hover{background:#333}input{width:90px}
#status{color:#aaa}#summary{margin:10px 0;color:#ccc;line-height:1.5}
.grid{display:grid;grid-template-columns:repeat(8,minmax(110px,1fr));gap:9px;max-width:1800px}
figure{margin:0;background:#1c1c1c;padding:7px;border-radius:5px}img{width:100%;display:block}
figcaption{font-size:12px;color:#bbb;margin-top:5px}
@media(max-width:800px){.grid{grid-template-columns:repeat(2,minmax(140px,1fr))}}
</style></head><body>
<header><strong>BSF block permutation</strong><span>usage rank</span>
<button id="prev">&#8592;</button><input id="rank" type="number" min="0" value="0">
<button id="next">&#8594;</button><button id="load">Load</button>
<span id="status">ready</span></header>
<div id="summary"></div><div id="grid" class="grid"></div>
<script>
const $=id=>document.getElementById(id);
async function load(){let rank=Math.max(0,parseInt($('rank').value||0));
 $('status').textContent='decoding block rank '+rank+'...';
 try{let r=await fetch('/api/block?rank='+rank);let d=await r.json();if(d.error)throw Error(d.error);
 $('summary').textContent='rank '+d.rank+' | block '+d.block+' | dimension '+d.block_dim+
 ' | selected '+(100*d.selected_fraction).toFixed(2)+'% | mean norm '+d.mean_norm.toFixed(4)+
 ' | norm std '+d.norm_std.toFixed(4)+' | examples '+d.examples;
 $('grid').innerHTML=d.images.map((x,i)=>x.variants.map(v=>
 '<figure><img src="'+v.image+'"><figcaption>example '+i+' '+v.label+'</figcaption></figure>').join('')+
 '<figure><img src="'+x.heatmap+'"><figcaption>example '+i+' spatial effect (+10 vs -10)</figcaption></figure>').join('');
 $('status').textContent='loaded';}catch(e){$('status').textContent='error: '+e.message}}
$('load').onclick=load;$('rank').onchange=load;
$('prev').onclick=()=>{$('rank').value=Math.max(0,parseInt($('rank').value||0)-1);load()};
$('next').onclick=()=>{$('rank').value=parseInt($('rank').value||0)+1;load()};load();
</script></body></html>"""


def png_data(image):
    image = image.detach().float().cpu().clamp(0, 1)
    buf = io.BytesIO()
    Image.fromarray((image.permute(1, 2, 0).numpy() * 255).round().astype("uint8")).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class BlockServer:
    def __init__(self, args):
        self.args = args
        sys.path.insert(0, os.path.abspath(args.repo))
        from experiment import LitModel
        from templates import ffhq128_autoenc_130M
        from train_zsem_bsf import BlockSparseFeaturizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        conf = ffhq128_autoenc_130M()
        conf.pretrain = None
        conf.latent_infer_path = None
        self.lit = LitModel(conf)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        self.lit.load_state_dict(state["state_dict"], strict=False)
        self.lit.eval().to(self.device)

        bsf_state = torch.load(args.bsf, map_location="cpu", weights_only=False)
        cfg = bsf_state["config"]
        self.mean = bsf_state["input_mean"].float().to(self.device)
        self.std = bsf_state["input_std"].float().to(self.device)
        self.bsf = BlockSparseFeaturizer(512, int(cfg["dict_mult"]), int(cfg["block_dim"]), int(cfg["top_blocks"])).to(self.device)
        self.bsf.load_state_dict(bsf_state["model"], strict=True)
        self.bsf.eval()

        codes_state = torch.load(args.codes, map_location="cpu", weights_only=False)
        self.codes = codes_state["conds"].float()
        self.n_blocks = self.bsf.n_blocks
        self.block_dim = self.bsf.block_dim
        self.block_norm_sum = torch.zeros(self.n_blocks, device=self.device)
        self.block_norm_sq = torch.zeros(self.n_blocks, device=self.device)
        self.block_selected = torch.zeros(self.n_blocks, device=self.device)
        with torch.no_grad():
            for start in range(0, len(self.codes), args.batch_size):
                raw = self.codes[start:start + args.batch_size].to(self.device)
                x = (raw - self.mean) / self.std
                coeff = self.bsf.encoder(x - self.bsf.dec_bias).view(-1, self.n_blocks, self.block_dim)
                norms = coeff.norm(dim=-1)
                _, selected = torch.topk(norms, self.bsf.top_blocks, dim=-1)
                self.block_norm_sum += norms.sum(0)
                self.block_norm_sq += (norms * norms).sum(0)
                self.block_selected.scatter_add_(0, selected.reshape(-1), torch.ones_like(selected, dtype=torch.float32).reshape(-1))
        n = float(len(self.codes))
        self.block_mean_norm = self.block_norm_sum / n
        self.block_std_norm = (self.block_norm_sq / n - self.block_mean_norm.square()).clamp_min(0).sqrt()
        self.block_selected_fraction = self.block_selected / n
        self.block_order = torch.argsort(self.block_selected_fraction, descending=True).tolist()
        self.cache = {}
        self.lock = threading.Lock()

    def decode(self, selected_coeff):
        return torch.einsum("bnk,nkd->bd", selected_coeff, self.bsf.decoder) + self.bsf.dec_bias

    @torch.no_grad()
    def block(self, rank):
        rank = int(rank)
        if rank < 0 or rank >= len(self.block_order):
            raise ValueError(f"rank must be in [0, {len(self.block_order) - 1}]")
        block = self.block_order[rank]
        with self.lock:
            if block in self.cache:
                return self.cache[block]

            # Choose examples by the continuous pre-TopK block norm. This also
            # supports examining a block on examples where it was not selected.
            candidates = []
            for start in range(0, len(self.codes), 4096):
                raw = self.codes[start:start + 4096].to(self.device)
                x = (raw - self.mean) / self.std
                coeff = self.bsf.encoder(x - self.bsf.dec_bias).view(-1, self.n_blocks, self.block_dim)
                scores = coeff[:, block].norm(dim=-1)
                candidates.extend((float(v), start + i) for i, v in enumerate(scores))
            candidates.sort(reverse=True)
            selected_ids = [i for _, i in candidates[:self.args.examples]]
            raw = self.codes[selected_ids].to(self.device)
            x = (raw - self.mean) / self.std
            coeff = self.bsf.encoder(x - self.bsf.dec_bias).view(-1, self.n_blocks, self.block_dim)
            norms = coeff.norm(dim=-1)
            _, indices = torch.topk(norms, self.bsf.top_blocks, dim=-1)
            mask = torch.zeros_like(norms, dtype=torch.bool)
            mask.scatter_(1, indices, True)
            selected_coeff = coeff * mask.unsqueeze(-1)
            clean_reconstruction = self.decode(selected_coeff)
            clean = coeff[:, block]
            clean_norm = clean.norm(dim=-1, keepdim=True)
            direction = clean / clean_norm.clamp_min(1e-6)
            fallback = torch.zeros_like(direction)
            fallback[:, 0] = 1.0
            direction = torch.where(clean_norm > 1e-6, direction, fallback)
            sigma = self.block_std_norm[block].clamp_min(1e-6)
            x_t = torch.randn(len(selected_ids), 3, 128, 128, device=self.device,
                              generator=torch.Generator(device=self.device).manual_seed(self.args.seed + block))
            variants = []
            for mult in (-10, -4, -2, 0, 2, 4, 10):
                if mult == 0:
                    # This is the exact BSF reconstruction, with no forced
                    # intervention on the inspected block.
                    modified = selected_coeff
                else:
                    target = clean + mult * sigma * direction
                    modified = selected_coeff.clone()
                    modified[:, block] = target
                if mult == 0:
                    # Match the SAE GUI: zero intervention preserves the raw
                    # Diff-AE latent, rather than replacing it by a lossy
                    # sparse reconstruction.
                    cond_raw = raw
                else:
                    # Inject only the BSF block's change into the original
                    # latent. Both reconstructions are standardized.
                    delta_norm = self.decode(modified) - clean_reconstruction
                    cond_raw = raw + delta_norm * self.std
                decoded = self.lit.render(x_t, cond=cond_raw, T=20).clamp(0, 1)
                variants.append((mult, decoded))
            effect = (variants[-1][1] - variants[0][1]).abs()
            effect = effect / effect.flatten(1).amax(1).clamp_min(1e-6)[:, None, None, None]
            result = {
                "rank": rank, "block": block, "block_dim": self.block_dim,
                "examples": len(selected_ids), "selected_fraction": float(self.block_selected_fraction[block]),
                "mean_norm": float(self.block_mean_norm[block]), "norm_std": float(sigma),
                "images": [{
                    "variants": [
                        {"label": f"{mult:+d} std", "image": png_data(decoded[i])}
                        for mult, decoded in variants
                    ],
                    "heatmap": png_data(effect[i]),
                } for i in range(len(selected_ids))],
            }
            self.cache[block] = result
            return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bsf", required=True)
    p.add_argument("--codes", required=True)
    p.add_argument("--port", type=int, default=8768)
    p.add_argument("--examples", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    server = BlockServer(args)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                payload = HTML.encode(); self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers(); self.wfile.write(payload)
            elif parsed.path == "/api/block":
                try:
                    rank = int(parse_qs(parsed.query).get("rank", ["0"])[0])
                    payload = json.dumps(server.block(rank)).encode(); self.send_response(200)
                except Exception as exc:
                    payload = json.dumps({"error": str(exc)}).encode(); self.send_response(400)
                self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(payload)
            else:
                self.send_response(404); self.end_headers()
        def log_message(self, *_):
            return

    print(f"BSF GUI ready at http://0.0.0.0:{args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
