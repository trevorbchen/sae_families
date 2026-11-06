#!/usr/bin/env python3
"""Shortlist gallery for the 117 features you already manually marked
"interesting" in proper_residual_feature_review/annotations.json, restricted
to the grouped_objective_{laplacian,fourier_low,fourier_mid,fourier_high}
sources -- these are single-objective FlatTopKSAEs trained with
transform_weight=1.0 (verified from their own saved config), i.e. the
"general features" (lambda=1-equivalent) counterpart to the matched512
lambda=100 shortlist in matched512_feature_permutation_gui.py's /shortlist.

Deliberately does NOT touch or rewrite annotations.json (4,278 entries
across 10 sources, most of it Kimi auto-labels this GUI has no business
near) -- reads it read-only for the interesting/comment fields already
there, and writes a small, separate "chosen for paper" file instead.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch
import torch.nn.functional as F
from PIL import Image

from spatial_conv_autoencoder import load_spatial_autoencoder

SOURCE_PREFIX = "grouped_objective_"
STAGE_LABELS = {
    "laplacian": "Laplacian", "fourier_low": "Fourier low",
    "fourier_mid": "Fourier mid", "fourier_high": "Fourier high",
}
MULTIPLIERS = (-10, -4, -2, 0, 2, 4, 10)


def png_data(image: torch.Tensor) -> str:
    image = image.detach().float().cpu().clamp(0, 1)
    buf = io.BytesIO()
    Image.fromarray((image.permute(1, 2, 0).numpy() * 255).round().astype("uint8")).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class ObjectiveSAE:
    """One loaded (stage, seed) FlatTopKSAE checkpoint plus its dataset-wide
    feature usage stats. Same math as matched512_feature_permutation_gui.py's
    class of the same name -- this family uses the identical FlatTopKSAE
    architecture and channel_mean/channel_std normalization convention."""

    def __init__(self, checkpoint_path: Path, latents: torch.Tensor, candidate_indices: torch.Tensor,
                 device: torch.device, batch_size: int):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        input_shape = tuple(state["input_shape"])
        config = state["config"]
        from train_spatial_map_sae import FlatTopKSAE
        self.sae = FlatTopKSAE(input_shape, config["feature_count"], config["k"]).to(device)
        self.sae.load_state_dict(state["model"])
        self.sae.eval()
        spatial_axes = max(0, len(input_shape) - 1)
        stats_shape = (1, -1) + (1,) * spatial_axes
        self.channel_mean = state["channel_mean"].float().to(device).view(stats_shape)
        self.channel_std = state["channel_std"].float().to(device).view(stats_shape)
        self.input_shape = input_shape
        self.dict_size = self.sae.h
        self.transform_weight = config.get("transform_weight")

        sums = torch.zeros(self.dict_size, device=device)
        squares = torch.zeros(self.dict_size, device=device)
        active = torch.zeros(self.dict_size, device=device)
        with torch.no_grad():
            for start in range(0, len(candidate_indices), batch_size):
                selected = candidate_indices[start:start + batch_size]
                raw = latents[selected].to(device).float()
                x = (raw - self.channel_mean) / self.channel_std
                pre = F.linear(x.flatten(1) - self.sae.dec_bias, self.sae.encoder, self.sae.encoder_bias)
                acts = torch.relu(pre)
                vals, idx = torch.topk(acts, self.sae.k, dim=-1)
                sums.scatter_add_(0, idx.reshape(-1), vals.reshape(-1))
                squares.scatter_add_(0, idx.reshape(-1), (vals * vals).reshape(-1))
                active.scatter_add_(0, idx.reshape(-1), torch.ones_like(vals).reshape(-1))
        count = len(candidate_indices)
        self.dataset_mean_activation = sums / count
        self.dataset_std_activation = (squares / count - self.dataset_mean_activation.square()).clamp_min(0).sqrt()
        self.dataset_active_fraction = active / count


class ChoiceStore:
    """Small, separate "chosen for paper" file -- never touches the
    original review annotations.json."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.records = json.loads(self.path.read_text()) if self.path.is_file() else {}

    def get(self, key: str) -> dict:
        with self.lock:
            return dict(self.records.get(key, {}))

    def set(self, key: str, source: str, feature: int, chosen: bool | None = None,
             chosen_example_index: int | None = None) -> dict:
        # chosen / chosen_example_index are independent toggles from two
        # different controls -- whichever one this call doesn't specify
        # must be preserved from the existing record, not reset.
        existing = self.records.get(key, {})
        record = {
            "source": source, "feature": feature,
            "chosen": bool(chosen) if chosen is not None else bool(existing.get("chosen", False)),
            "chosen_example_index": (
                chosen_example_index if chosen_example_index is not None else existing.get("chosen_example_index")
            ),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with self.lock:
            self.records[key] = record
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(json.dumps(self.records, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, self.path)
        return dict(record)


SHORTLIST_PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>__LABEL__ shortlist</title>
<style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:20px}
h1{font-size:18px;margin:0 0 4px}
#status{color:#aaa;margin-bottom:16px}
h2{font-size:15px;color:#e1a52b;border-bottom:1px solid #444;padding-bottom:4px;margin:28px 0 12px}
.card{background:#1b1b1b;border:1px solid #333;border-radius:6px;padding:14px;margin-bottom:14px}
.card.chosen{border-color:#e1a52b;box-shadow:0 0 0 1px #e1a52b}
.card-header{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:8px}
.card-header strong{font-size:14px}
.meta{color:#999;font-size:12px}
.comment{color:#ddd;font-size:13px;margin:6px 0 10px;white-space:pre-wrap}
label.choose{display:flex;align-items:center;gap:6px;cursor:pointer;font-size:13px;margin-left:auto}
.grid{display:grid;grid-template-columns:repeat(8,minmax(100px,1fr));gap:8px}
figure{margin:0;background:#111;padding:6px;border-radius:4px}img{width:100%;display:block}
figcaption{font-size:10px;color:#999;margin-top:4px}
.example-row{border:1px solid #2a2a2a;border-radius:4px;padding:8px;margin-bottom:8px}
.example-row.selected-example{border-color:#3fa9f5;box-shadow:0 0 0 1px #3fa9f5}
.example-row-header{display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:12px;color:#bbb}
label.pick{display:flex;align-items:center;gap:5px;cursor:pointer;font-size:12px;color:#3fa9f5}
</style></head><body>
<h1>__LABEL__ -- shortlist (manually marked interesting)</h1>
<label style="display:flex;align-items:center;gap:6px;font-size:13px;color:#ddd;margin-bottom:12px;cursor:pointer">
 <input type="checkbox" id="chosenOnly" checked> chosen for paper only (uncheck to see all interesting)</label>
<div id="status">loading...</div>
<div id="sections"></div>
<script>
const $=id=>document.getElementById(id);
async function saveChoice(item,chosen){
 let response=await fetch('/api/choice',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({key:item.key,source:item.source,feature:item.feature,chosen})});
 let data=await response.json();if(data.error)throw Error(data.error);return data.choice;
}
// Sends chosen_example_index but omits chosen -- the server preserves the
// existing chosen flag when a plain example-pick save omits it.
async function saveExampleChoice(item,index){
 let response=await fetch('/api/choice',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({key:item.key,source:item.source,feature:item.feature,chosen_example_index:index})});
 let data=await response.json();if(data.error)throw Error(data.error);return data.choice;
}
function card(item){
 let chosen=!!(item.choice&&item.choice.chosen);
 let chosenIndex=item.choice&&Number.isInteger(item.choice.chosen_example_index)?item.choice.chosen_example_index:null;
 let div=document.createElement('div');div.className='card'+(chosen?' chosen':'');
 div.innerHTML='<div class="card-header"><strong>'+item.stage_label+' | feature '+item.feature+' | rank '+item.rank+'</strong>'
  +'<span class="meta">active '+(100*item.dataset_active_fraction).toFixed(2)+'% of dataset | transform_weight '+item.transform_weight+'</span>'
  +'<label class="choose"><input type="checkbox" '+(chosen?'checked':'')+'> Chosen for paper</label></div>'
  +'<div class="comment">'+(item.comment?item.comment.replace(/</g,'&lt;'):'(no comment)')+'</div>'
  +'<div class="examples"></div>';
 let container=div.querySelector('.examples');
 item.images.forEach((x,i)=>{
  let row=document.createElement('div');row.className='example-row'+(i===chosenIndex?' selected-example':'');
  row.innerHTML='<div class="example-row-header"><span>example '+i+'</span><label class="pick"><input type="radio" name="example-'+item.key+'" '+(i===chosenIndex?'checked':'')+'> Use this example</label></div>'
   +'<div class="grid">'+x.variants.map(v=>'<figure><img src="'+v.image+'"><figcaption>'+v.label+'</figcaption></figure>').join('')+'<figure><img src="'+x.heatmap+'"><figcaption>effect</figcaption></figure></div>';
  row.querySelector('input[type=radio]').addEventListener('change',async event=>{
   try{item.choice=await saveExampleChoice(item,i);
    container.querySelectorAll('.example-row').forEach((r,ri)=>r.classList.toggle('selected-example',ri===i));
   }catch(e){event.target.checked=false;alert('save failed: '+e.message)}
  });
  container.appendChild(row);
 });
 div.querySelector('input[type=checkbox]').addEventListener('change',async event=>{
  try{item.choice=await saveChoice(item,event.target.checked);div.classList.toggle('chosen',event.target.checked)}
  catch(e){event.target.checked=!event.target.checked;alert('save failed: '+e.message)}
 });
 return div;
}
async function load(){
 let chosenOnly=$('chosenOnly').checked;
 $('status').textContent='loading shortlist'+(chosenOnly?' (chosen for paper only)':' (all interesting)')+' (this recomputes each feature\'s permutation grid, can take a bit)...';
 try{
  let response=await fetch('/api/shortlist?chosen_only='+(chosenOnly?'1':'0')),items=await response.json();
  if(items.error)throw Error(items.error);
  $('status').textContent=items.length+' feature(s)'+(chosenOnly?' chosen for paper':' marked interesting');
  let sections=$('sections');sections.innerHTML='';
  let byStage=new Map();
  for(let item of items){if(!byStage.has(item.stage_label))byStage.set(item.stage_label,[]);byStage.get(item.stage_label).push(item)}
  for(let [stage,group] of byStage){
   let h2=document.createElement('h2');h2.textContent=stage+' ('+group.length+')';sections.appendChild(h2);
   for(let item of group)sections.appendChild(card(item));
  }
 }catch(e){$('status').textContent='error: '+e.message}
}
$('chosenOnly').addEventListener('change',load);
load();
</script></body></html>"""


class FeatureServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder = load_spatial_autoencoder(args.autoencoder, self.device)
        self.autoencoder.eval()

        cache = torch.load(args.latent_cache, map_location="cpu", weights_only=False)
        self.latents = cache["latents"].float()
        candidate_source = cache.get("valid_indices", cache.get("train_indices"))
        self.candidate_indices = candidate_source.long()[:args.rank_examples]

        self.choices = ChoiceStore(args.choices)
        review = json.loads(Path(args.review_annotations).read_text())
        self.shortlist_records = [
            dict(v, key=k) for k, v in review.items()
            if v.get("source", "").startswith(SOURCE_PREFIX) and v.get("interesting")
        ]
        print(f"{len(self.shortlist_records)} interesting records under {SOURCE_PREFIX}*", flush=True)

        self.stages: dict[str, ObjectiveSAE] = {}
        for stage in STAGE_LABELS:
            checkpoint_path = Path(args.checkpoint_root) / f"weighted_{stage}" / args.seed / "last.pt"
            print(f"loading {stage} ...", flush=True)
            self.stages[stage] = ObjectiveSAE(
                checkpoint_path, self.latents, self.candidate_indices, self.device, args.batch_size,
            )
        print("all stage models loaded", flush=True)
        self.cache: dict[tuple[str, int], dict] = {}
        self.lock = threading.Lock()

    @torch.no_grad()
    def render_feature(self, stage: str, feature: int) -> dict:
        key = (stage, feature)
        with self.lock:
            if key in self.cache:
                return self.cache[key]
            obj = self.stages[stage]

            candidates = []
            for start in range(0, len(self.candidate_indices), 4096):
                positions = self.candidate_indices[start:start + 4096]
                raw = self.latents[positions].to(self.device)
                x = (raw - obj.channel_mean) / obj.channel_std
                pre = F.linear(x.flatten(1) - obj.sae.dec_bias, obj.sae.encoder, obj.sae.encoder_bias)
                scores = pre[:, feature]
                for row in range(len(scores)):
                    candidates.append((float(scores[row]), int(positions[row])))
            candidates.sort(reverse=True)
            selected = [idx for _, idx in candidates[:self.args.examples]]

            z_raw = self.latents[selected].to(self.device)
            x = (z_raw - obj.channel_mean) / obj.channel_std
            flat = x.flatten(1)
            pre = F.linear(flat - obj.sae.dec_bias, obj.sae.encoder, obj.sae.encoder_bias)
            acts = torch.relu(pre)
            vals, idx = torch.topk(acts, obj.sae.k, dim=-1)
            coeff = torch.zeros_like(acts)
            coeff.scatter_(1, idx, vals)
            clean = coeff[:, feature]
            sigma = obj.dataset_std_activation[feature].clamp_min(1e-6)
            naturally_active = clean > 0
            replace_pos = vals.argmin(dim=1)
            replace_feature = idx.gather(1, replace_pos[:, None]).squeeze(1)
            replace_value = vals.gather(1, replace_pos[:, None]).squeeze(1)

            variants = []
            for mult in MULTIPLIERS:
                if mult == 0:
                    flat_mod = flat
                else:
                    target = torch.clamp(clean + mult * sigma, min=0)
                    delta = (target - clean)[:, None] * obj.sae.decoder[feature]
                    support_added = (~naturally_active) & (target > 0)
                    if support_added.any():
                        delta[support_added] -= (
                            replace_value[support_added, None]
                            * obj.sae.decoder[replace_feature[support_added]]
                        )
                    flat_mod = flat + delta
                x_mod = flat_mod.view(-1, *obj.input_shape)
                z_mod_raw = x_mod * obj.channel_std + obj.channel_mean
                decoded = self.autoencoder.decode(z_mod_raw).clamp(0, 1)
                variants.append((mult, decoded))

            effect = (variants[-1][1] - variants[0][1]).abs()
            effect = effect / effect.flatten(1).amax(1).clamp_min(1e-6)[:, None, None, None]
            result = {
                "stage": stage, "feature": feature,
                "dataset_active_fraction": float(obj.dataset_active_fraction[feature]),
                "transform_weight": obj.transform_weight,
                "images": [{"variants": [{"label": f"{mult:+d} std", "image": png_data(decoded[i])}
                                          for mult, decoded in variants],
                            "heatmap": png_data(effect[i])}
                           for i in range(len(selected))],
            }
            self.cache[key] = result
            return result

    def shortlist(self, max_examples: int, chosen_only: bool = False) -> list[dict]:
        items = []
        for record in self.shortlist_records:
            stage = record["source"].removeprefix(SOURCE_PREFIX)
            if stage not in self.stages:
                continue
            if chosen_only:
                choice = self.choices.get(record["key"])
                if not (choice and choice.get("chosen")):
                    continue
            try:
                rendered = dict(self.render_feature(stage, record["feature"]))
            except Exception as exc:  # noqa: BLE001
                print(f"shortlist: skipping {record['key']}: {exc}", flush=True)
                continue
            rendered["images"] = rendered["images"][:max_examples]
            rendered["rank"] = record.get("rank")
            rendered["comment"] = record.get("comment", "")
            rendered["stage_label"] = STAGE_LABELS.get(stage, stage)
            rendered["key"] = record["key"]
            rendered["source"] = record["source"]
            rendered["choice"] = self.choices.get(record["key"])
            items.append(rendered)
        return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True,
                         help="e.g. results/spatial_map_sae_weighted_seed_replicates/flat")
    parser.add_argument("--seed", default="seed_2030")
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--latent-cache", required=True)
    parser.add_argument("--review-annotations", required=True,
                         help="the existing proper_residual_feature_review/annotations.json (read-only)")
    parser.add_argument("--choices", required=True,
                         help="separate file this GUI writes 'chosen for paper' flags to")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--rank-examples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shortlist-examples", type=int, default=10)
    parser.add_argument("--label", default="Residual SAE general features")
    args = parser.parse_args()
    server = FeatureServer(args)
    page = SHORTLIST_PAGE.replace("__LABEL__", html.escape(args.label))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/shortlist"):
                payload = page.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(payload)
            elif parsed.path == "/api/shortlist":
                try:
                    chosen_only = parse_qs(parsed.query).get("chosen_only", ["0"])[0] == "1"
                    payload = json.dumps(server.shortlist(args.shortlist_examples, chosen_only)).encode()
                    self.send_response(200)
                except Exception as exc:  # noqa: BLE001
                    payload = json.dumps({"error": str(exc)}).encode()
                    self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/api/choice":
                self.send_response(404)
                self.end_headers()
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload_in = json.loads(self.rfile.read(length))
                choice = server.choices.set(
                    payload_in["key"], payload_in["source"], int(payload_in["feature"]),
                    chosen=bool(payload_in["chosen"]) if "chosen" in payload_in else None,
                    chosen_example_index=(
                        int(payload_in["chosen_example_index"])
                        if payload_in.get("chosen_example_index") is not None else None
                    ),
                )
                payload = json.dumps({"choice": choice}).encode()
                self.send_response(200)
            except Exception as exc:  # noqa: BLE001
                payload = json.dumps({"error": str(exc)}).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            return

    print(f"GUI ready at http://0.0.0.0:{args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
