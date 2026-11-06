#!/usr/bin/env python3
"""Feature intervention GUI for flat or convolutional SAEs on a spatial AE map."""

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
from train_spatial_map_sae import ConvTopKSAE, FlatTopKSAE
from vector_conv_autoencoder import load_vector_autoencoder


OBJECTIVES = (
    "latent_mse",
    "laplacian",
    "fourier_bands",
    "fourier_low",
    "fourier_mid",
    "fourier_high",
)


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>__LABEL__ SAE features</title>
<style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:20px}
body.paper-candidate:before{content:"";position:fixed;z-index:20;left:0;top:0;bottom:0;width:6px;background:#e1a52b}
header{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:16px}
button,input,select,textarea,a.command{font:inherit;background:#222;color:#eee;border:1px solid #555;padding:7px 10px;border-radius:4px}
button{cursor:pointer}button:hover{background:#333}input{width:100px}
#interesting[aria-pressed=true]{background:#e1a52b;border-color:#f4c35b;color:#17130a;font-weight:700}
#status{color:#aaa}#summary{margin:10px 0;color:#ccc}
.annotation{display:grid;grid-template-columns:auto minmax(260px,1fr) auto;gap:10px;align-items:stretch;margin:12px 0 16px;max-width:1800px}
textarea{width:100%;min-height:48px;resize:vertical;line-height:1.35}a.command{text-decoration:none;display:inline-block}
.grid{display:grid;grid-template-columns:repeat(8,minmax(120px,1fr));gap:10px;max-width:1800px}
figure{margin:0;background:#1c1c1c;padding:8px;border-radius:5px}img{width:100%;display:block}
figcaption{font-size:12px;color:#bbb;margin-top:6px}
@media(max-width:700px){.grid{grid-template-columns:repeat(2,minmax(140px,1fr))}.annotation{grid-template-columns:1fr}}
</style></head><body>
<header><strong>__LABEL__ SAE</strong>
<span>objective</span><select id="objective">__OPTIONS__</select>
<span>importance rank</span><button id="prev">&#8592;</button><input id="feature" type="number" min="0" value="0">
<button id="next">&#8594;</button><button id="load">Load</button><span id="status">ready</span></header>
<div id="summary"></div>
<div class="annotation"><button id="interesting" aria-pressed="false" title="Toggle interesting (I)">Interesting</button><textarea id="comment" placeholder="Feature notes (C to focus)"></textarea><div><button id="save">Save note</button> <a class="command" href="/api/annotations" download="lambda10_vector4x_feature_annotations.json">Export</a></div></div>
<div id="grid" class="grid"></div>
<script>
const $=id=>document.getElementById(id);
const objectives=__OBJECTIVES__,results=new Map(),inflight=new Map();let requestToken=0,current=null,dirty=false,saveTimer=null,saveInFlight=null;
function resultKey(objective,rank){return objective+':'+rank}
function render(d){
 current=d;dirty=false;let a=d.annotation||{};$('interesting').setAttribute('aria-pressed',a.interesting?'true':'false');document.body.classList.toggle('paper-candidate',!!a.interesting);$('comment').value=a.comment||'';
 $('summary').textContent='objective '+d.objective+' | rank '+d.rank+' | feature '+d.feature+' | mean '+d.dataset_mean_activation.toFixed(5)+' | active RMS '+d.active_rms.toFixed(5)+' | active '+(100*d.dataset_active_fraction).toFixed(2)+'% | selected active '+d.selected_active+' / '+d.examples;
 $('grid').innerHTML=d.images.map((x,i)=>x.variants.map(v=>'<figure><img src="'+v.image+'"><figcaption>example '+i+' '+v.label+'</figcaption></figure>').join('')+'<figure><img src="'+x.heatmap+'"><figcaption>example '+i+' output effect (+10 vs -10)</figcaption></figure>').join('');
}
async function saveAnnotation(){clearTimeout(saveTimer);if(saveInFlight)await saveInFlight;if(!current||!dirty)return;
 const target=current,payload={key:target.annotation_key,objective:target.objective,rank:target.rank,feature:target.feature,interesting:$('interesting').getAttribute('aria-pressed')==='true',comment:$('comment').value};dirty=false;
 saveInFlight=(async()=>{let response=await fetch('/api/annotation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}),data=await response.json();if(data.error)throw Error(data.error);target.annotation=data.annotation;results.set(resultKey(target.objective,target.rank),target);$('status').textContent='note saved'})();
 try{await saveInFlight}catch(error){dirty=true;throw error}finally{saveInFlight=null}}
function queueSave(){dirty=true;clearTimeout(saveTimer);saveTimer=setTimeout(()=>saveAnnotation().catch(e=>$('status').textContent='save error: '+e.message),700)}
async function toggleInteresting(){if(!current)return;let value=$('interesting').getAttribute('aria-pressed')!=='true';$('interesting').setAttribute('aria-pressed',value?'true':'false');document.body.classList.toggle('paper-candidate',value);dirty=true;await saveAnnotation()}
async function fetchResult(objective,rank){let key=resultKey(objective,rank);if(results.has(key))return results.get(key);
 if(inflight.has(key))return inflight.get(key);let promise=(async()=>{let response=await fetch('/api/feature?objective='+encodeURIComponent(objective)+'&rank='+rank),data=await response.json();
  if(data.error)throw Error(data.error);results.set(key,data);return data})().finally(()=>inflight.delete(key));
 inflight.set(key,promise);return promise}
function trimCache(center,activeObjective){for(let [key,data] of results){
 if(data.objective!==activeObjective||data.rank<center-3||data.rank>center+10)results.delete(key)
}}
async function prefetchNext(rank,activeObjective,token){try{
 for(let next=rank+1;next<=rank+10;next++){
  if(token!==requestToken)return;
  await fetchResult(activeObjective,next);
  if(parseInt($('feature').value||0)===rank&&$('objective').value===activeObjective)$('status').textContent='loaded next '+(next-rank)+' / 10 features';
 }
 trimCache(rank,activeObjective);if(parseInt($('feature').value||0)===rank&&$('objective').value===activeObjective)$('status').textContent='current and next 10 features loaded';
 }catch(e){if(parseInt($('feature').value||0)===rank)$('status').textContent='current loaded; prefetch error: '+e.message}}
async function load(){await saveAnnotation();let rank=Math.max(0,parseInt($('feature').value||0)),token=++requestToken,objective=$('objective').value;
 $('status').textContent='loading '+objective+' rank '+rank+'...';
 try{let selected=await fetchResult(objective,rank);if(token!==requestToken)return;
 render(selected);$('status').textContent='selected loss loaded; prefetching next 10 features';
 trimCache(rank,objective);void prefetchNext(rank,objective,token);
 }catch(e){if(token===requestToken)$('status').textContent='error: '+e.message}}
async function showObjective(){await saveAnnotation();let rank=Math.max(0,parseInt($('feature').value||0)),data=results.get(resultKey($('objective').value,rank));
 if(data){render(data);$('status').textContent='feature loaded'}else load()}
async function moveFeature(delta){await saveAnnotation();$('feature').value=Math.max(0,parseInt($('feature').value||0)+delta);load()}
async function moveObjective(delta){await saveAnnotation();let select=$('objective'),count=select.options.length;
 select.selectedIndex=(select.selectedIndex+delta+count)%count;showObjective()}
$('load').onclick=load;$('feature').onchange=load;$('objective').onchange=showObjective;
$('prev').onclick=()=>moveFeature(-1);$('next').onclick=()=>moveFeature(1);
$('interesting').onclick=()=>toggleInteresting().catch(e=>$('status').textContent='save error: '+e.message);$('save').onclick=()=>saveAnnotation().catch(e=>$('status').textContent='save error: '+e.message);$('comment').oninput=queueSave;
$('comment').addEventListener('keydown',event=>{if((event.ctrlKey||event.metaKey)&&event.key==='Enter'){event.preventDefault();saveAnnotation().then(()=>$('comment').blur())}else if(event.key==='Escape'){event.preventDefault();saveAnnotation().then(()=>$('comment').blur())}});
document.addEventListener('keydown',event=>{if(['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName))return;
 if(event.key==='ArrowLeft'){event.preventDefault();moveFeature(-1)}
 else if(event.key==='ArrowRight'){event.preventDefault();moveFeature(1)}
 else if(event.key==='ArrowUp'){event.preventDefault();moveObjective(-1)}
 else if(event.key==='ArrowDown'){event.preventDefault();moveObjective(1)}
 else if(event.key.toLowerCase()==='i'){event.preventDefault();toggleInteresting()}
 else if(event.key.toLowerCase()==='c'){event.preventDefault();$('comment').focus()}});
window.addEventListener('beforeunload',()=>{if(current&&dirty){let payload={key:current.annotation_key,objective:current.objective,rank:current.rank,feature:current.feature,interesting:$('interesting').getAttribute('aria-pressed')==='true',comment:$('comment').value};navigator.sendBeacon('/api/annotation',new Blob([JSON.stringify(payload)],{type:'application/json'}))}});load();
</script></body></html>"""


def png_data(image):
    image = image.detach().float().cpu().clamp(0, 1)
    buffer = io.BytesIO()
    array = (image.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    Image.fromarray(array).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class AnnotationStore:
    def __init__(self, path):
        self.path = Path(path) if path else None
        self.lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = json.loads(self.path.read_text()) if self.path and self.path.is_file() else {}

    def get(self, key):
        with self.lock:
            return dict(self.records.get(key, {}))

    def update(self, payload):
        record = {
            "objective": str(payload["objective"]), "rank": int(payload["rank"]),
            "feature": int(payload["feature"]), "interesting": bool(payload.get("interesting", False)),
            "comment": str(payload.get("comment", "")), "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with self.lock:
            self.records[payload["key"]] = record
            if self.path:
                temporary = self.path.with_suffix(self.path.suffix + ".tmp")
                temporary.write_text(json.dumps(self.records, indent=2, sort_keys=True) + "\n")
                os.replace(temporary, self.path)
        return dict(record)

    def export(self):
        with self.lock:
            return json.dumps(self.records, indent=2, sort_keys=True) + "\n"


class SpatialFeatureServer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder = (
            load_vector_autoencoder(args.autoencoder, self.device)
            if args.autoencoder_type == "vector"
            else load_spatial_autoencoder(args.autoencoder, self.device)
        )
        cache = torch.load(args.latent_cache, map_location="cpu", weights_only=False)
        self.latents = cache["latents"]
        self.rank_indices = cache["train_indices"][:min(args.rank_examples, len(cache["train_indices"]))]
        self.objectives = tuple(args.objectives or OBJECTIVES)
        self.checkpoint_manifest = (
            json.loads(Path(args.checkpoint_manifest).read_text())
            if args.checkpoint_manifest else None
        )
        self.annotations = AnnotationStore(args.annotations)
        self.stats = {}
        self.cache = {}
        self.models = {}
        self.states = {}
        self.lock = threading.Lock()
        for objective in self.objectives:
            self.load_model(objective)
        for objective in self.objectives:
            print(f"precomputing {objective} ranking", flush=True)
            self.prepare_stats(objective)
        print("all objective models and rankings loaded", flush=True)

    def checkpoint_path(self, objective):
        if self.checkpoint_manifest is not None:
            return self.checkpoint_manifest[objective]
        return os.path.join(
            self.args.checkpoint_root, self.args.architecture, objective, "last.pt"
        )

    def load_model(self, objective):
        if objective not in self.objectives:
            raise ValueError(f"unknown objective: {objective}")
        if objective in self.models:
            return self.models[objective], self.states[objective]
        state = torch.load(
            self.checkpoint_path(objective), map_location="cpu", weights_only=False
        )
        config = state["config"]
        cls = FlatTopKSAE if self.args.architecture == "flat" else ConvTopKSAE
        model = cls(
            state["input_shape"], int(config["feature_count"]), int(config["k"])
        ).to(self.device)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        self.models[objective] = model
        self.states[objective] = state
        return model, state

    def standardize(self, raw, state):
        mean = state["channel_mean"].float().to(self.device)
        std = state["channel_std"].float().to(self.device)
        if raw.ndim == 2:
            mean, std = mean[None, :], std[None, :]
        else:
            mean, std = mean[None, :, None, None], std[None, :, None, None]
        return (raw - mean) / std, mean, std

    def selected_codes(self, model, standardized):
        if isinstance(model, FlatTopKSAE):
            flat = standardized.flatten(1)
            activations = F.relu(F.linear(
                flat - model.dec_bias, model.encoder, model.encoder_bias
            ))
            values, indices = torch.topk(activations, model.k, dim=1)
        else:
            centered = standardized.flatten(2) - model.dec_bias[None, :, None]
            activations = F.relu(
                torch.einsum("bcp,fc->bfp", centered, model.encoder)
                + model.encoder_bias[None, :, None]
            )
            values, indices = torch.topk(activations, model.k, dim=1)
        return values, indices

    @torch.no_grad()
    def prepare_stats(self, objective):
        if objective in self.stats:
            return self.stats[objective]
        model, state = self.load_model(objective)
        sums = torch.zeros(model.h, device=self.device)
        squares = torch.zeros(model.h, device=self.device)
        active = torch.zeros(model.h, device=self.device)
        candidates = [[] for _ in range(model.h)]
        observations = 0
        for start in range(0, len(self.rank_indices), self.args.batch_size):
            indices = self.rank_indices[start:start + self.args.batch_size]
            raw = self.latents[indices].to(self.device).float()
            standardized, _, _ = self.standardize(raw, state)
            values, selected = self.selected_codes(model, standardized)
            positive = values > 0
            ids = selected[positive]
            vals = values[positive]
            rows = torch.arange(len(raw), device=self.device)[:, None].expand_as(selected)[positive]
            sums.scatter_add_(0, ids, vals)
            squares.scatter_add_(0, ids, vals.square())
            active.scatter_add_(0, ids, torch.ones_like(vals))
            for feature, value, row in zip(
                ids.detach().cpu().tolist(),
                vals.detach().cpu().tolist(),
                rows.detach().cpu().tolist(),
            ):
                candidates[feature].append((value, int(indices[row])))
            observations += len(raw) * (model.positions if isinstance(model, ConvTopKSAE) else 1)
        mean = sums / observations
        fraction = active / observations
        active_rms = (squares / active.clamp_min(1)).sqrt()
        order = torch.argsort(sums, descending=True).tolist()
        result = {
            "mean": mean,
            "fraction": fraction,
            "active_rms": active_rms,
            "order": order,
            "top_examples": [
                [index for _, index in sorted(records, reverse=True)[:self.args.examples]]
                for records in candidates
            ],
        }
        self.stats[objective] = result
        return result

    @torch.no_grad()
    def feature_scores(self, model, standardized, feature):
        values, selected = self.selected_codes(model, standardized)
        matches = selected == feature
        if isinstance(model, FlatTopKSAE):
            coefficients = (values * matches).sum(1)
            return coefficients, coefficients
        coefficients = (values * matches).sum(1)
        return coefficients, coefficients.sum(1)

    @torch.no_grad()
    def feature(self, objective, rank):
        rank = int(rank)
        with self.lock:
            stats = self.prepare_stats(objective)
            if rank < 0 or rank >= len(stats["order"]):
                raise ValueError(f"rank must be in [0, {len(stats['order']) - 1}]")
            feature = stats["order"][rank]
            key = (objective, feature)
            if key in self.cache:
                return self.cache[key]
            model, state = self.load_model(objective)
            selected_examples = stats["top_examples"][feature]
            if not selected_examples:
                candidates = []
                for start in range(0, len(self.latents), self.args.batch_size):
                    raw = self.latents[start:start + self.args.batch_size].to(self.device).float()
                    standardized, _, _ = self.standardize(raw, state)
                    _, scores = self.feature_scores(model, standardized, feature)
                    candidates.extend(
                        (float(score), start + row) for row, score in enumerate(scores)
                        if score > 0
                    )
                candidates.sort(reverse=True)
                selected_examples = [index for _, index in candidates[:self.args.examples]]
            if not selected_examples:
                raise ValueError("feature had no active examples in the latent cache")
            raw = self.latents[selected_examples].to(self.device).float()
            standardized, _, input_std = self.standardize(raw, state)
            coefficients, _ = self.feature_scores(model, standardized, feature)
            scale = stats["active_rms"][feature].clamp_min(1e-6)
            multipliers = (-10, -4, -2, 0, 2, 4, 10)
            modified_batches = []
            for multiplier in multipliers:
                if multiplier == 0:
                    modified_raw = raw
                elif isinstance(model, FlatTopKSAE):
                    target = torch.clamp(coefficients + multiplier * scale, min=0)
                    delta = (target - coefficients)[:, None] * model.decoder[feature]
                    modified_raw = raw + delta.view_as(standardized) * input_std
                else:
                    naturally_active = coefficients > 0
                    target = torch.where(
                        naturally_active,
                        torch.clamp(coefficients + multiplier * scale, min=0),
                        coefficients,
                    )
                    delta = (
                        (target - coefficients)[:, None, :]
                        * model.decoder[feature][None, :, None]
                    )
                    modified_raw = raw + delta.view_as(standardized) * input_std
                modified_batches.append(modified_raw)
            decoded_batches = self.autoencoder.decode(torch.cat(modified_batches)).add(1).div(2).clamp(0, 1)
            variants = list(zip(multipliers, decoded_batches.split(len(raw))))
            effect = (variants[-1][1] - variants[0][1]).abs()
            effect = effect / effect.flatten(1).amax(1).clamp_min(1e-6)[:, None, None, None]
            if isinstance(model, FlatTopKSAE):
                selected_active = int((coefficients > 0).sum())
            else:
                selected_active = int((coefficients > 0).any(1).sum())
            result = {
                "architecture": self.args.architecture,
                "objective": objective,
                "rank": rank,
                "feature": feature,
                "examples": len(raw),
                "dataset_mean_activation": float(stats["mean"][feature]),
                "dataset_active_fraction": float(stats["fraction"][feature]),
                "active_rms": float(scale),
                "selected_active": selected_active,
                "annotation_key": f"{objective}:{feature}",
                "annotation": self.annotations.get(f"{objective}:{feature}"),
                "images": [
                    {
                        "variants": [
                            {"label": f"{multiplier:+d} scale", "image": png_data(decoded[row])}
                            for multiplier, decoded in variants
                        ],
                        "heatmap": png_data(effect[row]),
                    }
                    for row in range(len(raw))
                ],
            }
            self.cache[key] = result
            return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("flat", "conv"), required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--checkpoint-manifest")
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--autoencoder-type", choices=("spatial", "vector"), default="spatial")
    parser.add_argument("--latent-cache", required=True)
    parser.add_argument("--objectives", nargs="+")
    parser.add_argument("--annotations")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--rank-examples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    feature_server = SpatialFeatureServer(args)
    objectives = feature_server.objectives
    options = "".join(
        f'<option value="{html.escape(objective)}">{html.escape(objective)}</option>'
        for objective in objectives
    )
    page = (
        HTML.replace("__LABEL__", html.escape(args.label))
        .replace("__OPTIONS__", options)
        .replace("__OBJECTIVES__", json.dumps(objectives))
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                payload = page.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(payload)
                return
            if parsed.path == "/api/feature":
                try:
                    query = parse_qs(parsed.query)
                    objective = query.get("objective", [objectives[0]])[0]
                    rank = int(query.get("rank", ["0"])[0])
                    payload = json.dumps(feature_server.feature(objective, rank)).encode()
                    self.send_response(200)
                except Exception as error:
                    payload = json.dumps({"error": str(error)}).encode()
                    self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)
                return
            if parsed.path == "/api/annotations":
                payload = feature_server.annotations.export().encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Disposition", "attachment; filename=feature_annotations.json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/api/annotation":
                self.send_response(404)
                self.end_headers()
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                record = feature_server.annotations.update(json.loads(self.rfile.read(length)))
                payload = json.dumps({"annotation": record}).encode()
                self.send_response(200)
            except Exception as error:
                payload = json.dumps({"error": str(error)}).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            return

    print(f"GUI ready at http://0.0.0.0:{args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
