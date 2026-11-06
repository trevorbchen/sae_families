#!/usr/bin/env python3
"""Review full-latent and canonical residual-SAE feature interventions."""

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
from train_independent_routed_union_sae import load_independent_routed_union
from train_grouped_objective_sae import GroupSpec, PartitionedTopKSAE
from train_spatial_map_sae import FlatTopKSAE


STAGES = ("laplacian", "fourier_low", "fourier_mid", "fourier_high")
STAGE_LABELS = {
    "laplacian": "Laplacian",
    "fourier_low": "Fourier low",
    "fourier_mid": "Fourier mid",
    "fourier_high": "Fourier high",
}
MULTIPLIERS = (-10, -4, -2, 0, 2, 4, 10)


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>SAE feature review</title>
<style>
:root{color-scheme:dark;--bg:#101112;--panel:#1b1d1f;--line:#3c4145;--muted:#a8b0b5;--text:#f1f3f4;--accent:#e1a52b;--action:#168a84}
*{box-sizing:border-box}body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--text);margin:0;padding:18px 20px 32px}
body.paper-candidate:before{content:"";position:fixed;z-index:20;left:0;top:0;bottom:0;width:6px;background:var(--accent)}
header{position:sticky;z-index:10;top:0;display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:10px 0 12px;background:var(--bg);border-bottom:1px solid var(--line)}
strong{margin-right:6px}button,input,select,textarea,a.command{font:inherit;background:#202326;color:var(--text);border:1px solid #555d62;padding:7px 10px;border-radius:4px}
button,a.command{cursor:pointer;text-decoration:none}button:hover,a.command:hover{background:#2a2e31}button:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible{outline:2px solid #58b9b2;outline-offset:2px}
input[type=number]{width:88px}.nav{width:38px;padding-left:0;padding-right:0;font-size:18px}.primary{background:#126e69;border-color:#299d96}.primary:hover{background:#16827c}
#interesting[aria-pressed=true]{background:var(--accent);border-color:#f4c35b;color:#17130a;font-weight:700}#candidateBadge{display:none;color:#15120b;background:#f0bd4d;padding:5px 9px;border-radius:4px;font-weight:700}body.paper-candidate #candidateBadge{display:inline-block}
#status{color:var(--muted)}#summary{margin:12px 0;color:#c9ced1;min-height:22px}.annotation{display:grid;grid-template-columns:auto minmax(260px,1fr) auto;gap:10px;align-items:stretch;margin:12px 0 16px;max-width:1800px}
textarea{width:100%;min-height:48px;resize:vertical;line-height:1.35}.grid{display:grid;grid-template-columns:repeat(8,minmax(118px,1fr));gap:9px;max-width:1800px}
figure{margin:0;background:var(--panel);padding:7px;border:1px solid #24282a;border-radius:5px;overflow:hidden}img{width:100%;display:block;aspect-ratio:1;object-fit:cover}figcaption{font-size:12px;color:#bec4c8;margin-top:6px;min-height:18px}
@media(max-width:900px){.grid{grid-template-columns:repeat(4,minmax(110px,1fr))}.annotation{grid-template-columns:1fr}.annotation button,.annotation a{width:100%}}
@media(max-width:560px){body{padding:10px}.grid{grid-template-columns:repeat(2,minmax(120px,1fr))}}
</style></head><body>
<header><strong>SAE feature review</strong>
<span>source</span><select id="source">__SOURCE_OPTIONS__</select>
<span id="componentWrap" style="display:none">branch <select id="component"><option value="">All objective branches</option><option value="laplacian">Laplacian</option><option value="fourier_low">Fourier low</option><option value="fourier_mid">Fourier mid</option><option value="fourier_high">Fourier high</option></select></span>
<span>importance rank</span><button id="prev" class="nav" title="Previous feature">&#8592;</button><input id="rank" type="number" min="0" value="0">
<button id="next" class="nav" title="Next feature">&#8594;</button><button id="load" class="primary">Load</button>
<span id="candidateBadge">Paper candidate</span><span id="status">ready</span></header>
<div id="summary"></div>
<div class="annotation"><button id="interesting" aria-pressed="false" title="Toggle paper candidate">Interesting</button><textarea id="comment" placeholder="Feature notes"></textarea><div><button id="save" title="Save note">Save note</button> <a class="command" href="/api/annotations" download="sae_feature_annotations.json">Export</a></div></div>
<div id="grid" class="grid"></div>
<script>
const $=id=>document.getElementById(id),sources=__SOURCES__,cache=new Map(),inflight=new Map();
let current=null,dirty=false,saveTimer=null,saveInFlight=null,requestToken=0;
function selectedComponent(){return $('source').value==='tandem_canonical'?$('component').value:''}
function key(source,rank,component=''){return source+':'+component+':'+rank}
function updateComponent(){ $('componentWrap').style.display=$('source').value==='tandem_canonical'?'inline-flex':'none' }
function annotationKey(d){return d.annotation_key||d.source+':'+d.feature}
function setCandidate(value){$('interesting').setAttribute('aria-pressed',value?'true':'false');document.body.classList.toggle('paper-candidate',value)}
function render(d){current=d;dirty=false;let a=d.annotation||{};setCandidate(!!a.interesting);$('comment').value=a.comment||'';
 $('summary').textContent=d.source_label+' | rank '+d.rank+(d.component_label?' | '+d.component_label:'')+' | feature '+d.feature+' | dictionary '+d.dictionary_size+' / k='+d.k+' | active RMS '+d.active_rms.toFixed(5)+' | active '+(100*d.dataset_active_fraction).toFixed(2)+'% | selected active '+d.selected_active+' / '+d.examples;
 $('grid').innerHTML=d.images.map((x,i)=>x.variants.map(v=>'<figure><img src="'+v.image+'"><figcaption>example '+i+' '+v.label+'</figcaption></figure>').join('')+'<figure><img src="'+x.heatmap+'"><figcaption>example '+i+' output effect</figcaption></figure>').join('')}
async function fetchResult(source,rank,component=''){let k=key(source,rank,component);if(cache.has(k))return cache.get(k);if(inflight.has(k))return inflight.get(k);
 let url='/api/feature?source='+encodeURIComponent(source)+'&rank='+rank+(component?'&component='+encodeURIComponent(component):'');let promise=(async()=>{let response=await fetch(url),data=await response.json();if(data.error)throw Error(data.error);cache.set(k,data);return data})().finally(()=>inflight.delete(k));inflight.set(k,promise);return promise}
async function saveAnnotation(){clearTimeout(saveTimer);if(saveInFlight)await saveInFlight;if(!current||!dirty)return;
 const target=current,targetKey=key(target.source,target.rank,target.requested_component||''),payload={key:annotationKey(target),source:target.source,source_label:target.source_label,rank:target.rank,feature:target.feature,interesting:$('interesting').getAttribute('aria-pressed')==='true',comment:$('comment').value};dirty=false;
 saveInFlight=(async()=>{let response=await fetch('/api/annotation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}),data=await response.json();if(data.error)throw Error(data.error);target.annotation=data.annotation;cache.set(targetKey,target);if(current===target)$('status').textContent='note saved'})();
 try{await saveInFlight}catch(error){if(current===target)dirty=true;throw error}finally{saveInFlight=null}if(current===target&&dirty)return saveAnnotation()}
function queueSave(){dirty=true;clearTimeout(saveTimer);saveTimer=setTimeout(()=>saveAnnotation().catch(e=>$('status').textContent='save error: '+e.message),700)}
async function load(){try{await saveAnnotation();let rank=Math.max(0,parseInt($('rank').value||0)),source=$('source').value,component=selectedComponent(),token=++requestToken;$('status').textContent='loading '+sources[source]+' rank '+rank+'...';let data=await fetchResult(source,rank,component);if(token!==requestToken)return;render(data);$('status').textContent='loaded';void prefetch(source,rank,component)}catch(e){$('status').textContent='error: '+e.message}}
async function prefetch(source,rank,component){for(let next=rank+1;next<=rank+8;next++){try{await fetchResult(source,next,component)}catch(_){return}}for(let [k,v] of cache)if(v.source===source&&(v.requested_component||'')===component&&(v.rank<rank-4||v.rank>rank+10))cache.delete(k)}
async function moveRank(delta){await saveAnnotation();$('rank').value=Math.max(0,parseInt($('rank').value||0)+delta);load()}
async function moveSource(delta){await saveAnnotation();let select=$('source'),count=select.options.length;select.selectedIndex=(select.selectedIndex+delta+count)%count;$('rank').value=0;updateComponent();load()}
async function toggleInteresting(){if(!current)return;setCandidate($('interesting').getAttribute('aria-pressed')!=='true');dirty=true;await saveAnnotation()}
$('load').onclick=load;$('rank').onchange=load;$('source').onchange=()=>{$('rank').value=0;updateComponent();load()};$('component').onchange=()=>{$('rank').value=0;load()};$('prev').onclick=()=>moveRank(-1);$('next').onclick=()=>moveRank(1);$('interesting').onclick=()=>toggleInteresting().catch(e=>$('status').textContent='save error: '+e.message);$('save').onclick=()=>saveAnnotation().catch(e=>$('status').textContent='save error: '+e.message);$('comment').oninput=queueSave;
$('comment').addEventListener('keydown',event=>{if((event.ctrlKey||event.metaKey)&&event.key==='Enter'){event.preventDefault();saveAnnotation().then(()=>$('comment').blur())}else if(event.key==='Escape'){event.preventDefault();saveAnnotation().then(()=>$('comment').blur())}});
document.addEventListener('keydown',event=>{if(['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName))return;if(event.key==='ArrowLeft'){event.preventDefault();moveRank(-1)}else if(event.key==='ArrowRight'){event.preventDefault();moveRank(1)}else if(event.key==='ArrowUp'){event.preventDefault();moveSource(-1)}else if(event.key==='ArrowDown'){event.preventDefault();moveSource(1)}else if(event.key.toLowerCase()==='i'){event.preventDefault();toggleInteresting()}else if(event.key.toLowerCase()==='c'){event.preventDefault();$('comment').focus();$('comment').setSelectionRange($('comment').value.length,$('comment').value.length)}});
window.addEventListener('beforeunload',()=>{if(current&&dirty){let payload={key:annotationKey(current),source:current.source,source_label:current.source_label,rank:current.rank,feature:current.feature,interesting:$('interesting').getAttribute('aria-pressed')==='true',comment:$('comment').value};navigator.sendBeacon('/api/annotation',new Blob([JSON.stringify(payload)],{type:'application/json'}))}});updateComponent();load();
</script></body></html>"""


def png_data(image):
    image = image.detach().float().cpu().clamp(0, 1)
    buffer = io.BytesIO()
    array = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).numpy()
    Image.fromarray(array).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def load_model(path, input_dim, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    specs = tuple(GroupSpec(**record) for record in checkpoint["groups"])
    model = PartitionedTopKSAE(input_dim, specs)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False).to(device)
    return model, checkpoint


def load_flat_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = FlatTopKSAE(
        checkpoint["input_shape"], config["feature_count"], config["k"]
    )
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False).to(device)
    model.input_dim = model.d
    model.dict_size = model.h
    model.total_k = model.k
    model.specs = (GroupSpec("features", config["objective"], model.h, model.k),)
    model.ranges = {"features": (0, model.h)}
    return model, checkpoint


class AnnotationStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        if self.path.is_file():
            self.records = json.loads(self.path.read_text())
        else:
            self.records = {}

    def get(self, key):
        with self.lock:
            return dict(self.records.get(key, {}))

    def update(self, payload):
        key = payload["key"]
        record = {
            "source": payload["source"],
            "source_label": payload["source_label"],
            "rank": int(payload["rank"]),
            "feature": int(payload["feature"]),
            "interesting": bool(payload.get("interesting", False)),
            "comment": str(payload.get("comment", "")),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with self.lock:
            self.records[key] = record
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(json.dumps(self.records, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, self.path)
        return dict(record)

    def export(self):
        with self.lock:
            return json.dumps(self.records, indent=2, sort_keys=True) + "\n"


class ResidualFeatureServer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder = load_spatial_autoencoder(args.autoencoder, self.device)
        self.autoencoder.eval().requires_grad_(False)
        cache = torch.load(args.latent_cache, map_location="cpu", weights_only=False)
        self.latents = cache["latents"].float()
        train_indices = cache.get("train_indices", torch.arange(len(self.latents)))
        self.rank_indices = train_indices[:min(args.rank_examples, len(train_indices))]

        self.direct, direct_state = load_model(
            args.direct_mse, self.latents[0].numel(), self.device
        )
        self.mean = direct_state["input_mean"].float().to(self.device)
        self.std = direct_state["input_std"].float().to(self.device)
        self.input_shape = tuple(direct_state["input_shape"])

        self.grouped_objective = {}
        self.grouped_normalization = {}
        for stage in STAGES:
            path = (
                Path(args.grouped_objective_root)
                / f"weighted_{stage}" / args.grouped_objective_seed / "last.pt"
            )
            model, state = load_flat_model(path, self.device)
            self.grouped_objective[stage] = model
            self.grouped_normalization[stage] = (
                state["channel_mean"].float().to(self.device)[:, None, None],
                state["channel_std"].float().to(self.device)[:, None, None],
            )
        tandem, tandem_state = load_independent_routed_union(
            args.tandem_checkpoint, self.device
        )
        self.tandem = {stage: tandem.branches[stage] for stage in STAGES}
        self.tandem_normalization = (
            tandem_state["channel_mean"].float().to(self.device)[:, None, None],
            tandem_state["channel_std"].float().to(self.device)[:, None, None],
        )
        for stage in STAGES:
            self.tandem[stage].dict_size = self.tandem[stage].h
            self.tandem[stage].total_k = self.tandem[stage].k
            if self.tandem[stage].input_shape != self.input_shape:
                raise ValueError(f"tandem {stage} input shape differs from direct MSE")

        self.sources = {
            "direct_mse": {
                "label": "Direct MSE (16384/k64)", "family": "direct",
                "model": self.direct, "prefix": (),
            }
        }
        for stage in STAGES:
            self.sources[f"grouped_objective_{stage}"] = {
                "label": f"Grouped objective / {STAGE_LABELS[stage]}",
                "family": "grouped_objective", "stage": stage,
                "model": self.grouped_objective[stage],
                "prefix": (),
            }
        for stage in STAGES:
            self.sources[f"tandem_{stage}"] = {
                "label": f"Tandem / {STAGE_LABELS[stage]}",
                "family": "tandem", "stage": stage,
                "model": self.tandem[stage],
            }
        self.public_sources = {
            key: value for key, value in self.sources.items()
            if not key.startswith("tandem_")
        }
        self.public_sources["tandem_canonical"] = {
            "label": "Tandem 4-objective SAE (seed 2030)",
        }

        self.stats = {}
        self.results = {}
        self.model_lock = threading.Lock()
        self.annotations = AnnotationStore(args.annotations)
        for source in self.sources:
            print(f"precomputing ranking: {source}", flush=True)
            self.prepare_stats(source)
        self.prepare_tandem_canonical_stats()
        if args.precompute_ranks:
            for source in self.sources:
                print(f"precomputing examples: {source}", flush=True)
                self.prepare_examples(source, args.precompute_ranks)
        print("all SAE rankings loaded", flush=True)

    def normalization(self, source):
        spec = self.sources[source]
        if spec["family"] == "grouped_objective":
            return self.grouped_normalization[spec["stage"]]
        if spec["family"] == "tandem":
            return self.tandem_normalization
        return self.mean, self.std

    def standardize(self, raw, source):
        mean, std = self.normalization(source)
        return ((raw.to(self.device).float() - mean) / std).flatten(1)

    def unstandardize(self, standardized, source):
        mean, std = self.normalization(source)
        return standardized.view(-1, *self.input_shape) * std + mean

    def incoming(self, source, standardized):
        return standardized

    @staticmethod
    def selected_codes(model, inputs):
        if isinstance(model, FlatTopKSAE):
            activations = F.relu(F.linear(
                inputs - model.dec_bias, model.encoder, model.encoder_bias
            ))
            return torch.topk(activations, model.k, dim=1)
        if len(model.specs) != 1:
            raise ValueError("review sources must each contain one TopK group")
        spec = model.specs[0]
        start, end = model.ranges[spec.name]
        activations = F.relu(model.encoder(inputs - model.dec_bias)[:, start:end])
        values, local_indices = torch.topk(activations, spec.k, dim=1)
        return values, local_indices + start

    @torch.no_grad()
    def prepare_stats(self, source):
        if source in self.stats:
            return self.stats[source]
        model = self.sources[source]["model"]
        sums = torch.zeros(model.dict_size, device=self.device)
        squares = torch.zeros(model.dict_size, device=self.device)
        active = torch.zeros(model.dict_size, device=self.device)
        observations = 0
        for start in range(0, len(self.rank_indices), self.args.batch_size):
            selected = self.rank_indices[start:start + self.args.batch_size]
            standardized = self.standardize(self.latents[selected], source)
            incoming = self.incoming(source, standardized)
            values, indices = self.selected_codes(model, incoming)
            positive = values > 0
            ids, vals = indices[positive], values[positive]
            sums.scatter_add_(0, ids, vals)
            squares.scatter_add_(0, ids, vals.square())
            active.scatter_add_(0, ids, torch.ones_like(vals))
            observations += len(selected)
        self.stats[source] = {
            "order": torch.argsort(sums, descending=True).tolist(),
            "sums": sums,
            "fraction": active / observations,
            "active_rms": (squares / active.clamp_min(1)).sqrt(),
        }
        return self.stats[source]

    def prepare_tandem_canonical_stats(self):
        source = "tandem_canonical"
        if source in self.stats:
            return self.stats[source]
        entries = []
        for stage_index, stage in enumerate(STAGES):
            internal_source = f"tandem_{stage}"
            stage_stats = self.stats[internal_source]
            entries.extend(
                (score, stage_index, feature, internal_source)
                for feature, score in enumerate(stage_stats["sums"].tolist())
            )
        entries.sort(reverse=True)
        self.stats[source] = {
            "order": [(internal_source, feature) for _, _, feature, internal_source in entries],
        }
        return self.stats[source]

    @torch.no_grad()
    def prepare_examples(self, source, rank_limit):
        """Cache high-activation examples without rescanning the dataset per feature."""
        stats = self.prepare_stats(source)
        if "example_indices" in stats:
            return stats["example_indices"]
        model = self.sources[source]["model"]
        features = stats["order"][:min(rank_limit, len(stats["order"]))]
        feature_to_slot = torch.full(
            (model.dict_size,), -1, dtype=torch.long, device=self.device
        )
        feature_to_slot[torch.tensor(features, device=self.device)] = torch.arange(
            len(features), device=self.device
        )
        candidates = [[] for _ in features]
        for start in range(0, len(self.latents), self.args.batch_size):
            raw = self.latents[start:start + self.args.batch_size]
            standardized = self.standardize(raw, source)
            values, indices = self.selected_codes(model, self.incoming(source, standardized))
            slots = feature_to_slot[indices]
            rows = torch.arange(len(raw), device=self.device)[:, None].expand_as(indices)
            selected = (slots >= 0) & (values > 0)
            triples = torch.stack((
                slots[selected], values[selected], rows[selected] + start,
            ), dim=1).cpu().tolist()
            for slot, value, index in triples:
                candidates[int(slot)].append((float(value), int(index)))
        example_indices = {}
        for feature, feature_candidates in zip(features, candidates):
            feature_candidates.sort(reverse=True)
            example_indices[feature] = [
                index for _, index in feature_candidates[:self.args.examples]
            ]
        stats["example_indices"] = example_indices
        return example_indices

    @torch.no_grad()
    def feature(self, source, rank, component=""):
        if source not in self.public_sources:
            raise ValueError(f"unknown source: {source}")
        rank = int(rank)
        with self.model_lock:
            if source == "tandem_canonical":
                if component:
                    if component not in STAGES:
                        raise ValueError(f"unknown tandem component: {component}")
                    internal_source = f"tandem_{component}"
                    component_stats = self.stats[internal_source]
                    if rank < 0 or rank >= len(component_stats["order"]):
                        raise ValueError(
                            f"rank must be in [0, {len(component_stats['order']) - 1}]"
                        )
                    feature = component_stats["order"][rank]
                else:
                    canonical_stats = self.prepare_tandem_canonical_stats()
                    if rank < 0 or rank >= len(canonical_stats["order"]):
                        raise ValueError(
                            f"rank must be in [0, {len(canonical_stats['order']) - 1}]"
                        )
                    internal_source, feature = canonical_stats["order"][rank]
                stats = self.stats[internal_source]
                stage = internal_source.removeprefix("tandem_")
                component_label = f"{STAGE_LABELS[stage]} branch"
                public_label = self.public_sources[source]["label"]
                annotation_key = f"{source}:{stage}:{feature}"
            else:
                internal_source = source
                stats = self.prepare_stats(source)
                if rank < 0 or rank >= len(stats["order"]):
                    raise ValueError(f"rank must be in [0, {len(stats['order']) - 1}]")
                feature = stats["order"][rank]
                component_label = ""
                public_label = self.sources[source]["label"]
                annotation_key = f"{source}:{feature}"
                component = ""
            result_key = (source, component, internal_source, feature)
            if result_key in self.results:
                result = dict(self.results[result_key])
                result["annotation"] = self.annotations.get(annotation_key)
                return result

            model = self.sources[internal_source]["model"]
            example_indices = stats.get("example_indices", {}).get(feature, [])
            if not example_indices:
                candidates = []
                for start in range(0, len(self.latents), self.args.batch_size):
                    raw = self.latents[start:start + self.args.batch_size]
                    standardized = self.standardize(raw, internal_source)
                    incoming = self.incoming(internal_source, standardized)
                    values, indices = self.selected_codes(model, incoming)
                    coefficients = (values * (indices == feature)).sum(1)
                    candidates.extend(
                        (float(value), start + row)
                        for row, value in enumerate(coefficients)
                        if value > 0
                    )
                candidates.sort(reverse=True)
                example_indices = [index for _, index in candidates[:self.args.examples]]
            if not example_indices:
                raise ValueError("feature has no active examples")

            raw = self.latents[example_indices].to(self.device).float()
            standardized = self.standardize(raw, internal_source)
            incoming = self.incoming(internal_source, standardized)
            values, indices = self.selected_codes(model, incoming)
            coefficients = (values * (indices == feature)).sum(1)
            scale = stats["active_rms"][feature].clamp_min(1e-6)
            variants = []
            for multiplier in MULTIPLIERS:
                if multiplier == 0:
                    modified_raw = raw
                else:
                    target = torch.clamp(coefficients + multiplier * scale, min=0)
                    delta = (target - coefficients)[:, None] * model.decoder[feature]
                    modified = standardized + delta
                    modified_raw = self.unstandardize(modified, internal_source)
                decoded = self.autoencoder.decode(modified_raw).add(1).div(2).clamp(0, 1)
                variants.append((multiplier, decoded))
            effect = (variants[-1][1] - variants[0][1]).abs()
            effect = effect / effect.flatten(1).amax(1).clamp_min(1e-6)[:, None, None, None]
            result = {
                "source": source,
                "source_label": public_label,
                "requested_component": component,
                "component_label": component_label,
                "annotation_key": annotation_key,
                "rank": rank,
                "feature": feature,
                "dictionary_size": (
                    sum(self.tandem[stage].dict_size for stage in STAGES)
                    if source == "tandem_canonical" else model.dict_size
                ),
                "k": (
                    sum(self.tandem[stage].total_k for stage in STAGES)
                    if source == "tandem_canonical" else model.total_k
                ),
                "examples": len(raw),
                "dataset_active_fraction": float(stats["fraction"][feature]),
                "active_rms": float(scale),
                "selected_active": int((coefficients > 0).sum()),
                "example_indices": example_indices,
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
            self.results[result_key] = result
            result = dict(result)
            result["annotation"] = self.annotations.get(annotation_key)
            return result


def source_options(sources):
    groups = (
        ("Direct MSE", ("direct_mse",)),
        ("Grouped objective SAE", tuple(f"grouped_objective_{stage}" for stage in STAGES)),
        ("Tandem 4-objective SAE", ("tandem_canonical",)),
    )
    return "".join(
        f'<optgroup label="{html.escape(group)}">'
        + "".join(
            f'<option value="{html.escape(key)}">{html.escape(sources[key]["label"])}</option>'
            for key in keys
        )
        + "</optgroup>"
        for group, keys in groups
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-mse", required=True)
    parser.add_argument("--grouped-objective-root", required=True)
    parser.add_argument("--grouped-objective-seed", default="seed_2030")
    parser.add_argument("--tandem-checkpoint", required=True)
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--latent-cache", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--port", type=int, default=8812)
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--rank-examples", type=int, default=4096)
    parser.add_argument(
        "--precompute-ranks", type=int, default=0,
        help="cache top examples for this many activity-ranked features per internal source",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    server = ResidualFeatureServer(args)
    labels = {key: value["label"] for key, value in server.public_sources.items()}
    page = (
        HTML.replace("__SOURCE_OPTIONS__", source_options(server.public_sources))
        .replace("__SOURCES__", json.dumps(labels))
    )

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, payload, status=200):
            encoded = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                encoded = page.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return
            if parsed.path == "/api/feature":
                try:
                    query = parse_qs(parsed.query)
                    source = query.get("source", ["direct_mse"])[0]
                    rank = int(query.get("rank", ["0"])[0])
                    component = query.get("component", [""])[0]
                    self.send_json(server.feature(source, rank, component))
                except Exception as error:
                    self.send_json({"error": str(error)}, 400)
                return
            if parsed.path == "/api/annotations":
                encoded = server.annotations.export().encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header(
                    "Content-Disposition",
                    'attachment; filename="sae_feature_annotations.json"',
                )
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            if urlparse(self.path).path != "/api/annotation":
                self.send_response(404)
                self.end_headers()
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                self.send_json({"annotation": server.annotations.update(payload)})
            except Exception as error:
                self.send_json({"error": str(error)}, 400)

        def log_message(self, *_):
            return

    print(f"GUI ready at http://0.0.0.0:{args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
