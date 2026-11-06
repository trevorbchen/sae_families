#!/usr/bin/env python3
"""GUI for permuting one matched512 lambda-sweep SAE feature and watching the
decoded image change, across all 5 objectives at one lambda.

Ported from spatial_sae_feature_gui.py's interaction pattern (client-side
result caching, background prefetch of the next 10 features, keyboard
hotkeys, an interesting/comment annotation system with autosave + export)
but for this project's plain feed-forward vector/spatial autoencoder
instead of a diffusion model -- no noise/T needed, decode is a single
forward pass.
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
from vector_conv_autoencoder import load_vector_autoencoder

OBJECTIVE_DIRS = ("latent_mse", "laplacian", "fourier_low", "fourier_mid", "fourier_high")
OBJECTIVE_LABELS = {
    "latent_mse": "MSE", "laplacian": "Laplacian", "fourier_low": "Fourier low",
    "fourier_mid": "Fourier mid", "fourier_high": "Fourier high",
}
DEFAULT_SEEDS = (2030, 2031, 2032)
MULTIPLIERS = (-10, -4, -2, 0, 2, 4, 10)

HTML_PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>__LABEL__ feature permutation</title>
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
figure{margin:0;background:#1c1c1c;padding:8px;border-radius:5px} img{width:100%;display:block}
figcaption{font-size:12px;color:#bbb;margin-top:6px}
@media(max-width:700px){.grid{grid-template-columns:repeat(2,minmax(140px,1fr))}.annotation{grid-template-columns:1fr}}
</style></head><body>
<header><strong>__LABEL__ feature permutation</strong>
<span>objective</span><select id="objective">__OBJECTIVE_OPTIONS__</select>
<span>seed</span><select id="seed">__SEED_OPTIONS__</select>
<span>importance rank</span><button id="prev" title="Previous feature (&#8592;)">&#8592;</button><input id="feature" type="number" min="0" value="0">
<button id="next" title="Next feature (&#8594;)">&#8594;</button><button id="load">Load</button>
<span id="status">ready</span></header>
<div id="summary"></div>
<div class="annotation"><button id="interesting" aria-pressed="false" title="Toggle interesting (I)">Interesting</button><textarea id="comment" placeholder="Feature notes (C to focus)"></textarea><div><button id="save">Save note</button> <a class="command" href="/api/annotations" download="matched512_feature_annotations.json">Export</a></div></div>
<div id="grid" class="grid"></div>
<script>
const $=id=>document.getElementById(id);
const results=new Map(),inflight=new Map();
let requestToken=0,current=null,dirty=false,saveTimer=null,saveInFlight=null;
// Feature IDs are not comparable across seeds -- each seed is an
// independently-trained SAE with no alignment between them -- so the seed
// must be part of every cache/annotation key, or "feature 5228" from one
// seed would silently collide with an unrelated "feature 5228" from another.
function resultKey(objective,seed,rank){return objective+':'+seed+':'+rank}

function render(d){
 current=d;dirty=false;let a=d.annotation||{};
 $('interesting').setAttribute('aria-pressed',a.interesting?'true':'false');
 document.body.classList.toggle('paper-candidate',!!a.interesting);
 $('comment').value=a.comment||'';
 $('summary').textContent='objective '+d.objective+' | seed '+d.seed+' | rank '+d.rank+' | feature '+d.feature+' | dataset mean activation '+d.dataset_mean_activation.toFixed(5)+' | dataset std '+d.dataset_std_activation.toFixed(5)+' | active '+(100*d.dataset_active_fraction).toFixed(2)+'% | selected active '+d.selected_active+' / '+d.examples;
 $('grid').innerHTML=d.images.map((x,i)=>x.variants.map(v=>'<figure><img src="'+v.image+'"><figcaption>example '+i+' '+v.label+'</figcaption></figure>').join('')+'<figure><img src="'+x.heatmap+'"><figcaption>example '+i+' spatial effect (+10std vs -10std)</figcaption></figure>').join('');
}

async function saveAnnotation(){
 clearTimeout(saveTimer);if(saveInFlight)await saveInFlight;if(!current||!dirty)return;
 const target=current,payload={key:target.annotation_key,objective:target.objective,seed:target.seed,rank:target.rank,feature:target.feature,interesting:$('interesting').getAttribute('aria-pressed')==='true',comment:$('comment').value};
 dirty=false;
 saveInFlight=(async()=>{
  let response=await fetch('/api/annotation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}),data=await response.json();
  if(data.error)throw Error(data.error);
  target.annotation=data.annotation;results.set(resultKey(target.objective,target.seed,target.rank),target);$('status').textContent='note saved';
 })();
 try{await saveInFlight}catch(error){dirty=true;throw error}finally{saveInFlight=null}
}
function queueSave(){dirty=true;clearTimeout(saveTimer);saveTimer=setTimeout(()=>saveAnnotation().catch(e=>$('status').textContent='save error: '+e.message),700)}
async function toggleInteresting(){
 if(!current)return;let value=$('interesting').getAttribute('aria-pressed')!=='true';
 $('interesting').setAttribute('aria-pressed',value?'true':'false');document.body.classList.toggle('paper-candidate',value);dirty=true;await saveAnnotation();
}

async function fetchResult(objective,seed,rank){
 let key=resultKey(objective,seed,rank);if(results.has(key))return results.get(key);
 if(inflight.has(key))return inflight.get(key);
 let promise=(async()=>{
  let response=await fetch('/api/feature?objective='+encodeURIComponent(objective)+'&seed='+seed+'&rank='+rank),data=await response.json();
  if(data.error)throw Error(data.error);results.set(key,data);return data;
 })().finally(()=>inflight.delete(key));
 inflight.set(key,promise);return promise;
}
function trimCache(center,activeObjective,activeSeed){
 for(let [key,data] of results){
  if(data.objective!==activeObjective||data.seed!==activeSeed||data.rank<center-3||data.rank>center+10)results.delete(key);
 }
}
// Background, non-blocking prefetch of the next 10 features -- the current
// selection renders immediately from fetchResult, this just warms the
// cache afterward so pressing "next" repeatedly feels instant.
async function prefetchNext(rank,activeObjective,activeSeed,token){
 try{
  for(let next=rank+1;next<=rank+10;next++){
   if(token!==requestToken)return;
   await fetchResult(activeObjective,activeSeed,next);
   if(parseInt($('feature').value||0)===rank&&$('objective').value===activeObjective&&parseInt($('seed').value)===activeSeed)
    $('status').textContent='loaded next '+(next-rank)+' / 10 features';
  }
  trimCache(rank,activeObjective,activeSeed);
  if(parseInt($('feature').value||0)===rank&&$('objective').value===activeObjective&&parseInt($('seed').value)===activeSeed)
   $('status').textContent='current and next 10 features loaded';
 }catch(e){
  if(parseInt($('feature').value||0)===rank)$('status').textContent='current loaded; prefetch error: '+e.message;
 }
}
async function load(){
 await saveAnnotation();
 let rank=Math.max(0,parseInt($('feature').value||0)),token=++requestToken,objective=$('objective').value,seed=parseInt($('seed').value);
 $('status').textContent='loading '+objective+' seed '+seed+' rank '+rank+'...';
 try{
  let selected=await fetchResult(objective,seed,rank);if(token!==requestToken)return;
  render(selected);$('status').textContent='loaded; prefetching next 10 features';
  trimCache(rank,objective,seed);void prefetchNext(rank,objective,seed,token);
 }catch(e){if(token===requestToken)$('status').textContent='error: '+e.message}
}
async function showSelection(){
 await saveAnnotation();
 let rank=Math.max(0,parseInt($('feature').value||0)),seed=parseInt($('seed').value),data=results.get(resultKey($('objective').value,seed,rank));
 if(data){render(data);$('status').textContent='feature loaded'}else load();
}
async function moveFeature(delta){await saveAnnotation();$('feature').value=Math.max(0,parseInt($('feature').value||0)+delta);load()}
async function moveObjective(delta){
 await saveAnnotation();let select=$('objective'),count=select.options.length;
 select.selectedIndex=(select.selectedIndex+delta+count)%count;showSelection();
}
async function moveSeed(delta){
 await saveAnnotation();let select=$('seed'),count=select.options.length;
 select.selectedIndex=(select.selectedIndex+delta+count)%count;showSelection();
}

$('load').onclick=load;$('feature').onchange=load;$('objective').onchange=showSelection;$('seed').onchange=showSelection;
$('prev').onclick=()=>moveFeature(-1);$('next').onclick=()=>moveFeature(1);
$('interesting').onclick=()=>toggleInteresting().catch(e=>$('status').textContent='save error: '+e.message);
$('save').onclick=()=>saveAnnotation().catch(e=>$('status').textContent='save error: '+e.message);
$('comment').oninput=queueSave;
$('comment').addEventListener('keydown',event=>{
 if((event.ctrlKey||event.metaKey)&&event.key==='Enter'){event.preventDefault();saveAnnotation().then(()=>$('comment').blur())}
 else if(event.key==='Escape'){event.preventDefault();saveAnnotation().then(()=>$('comment').blur())}
});
document.addEventListener('keydown',event=>{
 if(['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName))return;
 if(event.key==='ArrowLeft'){event.preventDefault();moveFeature(-1)}
 else if(event.key==='ArrowRight'){event.preventDefault();moveFeature(1)}
 else if(event.key==='ArrowUp'){event.preventDefault();moveObjective(-1)}
 else if(event.key==='ArrowDown'){event.preventDefault();moveObjective(1)}
 else if(event.key==='['){event.preventDefault();moveSeed(-1)}
 else if(event.key===']'){event.preventDefault();moveSeed(1)}
 else if(event.key.toLowerCase()==='i'){event.preventDefault();toggleInteresting()}
 else if(event.key.toLowerCase()==='c'){event.preventDefault();$('comment').focus()}
});
window.addEventListener('beforeunload',()=>{
 if(current&&dirty){
  let payload={key:current.annotation_key,objective:current.objective,seed:current.seed,rank:current.rank,feature:current.feature,interesting:$('interesting').getAttribute('aria-pressed')==='true',comment:$('comment').value};
  navigator.sendBeacon('/api/annotation',new Blob([JSON.stringify(payload)],{type:'application/json'}));
 }
});
load();
</script></body></html>"""

# Gallery of every feature currently marked "interesting", grouped by
# objective, for picking a final paper-ready subset -- separate from the
# rank-by-rank browsing page above. Each card gets a capped number of
# example rows (not the full --examples count) to keep the page a
# reasonable size across dozens of features at once.
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
.example-row{border:1px solid #2a2a2a;border-radius:4px;padding:8px;margin-top:8px}
.example-row.selected-example{border-color:#3fa9f5;box-shadow:0 0 0 1px #3fa9f5}
.example-row-header{display:flex;justify-content:space-between;align-items:center;font-size:12px;color:#999;margin-bottom:6px}
label.pick{display:flex;align-items:center;gap:6px;cursor:pointer;color:#ddd}
.grid{display:grid;grid-template-columns:repeat(8,minmax(100px,1fr));gap:8px}
figure{margin:0;background:#111;padding:6px;border-radius:4px}img{width:100%;display:block}
figcaption{font-size:10px;color:#999;margin-top:4px}
a.command{color:#e1a52b}
</style></head><body>
<h1>__LABEL__ -- shortlist (marked interesting)</h1>
<label style="display:flex;align-items:center;gap:6px;font-size:13px;color:#ddd;margin-bottom:12px;cursor:pointer">
 <input type="checkbox" id="chosenOnly" checked> chosen for paper only (uncheck to see all interesting)</label>
<div id="status">loading...</div>
<div id="sections"></div>
<script>
const $=id=>document.getElementById(id);
async function saveChoice(item,chosen){
 let payload={key:item.annotation_key,objective:item.objective,seed:item.seed,rank:item.rank,feature:item.feature,interesting:true,chosen};
 let response=await fetch('/api/annotation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
 let data=await response.json();if(data.error)throw Error(data.error);return data.annotation;
}
// Neither of these sends "comment" -- the server preserves whatever comment
// was already there, the same way it preserves "chosen" when an
// example-pick save omits it, and vice versa.
async function saveExampleChoice(item,index){
 let payload={key:item.annotation_key,objective:item.objective,seed:item.seed,rank:item.rank,feature:item.feature,interesting:true,chosen_example_index:index};
 let response=await fetch('/api/annotation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
 let data=await response.json();if(data.error)throw Error(data.error);return data.annotation;
}
function card(item){
 let a=item.annotation||{},chosen=!!a.chosen;
 let chosenIndex=Number.isInteger(a.chosen_example_index)?a.chosen_example_index:null;
 let images=item.images.slice(0,__MAX_EXAMPLES__);
 let div=document.createElement('div');div.className='card'+(chosen?' chosen':'');
 div.innerHTML='<div class="card-header"><strong>'+item.objective+' | seed '+item.seed+' | rank '+item.rank+' | feature '+item.feature+'</strong>'
  +'<span class="meta">active '+(100*item.dataset_active_fraction).toFixed(2)+'% of dataset</span>'
  +'<label class="choose"><input type="checkbox" '+(chosen?'checked':'')+'> Chosen for paper</label></div>'
  +'<div class="comment">'+(item.annotation&&item.annotation.comment?item.annotation.comment.replace(/</g,'&lt;'):'(no comment)')+'</div>';
 let rows=document.createElement('div');
 images.forEach((x,i)=>{
  let row=document.createElement('div');row.className='example-row'+(i===chosenIndex?' selected-example':'');
  row.innerHTML='<div class="example-row-header"><span>example '+i+'</span><label class="pick"><input type="radio" name="example-'+item.annotation_key+'" '+(i===chosenIndex?'checked':'')+'> Use this example</label></div>'
   +'<div class="grid">'+x.variants.map(v=>'<figure><img src="'+v.image+'"><figcaption>'+v.label+'</figcaption></figure>').join('')+'<figure><img src="'+x.heatmap+'"><figcaption>effect</figcaption></figure></div>';
  row.querySelector('input[type=radio]').addEventListener('change',async()=>{
   try{item.annotation=await saveExampleChoice(item,i);
    rows.querySelectorAll('.example-row').forEach((r,ri)=>r.classList.toggle('selected-example',ri===i));
   }catch(e){alert('save failed: '+e.message)}
  });
  rows.appendChild(row);
 });
 div.appendChild(rows);
 div.querySelector('input[type=checkbox]').addEventListener('change',async event=>{
  try{await saveChoice(item,event.target.checked);div.classList.toggle('chosen',event.target.checked)}
  catch(e){event.target.checked=!event.target.checked;alert('save failed: '+e.message)}
 });
 return div;
}
async function load(){
 let chosenOnly=$('chosenOnly').checked;
 $('status').textContent='loading shortlist'+(chosenOnly?' (chosen for paper only)':' (all interesting)')+'...';
 try{
  let response=await fetch('/api/shortlist?chosen_only='+(chosenOnly?'1':'0')),items=await response.json();
  if(items.error)throw Error(items.error);
  $('status').textContent=items.length+' feature(s)'+(chosenOnly?' chosen for paper':' marked interesting');
  let sections=$('sections');sections.innerHTML='';
  let byObjective=new Map();
  for(let item of items){if(!byObjective.has(item.objective))byObjective.set(item.objective,[]);byObjective.get(item.objective).push(item)}
  for(let [objective,group] of byObjective){
   let h2=document.createElement('h2');h2.textContent=objective+' ('+group.length+')';sections.appendChild(h2);
   for(let item of group)sections.appendChild(card(item));
  }
 }catch(e){$('status').textContent='error: '+e.message}
}
$('chosenOnly').addEventListener('change',load);
load();
</script></body></html>"""


def png_data(image: torch.Tensor) -> str:
    image = image.detach().float().cpu().clamp(0, 1)
    buf = io.BytesIO()
    Image.fromarray((image.permute(1, 2, 0).numpy() * 255).round().astype("uint8")).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class AnnotationStore:
    """Interesting/comment notes per (objective, seed, feature), persisted
    to disk so they survive a GUI restart. Ported from
    spatial_sae_feature_gui.py, extended with seed since feature IDs are
    not comparable across seeds (each seed is an independently-trained
    SAE) -- a note on "feature 5228" must not bleed across seeds."""

    def __init__(self, path: str | None):
        self.path = Path(path) if path else None
        self.lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = json.loads(self.path.read_text()) if self.path and self.path.is_file() else {}
        self._migrate_pre_seed_keys()

    def _migrate_pre_seed_keys(self) -> None:
        # Pre-seed-selector records are keyed "objective:feature" (one
        # colon) -- all made while the GUI could only load seed 2030, so
        # they migrate to "objective:2030:feature" rather than becoming
        # invisible once every key is expected to carry a seed.
        migrated = {}
        changed = False
        for key, record in self.records.items():
            if key.count(":") == 1 and "seed" not in record:
                objective, feature = key.split(":")
                new_key = f"{objective}:2030:{feature}"
                record = dict(record)
                record["seed"] = 2030
                migrated[new_key] = record
                changed = True
            else:
                migrated[key] = record
        self.records = migrated
        if changed and self.path:
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(json.dumps(self.records, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, self.path)

    def get(self, key: str) -> dict:
        with self.lock:
            return dict(self.records.get(key, {}))

    def update(self, payload: dict) -> dict:
        # "chosen", "chosen_example_index", and "comment" are each only sent
        # explicitly by the control that changes them -- the shortlist
        # gallery's chosen-checkbox and example-radio saves never send a
        # comment, and the main page's comment/interesting save never sends
        # chosen or chosen_example_index. Every one of these fields must
        # default to whatever was already there, not reset, or any one
        # control would silently discard what another control set.
        existing = self.records.get(payload["key"], {})
        chosen_example_index = (
            payload["chosen_example_index"] if "chosen_example_index" in payload
            else existing.get("chosen_example_index")
        )
        record = {
            "objective": str(payload["objective"]), "seed": int(payload["seed"]), "rank": int(payload["rank"]),
            "feature": int(payload["feature"]), "interesting": bool(payload.get("interesting", False)),
            "comment": str(payload["comment"]) if "comment" in payload else str(existing.get("comment", "")),
            "chosen": bool(payload["chosen"]) if "chosen" in payload else bool(existing.get("chosen", False)),
            "chosen_example_index": int(chosen_example_index) if chosen_example_index is not None else None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with self.lock:
            self.records[payload["key"]] = record
            if self.path:
                temporary = self.path.with_suffix(self.path.suffix + ".tmp")
                temporary.write_text(json.dumps(self.records, indent=2, sort_keys=True) + "\n")
                os.replace(temporary, self.path)
        return dict(record)

    def export(self) -> str:
        with self.lock:
            return json.dumps(self.records, indent=2, sort_keys=True) + "\n"

    def list_interesting(self) -> list[dict]:
        with self.lock:
            records = [dict(r, key=k) for k, r in self.records.items() if r.get("interesting")]
        records.sort(key=lambda r: (r["objective"], r["seed"], r["rank"]))
        return records


class ObjectiveSAE:
    """One loaded (lambda, objective, seed) SAE checkpoint plus its
    dataset-wide feature usage stats, computed once at startup."""

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
        self.feature_order = torch.argsort(self.dataset_mean_activation, descending=True).tolist()


class FeatureServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.autoencoder_type == "vector":
            self.autoencoder = load_vector_autoencoder(args.autoencoder, self.device)
        else:
            self.autoencoder = load_spatial_autoencoder(args.autoencoder, self.device)
        self.autoencoder.eval()

        cache = torch.load(args.latent_cache, map_location="cpu", weights_only=False)
        self.latents = cache["latents"].float()
        self.candidate_indices = cache["valid_indices"].long()

        self.annotations = AnnotationStore(args.annotations)
        self.default_seed = args.seed
        self.available_seeds = list(args.seeds)
        # Eagerly load every objective at the default seed (fast first
        # page load, matches the old single-seed behavior); other seeds
        # for a given objective are loaded lazily on first request from
        # the seed dropdown, in get_objective().
        self.objectives: dict[tuple[str, int], ObjectiveSAE] = {}
        self.loaded_objective_names: set[str] = set()
        for objective in OBJECTIVE_DIRS:
            checkpoint_path = Path(args.checkpoint_root) / objective / f"seed_{args.seed}" / "last.pt"
            if not checkpoint_path.is_file():
                print(f"skipping {objective}: no checkpoint at {checkpoint_path}", flush=True)
                continue
            print(f"loading {objective} seed {args.seed} ...", flush=True)
            self.objectives[(objective, args.seed)] = ObjectiveSAE(
                checkpoint_path, self.latents, self.candidate_indices, self.device, args.batch_size,
            )
            self.loaded_objective_names.add(objective)
        if not self.objectives:
            raise RuntimeError(f"no objective checkpoints found under {args.checkpoint_root}")
        print("all objective models and rankings loaded", flush=True)

        self.cache: dict[tuple[str, int, int], dict] = {}
        self.lock = threading.Lock()

    def get_objective(self, objective: str, seed: int) -> ObjectiveSAE:
        if objective not in self.loaded_objective_names:
            raise ValueError(f"objective must be one of {sorted(self.loaded_objective_names)}")
        key = (objective, seed)
        if key not in self.objectives:
            checkpoint_path = Path(self.args.checkpoint_root) / objective / f"seed_{seed}" / "last.pt"
            if not checkpoint_path.is_file():
                raise ValueError(f"no checkpoint for {objective} seed {seed} at {checkpoint_path}")
            print(f"loading {objective} seed {seed} (lazy) ...", flush=True)
            self.objectives[key] = ObjectiveSAE(
                checkpoint_path, self.latents, self.candidate_indices, self.device, self.args.batch_size,
            )
        return self.objectives[key]

    @torch.no_grad()
    def feature(self, objective: str, seed: int, rank: int) -> dict:
        seed = int(seed)
        with self.lock:
            obj = self.get_objective(objective, seed)
            rank = int(rank)
            if rank < 0 or rank >= obj.dict_size:
                raise ValueError(f"rank must be in [0, {obj.dict_size - 1}]")
            feature = obj.feature_order[rank]
            key = (objective, seed, feature)
            if key in self.cache:
                # The expensive part (images, dataset stats) is safe to
                # cache forever -- it never changes. The annotation is not:
                # it can be updated at any time via /api/annotation, so it
                # must be looked up live on every call, even for a cache
                # hit, or a stale "not interesting" snapshot from the first
                # time this feature was viewed would keep coming back after
                # the client's own cache evicts the entry and re-fetches.
                cached = self.cache[key]
                cached["annotation"] = self.annotations.get(cached["annotation_key"])
                return cached

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
            annotation_key = f"{objective}:{seed}:{feature}"
            result = {
                "objective": objective, "seed": seed, "rank": rank, "feature": feature, "examples": len(selected),
                "dataset_mean_activation": float(obj.dataset_mean_activation[feature]),
                "dataset_active_fraction": float(obj.dataset_active_fraction[feature]),
                "dataset_std_activation": float(sigma),
                "selected_active": int(naturally_active.sum()),
                "annotation_key": annotation_key,
                "annotation": self.annotations.get(annotation_key),
                "images": [{"variants": [{"label": f"{mult:+d} std", "image": png_data(decoded[i])}
                                          for mult, decoded in variants],
                            "heatmap": png_data(effect[i])}
                           for i in range(len(selected))],
            }
            self.cache[key] = result
            return result

    def shortlist(self, max_examples: int, chosen_only: bool = False) -> list[dict]:
        items = []
        for record in self.annotations.list_interesting():
            if chosen_only and not record.get("chosen"):
                continue
            try:
                result = dict(self.feature(record["objective"], record["seed"], record["rank"]))
            except Exception as exc:  # noqa: BLE001 -- a stale/renamed checkpoint shouldn't break the whole gallery
                print(f"shortlist: skipping {record.get('key')}: {exc}", flush=True)
                continue
            result["images"] = result["images"][:max_examples]
            items.append(result)
        return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True,
                         help="e.g. .../experiments/vector_16x_k32/checkpoints/lambda_100p0")
    parser.add_argument("--autoencoder", required=True)
    parser.add_argument("--autoencoder-type", choices=("vector", "spatial"), default="vector")
    parser.add_argument("--latent-cache", required=True)
    parser.add_argument("--annotations", help="JSON file to persist interesting/comment notes across restarts")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2030, help="seed loaded eagerly at startup for every objective")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                         help="seeds offered in the dropdown; non-default ones load lazily on first use")
    parser.add_argument("--label", default="matched512")
    parser.add_argument("--shortlist-examples", type=int, default=10,
                         help="example rows shown per feature on the /shortlist gallery (capped, unlike the main page)")
    args = parser.parse_args()
    if args.seed not in args.seeds:
        args.seeds = [args.seed] + list(args.seeds)
    server = FeatureServer(args)

    options_html = "".join(
        f'<option value="{obj}">{html.escape(OBJECTIVE_LABELS.get(obj, obj))}</option>'
        for obj in OBJECTIVE_DIRS if obj in server.loaded_objective_names
    )
    seed_options_html = "".join(
        f'<option value="{seed}"{" selected" if seed == args.seed else ""}>{seed}</option>'
        for seed in server.available_seeds
    )
    page = (
        HTML_PAGE.replace("__LABEL__", html.escape(args.label))
        .replace("__OBJECTIVE_OPTIONS__", options_html)
        .replace("__SEED_OPTIONS__", seed_options_html)
    )
    shortlist_page = (
        SHORTLIST_PAGE.replace("__LABEL__", html.escape(args.label))
        .replace("__MAX_EXAMPLES__", str(args.shortlist_examples))
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
            elif parsed.path == "/shortlist":
                payload = shortlist_page.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(payload)
            elif parsed.path == "/api/feature":
                try:
                    query = parse_qs(parsed.query)
                    objective = query.get("objective", [next(iter(server.loaded_objective_names))])[0]
                    seed = int(query.get("seed", [str(args.seed)])[0])
                    rank = int(query.get("rank", ["0"])[0])
                    payload = json.dumps(server.feature(objective, seed, rank)).encode()
                    self.send_response(200)
                except Exception as exc:  # noqa: BLE001
                    payload = json.dumps({"error": str(exc)}).encode()
                    self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)
            elif parsed.path == "/api/annotations":
                payload = server.annotations.export().encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Disposition", "attachment; filename=matched512_feature_annotations.json")
                self.send_header("Content-Length", str(len(payload)))
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
            if parsed.path != "/api/annotation":
                self.send_response(404)
                self.end_headers()
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                record = server.annotations.update(json.loads(self.rfile.read(length)))
                payload = json.dumps({"annotation": record}).encode()
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

    print(f"GUI ready at http://0.0.0.0:{args.port} -- objectives: {sorted(server.loaded_objective_names)}, "
          f"default seed: {args.seed}, seeds offered: {server.available_seeds}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
