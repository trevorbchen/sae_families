#!/usr/bin/env python3
"""Focused local frontend for reviewing the 3,000 frequency-family features."""

import argparse
import csv
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urljoin, urlparse


SOURCES = {
    "direct_mse": "Direct MSE",
    "grouped_objective_laplacian": "Single objective: Laplacian",
    "grouped_objective_fourier_low": "Single objective: Fourier low",
    "grouped_objective_fourier_mid": "Single objective: Fourier mid",
    "grouped_objective_fourier_high": "Single objective: Fourier high",
    "tandem_canonical": "Routed SAE",
}

GEOMETRY_OBJECTIVES = {
    "weighted_laplacian": ("grouped_objective_laplacian", "laplacian_norm"),
    "weighted_fourier_low": ("grouped_objective_fourier_low", "fourier_low_fraction"),
    "weighted_fourier_mid": ("grouped_objective_fourier_mid", "fourier_mid_fraction"),
    "weighted_fourier_high": ("grouped_objective_fourier_high", "fourier_high_fraction"),
}

GEOMETRY_METRICS = (
    "laplacian_norm",
    "fourier_low_fraction",
    "fourier_mid_fraction",
    "fourier_high_fraction",
)

SECTION_ORDER = {
    "direct_mse": 0,
    "grouped_objective_laplacian": 1,
    "grouped_objective_fourier_low": 2,
    "grouped_objective_fourier_mid": 3,
    "grouped_objective_fourier_high": 4,
    "tandem_canonical:Laplacian": 5,
    "tandem_canonical:Fourier low": 6,
    "tandem_canonical:Fourier mid": 7,
    "tandem_canonical:Fourier high": 8,
}


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>3,000-feature review</title>
<style>
:root{color-scheme:dark;--bg:#101112;--panel:#1b1d1f;--line:#3c4145;--muted:#a8b0b5;--text:#f1f3f4;--accent:#e1a52b;--action:#168a84}
*{box-sizing:border-box}body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--text);margin:0;padding:18px 20px 32px}
body.paper-candidate:before{content:"";position:fixed;z-index:20;left:0;top:0;bottom:0;width:6px;background:var(--accent)}
header{position:sticky;z-index:10;top:0;display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:10px 0 12px;background:var(--bg);border-bottom:1px solid var(--line)}
strong{margin-right:6px}button,input,select,textarea,a.command{font:inherit;background:#202326;color:var(--text);border:1px solid #555d62;padding:7px 10px;border-radius:4px}
button,a.command{cursor:pointer;text-decoration:none}button:hover,a.command:hover{background:#2a2e31}button:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible{outline:2px solid #58b9b2;outline-offset:2px}
input[type=number]{width:88px}.nav{width:38px;padding-left:0;padding-right:0;font-size:18px}.primary{background:#126e69;border-color:#299d96}.primary:hover{background:#16827c}
#interesting[aria-pressed=true]{background:var(--accent);border-color:#f4c35b;color:#17130a;font-weight:700}#candidateBadge{display:none;color:#15120b;background:#f0bd4d;padding:5px 9px;border-radius:4px;font-weight:700}body.paper-candidate #candidateBadge{display:inline-block}
#routeBadge{color:#d8eeee;background:#174c49;border:1px solid #2c8b85;padding:5px 9px;border-radius:4px;font-weight:700}
#status{color:var(--muted)}#summary{margin:12px 0;color:#c9ced1;min-height:22px}.annotation{display:grid;grid-template-columns:auto minmax(260px,1fr) auto;gap:10px;align-items:stretch;margin:12px 0 16px;max-width:1800px}
.geometry{display:grid;grid-template-columns:repeat(3,minmax(240px,1fr));gap:10px;max-width:1100px;margin:10px 0 14px}.geometry-card{border:1px solid var(--line);background:var(--panel);padding:10px 12px;border-radius:5px}.geometry-card h3{font-size:14px;margin:0 0 7px}.metric-row{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}.metric{font-size:12px;color:var(--muted)}.metric b{display:block;color:var(--text);font-size:14px;margin-top:2px}.alignment-lines{font-size:13px;line-height:1.5}.unavailable{color:var(--muted);font-size:13px}
textarea{width:100%;min-height:48px;resize:vertical;line-height:1.35}.grid{display:grid;grid-template-columns:repeat(8,minmax(118px,1fr));gap:9px;max-width:1800px}
figure{margin:0;background:var(--panel);padding:7px;border:1px solid #24282a;border-radius:5px;overflow:hidden}img{width:100%;display:block;aspect-ratio:1;object-fit:cover}figcaption{font-size:12px;color:#bec4c8;margin-top:6px;min-height:18px}
@media(max-width:900px){.grid{grid-template-columns:repeat(4,minmax(110px,1fr))}.annotation,.geometry{grid-template-columns:1fr}.annotation button,.annotation a{width:100%}}
@media(max-width:560px){body{padding:10px}.grid{grid-template-columns:repeat(2,minmax(120px,1fr))}}
</style></head><body>
<header><strong>New feature review</strong>
<span>section</span><select id="section"></select><span>organize</span><select id="organize"><option value="activity">Activity rank</option><option value="edited_alignment">Edited-output alignment</option><option value="causal_alignment">Causal-delta alignment</option><option value="edited_laplacian_norm">Edited Laplacian</option><option value="edited_fourier_low_fraction">Edited Fourier low</option><option value="edited_fourier_mid_fraction">Edited Fourier mid</option><option value="edited_fourier_high_fraction">Edited Fourier high</option><option value="causal_laplacian_norm">Delta Laplacian</option><option value="causal_fourier_low_fraction">Delta Fourier low</option><option value="causal_fourier_mid_fraction">Delta Fourier mid</option><option value="causal_fourier_high_fraction">Delta Fourier high</option></select>
<span>item</span><button id="prev" class="nav" title="Previous feature">&#8592;</button><input id="index" type="number" min="1" value="1">
<button id="next" class="nav" title="Next feature">&#8594;</button><button id="load" class="primary">Load</button>
<span id="routeBadge"></span><span id="candidateBadge">Interesting</span><span id="status">ready</span></header>
<div id="summary"></div>
<div id="geometry" class="geometry"></div>
<div class="annotation"><button id="interesting" aria-pressed="false" title="Toggle interesting (I)">Interesting</button><textarea id="comment" placeholder="Feature notes (C to focus)"></textarea><div><button id="save" title="Save note">Save note</button> <a class="command" href="/api/annotations" download="sae_feature_annotations.json">Export</a></div></div>
<div id="grid" class="grid"></div>
<script>
const $=id=>document.getElementById(id),allQueue=__MANIFEST__,cache=new Map(),inflight=new Map();
let queue=[];
let current=null,currentIndex=0,dirty=false,saveTimer=null,saveInFlight=null,requestToken=0;
function key(source,rank){return source+':'+rank}
function annotationKey(d){return d.annotation_key||d.source+':'+d.feature}
function setCandidate(value){$('interesting').setAttribute('aria-pressed',value?'true':'false');document.body.classList.toggle('paper-candidate',value)}
function metricValue(value,kind){if(value===null||value===undefined)return 'n/a';return kind==='laplacian_norm'?Number(value).toFixed(4):(100*Number(value)).toFixed(1)+'%'}
function percentile(value){return value===null||value===undefined?'n/a':Number(value).toFixed(1)+'th pct'}
function renderGeometry(item){let g=item.geometry;if(!g){$('geometry').innerHTML='<div class="geometry-card unavailable">Feature-level geometry has not yet been measured for this model.</div>';return}
 const card=(title,m)=>'<div class="geometry-card"><h3>'+title+'</h3><div class="metric-row"><span class="metric">Laplacian<b>'+metricValue(m.laplacian_norm,'laplacian_norm')+'</b></span><span class="metric">Fourier low<b>'+metricValue(m.fourier_low_fraction,'fourier')+'</b></span><span class="metric">Fourier mid<b>'+metricValue(m.fourier_mid_fraction,'fourier')+'</b></span><span class="metric">Fourier high<b>'+metricValue(m.fourier_high_fraction,'fourier')+'</b></span></div></div>';
 let a=g.alignment;$('geometry').innerHTML=card('Feature output geometry',g.edited_output)+card('Causal-delta geometry',g.causal_delta)+'<div class="geometry-card"><h3>Objective alignment</h3><div class="alignment-lines">Target: <b>'+a.metric_label+'</b><br>Edited output: <b>'+percentile(a.edited_percentile)+'</b> &middot; source rank '+a.edited_rank+'/'+a.source_count+'<br>Causal delta: <b>'+percentile(a.causal_percentile)+'</b> &middot; source rank '+a.causal_rank+'/'+a.source_count+'</div></div>'}
function displayRoute(d,item){if(d.source==='tandem_canonical'&&d.component_label)return 'Independent route: '+d.component_label.replace(' branch','');return item.label}
function render(d){current=d;dirty=false;let a=d.annotation||{};setCandidate(!!a.interesting);$('comment').value=a.comment||'';
 let item=queue[currentIndex],route=displayRoute(d,item);$('routeBadge').textContent=route;
 $('summary').textContent=(currentIndex+1)+' / '+queue.length+' | source position '+(item.source_position+1)+' | activity rank '+(d.rank+1)+' | feature '+d.feature+' | active RMS '+d.active_rms.toFixed(5)+' | active '+(100*d.dataset_active_fraction).toFixed(2)+'% | selected active '+d.selected_active+' / '+d.examples;
 renderGeometry(item);
 $('grid').innerHTML=d.images.map((x,i)=>x.variants.map(v=>'<figure><img src="'+v.image+'"><figcaption>example '+i+' '+v.label+'</figcaption></figure>').join('')+'<figure><img src="'+x.heatmap+'"><figcaption>example '+i+' output effect</figcaption></figure>').join('')}
async function fetchResult(source,rank){let k=key(source,rank);if(cache.has(k))return cache.get(k);if(inflight.has(k))return inflight.get(k);
 let promise=(async()=>{let response=await fetch('/api/feature?source='+encodeURIComponent(source)+'&rank='+rank),data=await response.json();if(data.error)throw Error(data.error);cache.set(k,data);return data})().finally(()=>inflight.delete(k));inflight.set(k,promise);return promise}
async function saveAnnotation(){clearTimeout(saveTimer);if(saveInFlight)await saveInFlight;if(!current||!dirty)return;
 const target=current,targetKey=key(target.source,target.rank),payload={key:annotationKey(target),source:target.source,source_label:target.source_label,rank:target.rank,feature:target.feature,interesting:$('interesting').getAttribute('aria-pressed')==='true',comment:$('comment').value};dirty=false;
 saveInFlight=(async()=>{let response=await fetch('/api/annotation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}),data=await response.json();if(data.error)throw Error(data.error);target.annotation=data.annotation;cache.set(targetKey,target);if(current===target)$('status').textContent='saved'})();
 try{await saveInFlight}catch(error){if(current===target)dirty=true;throw error}finally{saveInFlight=null}if(current===target&&dirty)return saveAnnotation()}
function queueSave(){dirty=true;clearTimeout(saveTimer);saveTimer=setTimeout(()=>saveAnnotation().catch(e=>$('status').textContent='save error: '+e.message),500)}
async function load(){try{await saveAnnotation();currentIndex=Math.max(0,Math.min(queue.length-1,parseInt($('index').value||1)-1));$('index').value=currentIndex+1;let item=queue[currentIndex],token=++requestToken;$('status').textContent='loading '+(currentIndex+1)+' / '+queue.length+'...';let data=await fetchResult(item.source,item.rank);if(token!==requestToken)return;render(data);$('status').textContent='loaded';void prefetch(currentIndex)}catch(e){$('status').textContent='error: '+e.message}}
async function prefetch(index){for(let next=index+1;next<=Math.min(queue.length-1,index+8);next++){let item=queue[next];try{await fetchResult(item.source,item.rank)}catch(_){return}}for(let [k,v] of cache){let keep=queue.slice(Math.max(0,index-4),Math.min(queue.length,index+11)).some(item=>item.source===v.source&&item.rank===v.rank);if(!keep)cache.delete(k)}}
async function moveIndex(delta){await saveAnnotation();currentIndex=Math.max(0,Math.min(queue.length-1,currentIndex+delta));$('index').value=currentIndex+1;load()}
async function toggleInteresting(){if(!current)return;setCandidate($('interesting').getAttribute('aria-pressed')!=='true');dirty=true;await saveAnnotation()}
function sortValue(item,mode){let g=item.geometry;if(mode==='activity')return -item.rank;if(!g)return -Infinity;if(mode==='edited_alignment')return g.alignment.edited_percentile;if(mode==='causal_alignment')return g.alignment.causal_percentile;let [view,...parts]=mode.split('_'),metric=parts.join('_');return g[view==='edited'?'edited_output':'causal_delta'][metric]}
function rebuildQueue(loadFirst=true){let section=$('section').value,mode=$('organize').value;queue=allQueue.filter(item=>section==='all'||item.section===section);queue.sort((a,b)=>{if(mode==='activity'){if(section==='all')return a.source_position-b.source_position||a.section_order-b.section_order;return a.source_position-b.source_position}let delta=sortValue(b,mode)-sortValue(a,mode);return Number.isFinite(delta)&&delta!==0?delta:a.source_position-b.source_position});currentIndex=0;$('index').value=1;$('index').max=queue.length;if(loadFirst&&queue.length)load()}
function initializeSections(){let entries=[...new Map(allQueue.map(item=>[item.section,item.section_label])).entries()].sort((a,b)=>{let ai=allQueue.find(x=>x.section===a[0]).section_order,bi=allQueue.find(x=>x.section===b[0]).section_order;return ai-bi});$('section').innerHTML='<option value="all">All sources, interleaved</option>'+entries.map(([value,label])=>'<option value="'+value+'">'+label+'</option>').join('');rebuildQueue(false)}
$('load').onclick=load;$('index').onchange=load;$('section').onchange=()=>rebuildQueue();$('organize').onchange=()=>rebuildQueue();$('prev').onclick=()=>moveIndex(-1);$('next').onclick=()=>moveIndex(1);$('interesting').onclick=()=>toggleInteresting().catch(e=>$('status').textContent='save error: '+e.message);$('save').onclick=()=>saveAnnotation().catch(e=>$('status').textContent='save error: '+e.message);$('comment').oninput=queueSave;
$('comment').addEventListener('keydown',event=>{if((event.ctrlKey||event.metaKey)&&event.key==='Enter'){event.preventDefault();saveAnnotation().then(()=>$('comment').blur())}else if(event.key==='Escape'){event.preventDefault();saveAnnotation().then(()=>$('comment').blur())}});
document.addEventListener('keydown',event=>{if(['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName))return;if(event.key==='ArrowLeft'){event.preventDefault();moveIndex(-1)}else if(event.key==='ArrowRight'){event.preventDefault();moveIndex(1)}else if(event.key.toLowerCase()==='i'){event.preventDefault();toggleInteresting()}else if(event.key.toLowerCase()==='c'){event.preventDefault();$('comment').focus();$('comment').setSelectionRange($('comment').value.length,$('comment').value.length)}});
window.addEventListener('beforeunload',()=>{if(current&&dirty){let payload={key:annotationKey(current),source:current.source,source_label:current.source_label,rank:current.rank,feature:current.feature,interesting:$('interesting').getAttribute('aria-pressed')==='true',comment:$('comment').value};navigator.sendBeacon('/api/annotation',new Blob([JSON.stringify(payload)],{type:'application/json'}))}});initializeSections();load();
</script></body></html>"""


def percentile(values, value):
    if not values:
        return None
    below = sum(candidate < value for candidate in values)
    equal = sum(candidate == value for candidate in values)
    return 100.0 * (below + 0.5 * equal) / len(values)


def load_geometry(path):
    if not path:
        return {}
    rows = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["seed"]) != 2030 or row["objective"] not in GEOMETRY_OBJECTIVES:
                continue
            rows.append(row)
    distributions = {
        (measurement, metric): [
            float(row[metric]) for row in rows if row["measurement"] == measurement
        ]
        for measurement in ("edited_output", "causal_delta")
        for metric in GEOMETRY_METRICS
    }
    paired = {}
    for row in rows:
        source, aligned_metric = GEOMETRY_OBJECTIVES[row["objective"]]
        record = paired.setdefault((source, int(row["feature"])), {})
        record[row["measurement"]] = {
            metric: float(row[metric]) for metric in GEOMETRY_METRICS
        }
        record["aligned_metric"] = aligned_metric
    source_values = {}
    for (source, _), record in paired.items():
        metric = record["aligned_metric"]
        for measurement in ("edited_output", "causal_delta"):
            source_values.setdefault((source, measurement), []).append(
                record[measurement][metric]
            )
    metric_labels = {
        "laplacian_norm": "Laplacian",
        "fourier_low_fraction": "Fourier low",
        "fourier_mid_fraction": "Fourier mid",
        "fourier_high_fraction": "Fourier high",
    }
    output = {}
    for key, record in paired.items():
        source, _ = key
        metric = record.pop("aligned_metric")
        edited_value = record["edited_output"][metric]
        causal_value = record["causal_delta"][metric]
        edited_source = source_values[(source, "edited_output")]
        causal_source = source_values[(source, "causal_delta")]
        record["alignment"] = {
            "metric": metric,
            "metric_label": metric_labels[metric],
            "edited_percentile": percentile(
                distributions[("edited_output", metric)], edited_value
            ),
            "causal_percentile": percentile(
                distributions[("causal_delta", metric)], causal_value
            ),
            "edited_rank": 1 + sum(value > edited_value for value in edited_source),
            "causal_rank": 1 + sum(value > causal_value for value in causal_source),
            "source_count": len(edited_source),
        }
        output[key] = record
    return output


def load_manifest(path, geometry):
    records = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("interesting", "false")).strip().lower() == "true":
                continue
            source = row["source"]
            component = row.get("component_label", "").removesuffix(" branch")
            if source == "tandem_canonical" and component:
                label = f"Independent route: {component}"
            elif source == "direct_mse":
                label = "Source: Direct MSE"
            else:
                label = "Objective: " + SOURCES[source].removeprefix("Single objective: ")
            records.append({
                "source": source,
                "rank": int(row["rank"]),
                "feature": int(row["feature"]),
                "label": label,
                "section": (
                    f"{source}:{component}" if source == "tandem_canonical" else source
                ),
                "section_label": label,
                "geometry": geometry.get((source, int(row["feature"]))),
            })
    by_section = {}
    for record in records:
        by_section.setdefault(record["section"], []).append(record)
    for section_records in by_section.values():
        section_records.sort(key=lambda record: record["rank"])
        for position, record in enumerate(section_records):
            record["source_position"] = position
            record["section_order"] = SECTION_ORDER[record["section"]]
    return records


def proxy_request(backend, method, path, headers, body=None):
    request = urllib.request.Request(
        urljoin(backend.rstrip("/") + "/", path.lstrip("/")),
        data=body,
        headers={
            key: value for key, value in headers.items()
            if key.lower() in {"content-type", "accept"}
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="http://127.0.0.1:18813")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8813)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--geometry-csv")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest_csv, load_geometry(args.geometry_csv))
    if not manifest:
        raise ValueError("review manifest is empty")
    page = HTML.replace("__MANIFEST__", json.dumps(manifest)).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if urlparse(self.path).path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return
            self.proxy("GET")

        def do_POST(self):
            self.proxy("POST")

        def proxy(self, method):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else None
            try:
                status, headers, payload = proxy_request(
                    args.backend, method, self.path, self.headers, body
                )
                self.send_response(status)
                self.send_header("Content-Type", headers.get("Content-Type", "application/json"))
                disposition = headers.get("Content-Disposition")
                if disposition:
                    self.send_header("Content-Disposition", disposition)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception as error:
                payload = json.dumps({"error": str(error)}).encode()
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        def log_message(self, *_):
            return

    print(f"Review GUI ready at http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
