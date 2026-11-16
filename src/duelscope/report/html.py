"""Self-contained HTML report: inline CSS and JS, base64 images, no external requests."""

from __future__ import annotations

import base64
import json
from html import escape
from pathlib import Path

CSS = """
:root{--bg:#f6f8fa;--card:#fff;--ink:#24292f;--mute:#57606a;--line:#d0d7de;--acc:#0072B2;
--good:#1a7f37;--warn:#9a6700}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--card:#161b22;--ink:#e6edf3;
--mute:#8b949e;--line:#30363d;--acc:#58a6ff;--good:#3fb950;--warn:#d29922}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1120px;margin:0 auto;padding:24px 16px 64px}
h1{margin:0 0 4px;font-size:26px}h2{font-size:19px;margin:0 0 12px}
.sub{color:var(--mute);margin:0 0 16px}
nav{display:flex;flex-wrap:wrap;gap:6px 16px;margin:0 0 20px;font-size:14px}
nav a{color:var(--acc);text-decoration:none}
section.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:18px;margin:0 0 18px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{cursor:pointer;user-select:none;color:var(--mute);font-weight:600}
th.sorted-asc::after{content:" \\25B2"}th.sorted-desc::after{content:" \\25BC"}
td.best{font-weight:700;color:var(--good)}
.chart{background:#fff;border-radius:8px;padding:8px;text-align:center}
.chart img{max-width:100%;height:auto}
ul.insights{margin:0;padding-left:20px}ul.insights li{margin:4px 0}
.warn{color:var(--warn)}
details{border:1px solid var(--line);border-radius:8px;margin:8px 0;padding:6px 12px}
details>summary{cursor:pointer;font-weight:600;padding:4px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px;margin:10px 0}
figure{margin:0}figure img{width:100%;height:auto;border-radius:6px;border:1px solid var(--line)}
figcaption{font-size:12px;color:var(--mute)}
.legend span{display:inline-block;margin-right:14px;font-size:13px}
.sw{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:4px;
vertical-align:-1px}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:4px 18px;font-size:14px}
.kv dt{color:var(--mute)}.kv dd{margin:0}
@media (max-width:600px){main{padding:16px 12px}}
"""

JS = """
document.querySelectorAll('table.sortable').forEach(function(t){
 t.querySelectorAll('th').forEach(function(th,i){
  th.addEventListener('click',function(){
   var asc=!th.classList.contains('sorted-asc');
   t.querySelectorAll('th').forEach(function(x){x.classList.remove('sorted-asc','sorted-desc')});
   th.classList.add(asc?'sorted-asc':'sorted-desc');
   var rows=Array.from(t.tBodies[0].rows);
   rows.sort(function(a,b){
    var x=a.cells[i].dataset.v,y=b.cells[i].dataset.v,nx=parseFloat(x),ny=parseFloat(y);
    var c=(isNaN(nx)||isNaN(ny))?String(x).localeCompare(String(y)):nx-ny;
    return asc?c:-c;});
   rows.forEach(function(r){t.tBodies[0].appendChild(r)});
  });});});
"""

LOAD_IMGS = "document.querySelectorAll('img[data-k]').forEach(function(i){i.src=IMG[i.dataset.k]});"

COLUMNS = [
    ("mAP50-95", lambda m: m["accuracy"]["map5095"] * 100, "{:.1f}", True),
    ("mAP50", lambda m: m["accuracy"]["map50"] * 100, "{:.1f}", True),
    ("mAP S", lambda m: _pct(m["accuracy"]["map_small"]), "{:.1f}", True),
    ("mAP M", lambda m: _pct(m["accuracy"]["map_medium"]), "{:.1f}", True),
    ("mAP L", lambda m: _pct(m["accuracy"]["map_large"]), "{:.1f}", True),
    ("Precision", lambda m: m["accuracy"]["precision"] * 100, "{:.1f}", True),
    ("Recall", lambda m: m["accuracy"]["recall"] * 100, "{:.1f}", True),
    ("F1", lambda m: m["accuracy"]["f1"] * 100, "{:.1f}", True),
    ("p50 ms", lambda m: _sp(m, "latency_ms", "p50"), "{:.1f}", False),
    ("p99 ms", lambda m: _sp(m, "latency_ms", "p99"), "{:.1f}", False),
    ("FPS (bs 1)", lambda m: _sp(m, "fps_bs1"), "{:.1f}", True),
    ("Params (M)", lambda m: _opt(m["metadata"].get("params"), 1e-6), "{:.2f}", False),
    ("Size (MB)", lambda m: m["metadata"].get("weights_mb"), "{:.1f}", False),
    ("GPU peak MB", lambda m: _sp(m, "gpu_peak_mb"), "{:.0f}", False),
]


def _pct(v: float | None) -> float | None:
    return None if v is None else v * 100


def _opt(v: float | None, scale: float) -> float | None:
    return None if v is None else v * scale


def _sp(m: dict, *path: str) -> float | None:
    cur = m.get("speed")
    for p in path:
        if cur is None:
            return None
        cur = cur.get(p)
    return cur


def b64_image(path: Path, mime: str = "image/jpeg") -> str | None:
    """Return a data URI for an image file, or None if it does not exist."""
    if not path.is_file():
        return None
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def leaderboard(models: list[dict]) -> str:
    """Sortable leaderboard table with the best value of each column highlighted."""
    vals = [[get(m) for _, get, _, _ in COLUMNS] for m in models]
    best = []
    for c, (_, _, _, hi) in enumerate(COLUMNS):
        col = [v[c] for v in vals if v[c] is not None]
        best.append((max(col) if hi else min(col)) if col and len(models) > 1 else None)
    head = "<tr><th>Model</th>" + "".join(f"<th>{escape(h)}</th>" for h, *_ in COLUMNS) + "</tr>"
    rows = []
    for m, v in zip(models, vals, strict=True):
        cells = [f'<td data-v="{escape(m["name"])}">{escape(m["name"])}</td>']
        for c, (_, _, fmt, _) in enumerate(COLUMNS):
            x = v[c]
            if x is None:
                cells.append('<td data-v="">-</td>')
            else:
                cls = ' class="best"' if best[c] is not None and x == best[c] else ""
                cells.append(f'<td{cls} data-v="{x:.6f}">{fmt.format(x)}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="sortable"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'


def _chart(png: bytes | None, alt: str) -> str:
    if png is None:
        return '<p class="sub">Not available for this run.</p>'
    uri = "data:image/png;base64," + base64.b64encode(png).decode()
    return f'<div class="chart"><img src="{uri}" alt="{escape(alt)}"></div>'


def _figures(items: list[dict], root: Path, imgs: dict[str, str]) -> str:
    if not items:
        return '<p class="sub">Nothing to show: no image met the criterion.</p>'
    figs = []
    for it in items:
        uri = imgs.get(it["thumb"]) or b64_image(root / it["thumb"])
        if uri:
            imgs[it["thumb"]] = uri  # embedded once, referenced by key (see IMG script)
            img = f'<img data-k="{escape(it["thumb"])}" alt="{escape(it["image"])}">'
        else:
            img = "<em>thumbnail missing</em>"
        cap = (
            f"{escape(it['model'])} | {escape(it['image'])}<br>score {it['score']} | "
            f"TP {it['tp']} FP {it['fp']} missed {it['missed']}"
        )
        figs.append(f"<figure>{img}<figcaption>{cap}</figcaption></figure>")
    return f'<div class="grid">{"".join(figs)}</div>'


def _gallery(gallery: dict | None, root: Path, imgs: dict[str, str]) -> str:
    if not gallery:
        return '<p class="sub">Gallery was disabled for this run.</p>'
    legend = (
        '<p class="legend"><span><i class="sw" style="background:#2ea043"></i>ground truth</span>'
        '<span><i class="sw" style="background:#2979ff"></i>true positive</span>'
        '<span><i class="sw" style="background:#e63946"></i>false positive</span>'
        '<span><i class="sw" style="background:#ffc400"></i>missed (dashed)</span></p>'
        f'<p class="sub">Operating threshold: confidence {gallery["conf"]}. '
        "Image score = missed objects + sum of false-positive confidences.</p>"
    )
    parts = [legend]
    for m, items in gallery["per_model"].items():
        parts.append(
            f"<details><summary>Worst images: {escape(m)} ({len(items)})</summary>"
            f"{_figures(items, root, imgs)}</details>"
        )
    for title, items in gallery["a_ok_b_fails"].items():
        parts.append(
            f"<details><summary>{escape(title)} ({len(items)})</summary>"
            f"{_figures(items, root, imgs)}</details>"
        )
    parts.append(
        f"<details><summary>Hard for every model ({len(gallery['all_fail'])}) - "
        "possible label errors</summary>"
        f"{_figures(gallery['all_fail'], root, imgs)}</details>"
    )
    empty = '<p class="sub">No class errors.</p>'
    for m, per in gallery["per_class"].items():
        inner = "".join(
            f"<details><summary>{escape(c)} ({len(items)})</summary>"
            f"{_figures(items, root, imgs)}</details>"
            for c, items in per.items()
        )
        parts.append(
            f"<details><summary>Per-class failures: {escape(m)}</summary>{inner or empty}</details>"
        )
    return "".join(parts)


def _errors_table(models: list[dict]) -> str:
    from duelscope.report.insights import PHRASES

    keys = list(PHRASES)
    head = "".join(f"<th>{escape(PHRASES[k])}</th>" for k in keys)
    rows = []
    for m in models:
        e = m["errors"]
        cells = "".join(
            f"<td>{e['counts'][k]} ({e['share'][k] * 100:.0f}%) / "
            f"+{e.get('delta_map50', {}).get(k, 0) * 100:.1f}</td>"
            for k in keys
        )
        rows.append(f"<tr><td>{escape(m['name'])}</td>{cells}</tr>")
    return (
        f"<table><thead><tr><th>Model</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        '<p class="sub">Each cell: count (share of errors) / mAP50 points gained if fixed. '
        "TP count is in results.json.</p>"
    )


def _hardware(hw: dict) -> str:
    gpu = hw.get("gpu")
    items = {
        "CPU": f"{hw['cpu']} ({hw['cores_physical']} cores / {hw['cores_logical']} threads)",
        "RAM": f"{hw['ram_gb']} GB",
        "GPU": (
            f"{gpu['name']} ({gpu.get('vram_mb', '?')} MB"
            + (f", driver {gpu['driver']}" if gpu.get("driver") else "")
            + (f", CUDA {gpu['cuda_driver_version']}" if gpu.get("cuda_driver_version") else "")
            + ")"
            if gpu
            else "none detected"
        ),
        "PyTorch CUDA usable": "yes" if hw.get("torch_cuda_available") else "no",
        "OS": hw["os"],
        "Python": hw["python"],
        "Libraries": ", ".join(f"{k} {v}" for k, v in hw["libraries"].items()),
    }
    return (
        '<dl class="kv">'
        + "".join(f"<dt>{escape(k)}</dt><dd>{escape(str(v))}</dd>" for k, v in items.items())
        + "</dl>"
    )


def render_html(results: dict, charts: dict[str, bytes | None], gallery_root: Path) -> str:
    """Build the complete HTML document."""
    models = sorted(results["models"], key=lambda m: -m["accuracy"]["map5095"])
    imgs: dict[str, str] = {}
    ds, st = results["dataset"], results["settings"]
    insights = "".join(f"<li>{escape(s)}</li>" for s in results.get("insights", []))
    warns = [w for m in models for w in m.get("warnings", [])]
    warn_html = (
        '<ul class="warn">' + "".join(f"<li>{escape(w)}</li>" for w in warns) + "</ul>"
        if warns
        else ""
    )
    settings = ", ".join(f"{k}={v}" for k, v in st.items())
    sections = [
        ("insights", "Key findings", f'<ul class="insights">{insights}</ul>{warn_html}'),
        ("leaderboard", "Leaderboard (click a header to sort)", leaderboard(models)),
        ("speed", "Accuracy versus speed", _chart(charts.get("speed"), "accuracy vs speed")),
        ("classes", "Per-class AP", _chart(charts.get("classes"), "per-class AP heatmap")),
        ("pr", "Precision-recall curves", _chart(charts.get("pr"), "precision-recall curves")),
        (
            "errors",
            "Error taxonomy",
            _chart(charts.get("errors"), "error taxonomy") + _errors_table(models),
        ),
        ("gallery", "Failure-case gallery", _gallery(results.get("gallery"), gallery_root, imgs)),
        (
            "hardware",
            "Hardware and settings",
            _hardware(results["hardware"]) + f'<p class="sub">Settings: {escape(settings)}. '
            f"Dataset hash {escape(ds['hash'])}.</p>",
        ),
    ]
    nav = "".join(f'<a href="#{i}">{escape(t.split(" (")[0])}</a>' for i, t, _ in sections)
    body = "".join(
        f'<section class="card" id="{i}"><h2>{escape(t)}</h2>{c}</section>' for i, t, c in sections
    )
    title = f"Duelscope report: {ds['name']}"
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escape(title)}</title><style>{CSS}</style></head><body><main>"
        f"<h1>{escape(title)}</h1>"
        f'<p class="sub">{ds["images"]} images, {ds["boxes"]} boxes, {len(ds["classes"])} classes '
        f"(split {escape(ds['split'])}) | {len(models)} models | Duelscope "
        f"{escape(results['duelscope_version'])} | {escape(results['created'])}</p>"
        f"<nav>{nav}</nav>{body}</main><script>{JS}</script>"
        f"<script>var IMG={json.dumps(imgs)};{LOAD_IMGS}</script></body></html>"
    )
