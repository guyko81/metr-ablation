"""
Runner: logistic vs isotonic.
Runs both fits and generates viewer.html.

Usage:
    python run.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from curve_models import IsotonicModel, LogisticModel
from pipeline import run_ablation

BASE_DIR = os.path.dirname(__file__)
RUNS_PATH = os.path.join(BASE_DIR, "data", "runs.jsonl")
BENCH_PATH = os.path.join(BASE_DIR, "data", "benchmark_results_1_1.yaml")
RESULTS_PATH = os.path.join(BASE_DIR, "all_results.json")

FITS = [
    ("logistic", LogisticModel(regularization=0.1),
     "Logistic regression (METR default, C=10)"),
    ("isotonic", IsotonicModel(),
     "Isotonic regression (decreasing, weighted)"),
]


def main():
    all_results = {}

    for fit_id, model_factory, description in FITS:
        output_dir = os.path.join(BASE_DIR, fit_id)

        print(f"\n{'='*60}")
        print(f"  {fit_id}: {description}")
        print(f"{'='*60}\n")

        summary = run_ablation(
            model_factory=model_factory,
            output_dir=output_dir,
            runs_path=RUNS_PATH,
            bench_path=BENCH_PATH,
            n_boot=200,
        )

        all_results[fit_id] = summary

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  ALL RESULTS SAVED TO {RESULTS_PATH}")
    print(f"{'='*60}")

    # Print comparison table
    print(f"\n{'Method':<22} {'CV-Brier':<12} {'CV-LogLoss':<12} {'IS-Brier':<12} "
          f"{'p50 Dbl(d)':<12} {'p50 R2':<10} {'G[T] Dbl(d)':<12} {'G[T] R2':<10} {'Bootstrap'}")
    print("-" * 120)
    for fit_id, _, _ in FITS:
        s = all_results[fit_id]
        fq = s["fit_quality"]
        t_p50 = s["trends"].get("p50", {})
        t_gt = s["trends"].get("G[T]", {})
        bt = s.get("bootstrap_type", "standard")
        print(f"{s['model_name']:<22} "
              f"{fq['avg_cv_brier']:<12.4f} "
              f"{fq['avg_cv_log_loss']:<12.3f} "
              f"{fq['avg_insample_brier']:<12.4f} "
              f"{t_p50.get('doubling_time_days', 'N/A'):<12} "
              f"{t_p50.get('r_squared', 'N/A'):<10} "
              f"{t_gt.get('doubling_time_days', 'N/A'):<12} "
              f"{t_gt.get('r_squared', 'N/A'):<10} "
              f"{bt}")

    # Generate viewer
    generate_viewer(all_results)


def generate_viewer(all_results):
    """Generate self-contained HTML viewer with embedded charts."""
    import base64
    from pipeline import SHORT_NAMES

    fit_ids = [f[0] for f in FITS]
    fit_labels = {f[0]: f[0].replace("_", " ").title() for f in FITS}

    model_list = []
    for entry in all_results[fit_ids[0]]["per_model_results"]:
        alias = entry["alias"]
        short = SHORT_NAMES.get(alias, alias)
        date = entry["release_date"]
        model_list.append([alias, short, date])

    p50_data = {}
    gt_data = {}
    for fid in fit_ids:
        p50_data[fid] = {}
        gt_data[fid] = {}
        for entry in all_results[fid]["per_model_results"]:
            alias = entry["alias"]
            p50_data[fid][alias] = [entry["p50_minutes"], entry["p50_ci_lo"], entry["p50_ci_hi"]]
            gt_data[fid][alias] = [entry["integral_minutes"], entry["integral_ci_lo"], entry["integral_ci_hi"]]

    p50_trend = {}
    gt_trend = {}
    for fid in fit_ids:
        t = all_results[fid]["trends"]
        p50_trend[fid] = [t["p50"]["doubling_time_days"], t["p50"]["r_squared"]]
        gt_trend[fid] = [t["G[T]"]["doubling_time_days"], t["G[T]"]["r_squared"]]

    safe_names = []
    for entry in all_results[fit_ids[0]]["per_model_results"]:
        alias = entry["alias"]
        safe = alias.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "_")
        safe_names.append(safe)

    # Read and base64-encode all pre-computed chart HTML files
    charts = {}
    for fid in fit_ids:
        fid_dir = os.path.join(BASE_DIR, fid)
        for chart_name in ["metr_trend_p50.html", "metr_trend_GT.html"]:
            path = os.path.join(fid_dir, chart_name)
            with open(path, "r", encoding="utf-8") as cf:
                content = cf.read()
            key = f"{fid}/{chart_name}"
            charts[key] = base64.b64encode(content.encode("utf-8")).decode("ascii")
        for safe in safe_names:
            for suffix in ["_fit.html", "_binned.html"]:
                path = os.path.join(fid_dir, "per_model", f"{safe}{suffix}")
                if not os.path.exists(path):
                    continue
                with open(path, "r", encoding="utf-8") as cf:
                    content = cf.read()
                key = f"{fid}/per_model/{safe}{suffix}"
                charts[key] = base64.b64encode(content.encode("utf-8")).decode("ascii")

    viewer_data = json.dumps({
        "fits": {fid: {
            "model_name": all_results[fid]["model_name"],
            "bootstrap_type": all_results[fid].get("bootstrap_type", "standard"),
            "fit_quality": all_results[fid]["fit_quality"],
            "trends": all_results[fid]["trends"],
            "description": [f[2] for f in FITS if f[0] == fid][0],
        } for fid in fit_ids},
        "fit_ids": fit_ids,
        "fit_labels": fit_labels,
        "models": model_list,
        "safe_names": safe_names,
        "p50": p50_data,
        "gt": gt_data,
        "p50_trend": p50_trend,
        "gt_trend": gt_trend,
        "charts": charts,
    }, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Ablation v2: Logistic vs Isotonic</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; }}

  header {{ background: #1b5e20; color: white; padding: 16px 24px; }}
  header h1 {{ font-size: 20px; font-weight: 600; }}
  header p {{ font-size: 12px; opacity: 0.85; margin-top: 4px; }}

  .method-bar {{ background: #fff; border-bottom: 1px solid #ddd; padding: 12px 24px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; position: sticky; top: 0; z-index: 100; }}
  .method-bar label {{ font-weight: 600; margin-right: 8px; font-size: 14px; }}
  .method-btn {{
    padding: 8px 16px; border: 2px solid #1b5e20; border-radius: 6px;
    background: #fff; color: #1b5e20; cursor: pointer; font-size: 13px; font-weight: 600;
    transition: all 0.15s;
  }}
  .method-btn:hover {{ background: #e8f5e9; }}
  .method-btn.active {{ background: #1b5e20; color: #fff; }}

  .stats-row {{ display: flex; gap: 12px; padding: 12px 24px; flex-wrap: wrap; }}
  .stat-card {{ background: #fff; border-radius: 8px; padding: 10px 16px; border: 1px solid #ddd; min-width: 140px; }}
  .stat-card .label {{ font-size: 11px; color: #666; text-transform: uppercase; }}
  .stat-card .value {{ font-size: 18px; font-weight: 700; color: #1b5e20; }}
  .stat-card .sub {{ font-size: 11px; color: #888; }}

  .section-title {{ padding: 16px 24px 8px; font-size: 18px; font-weight: 600; }}
  .section-desc {{ padding: 0 24px 12px; font-size: 13px; color: #666; }}

  .metr-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4px;
    padding: 0 24px 16px;
  }}
  .metr-cell {{
    background: #fff; border-radius: 8px; border: 1px solid #ddd; overflow: hidden; position: relative;
  }}
  .metr-cell .cell-label {{
    position: absolute; top: 4px; left: 6px; z-index: 10;
    background: rgba(27,94,32,0.9); color: white; padding: 2px 8px; border-radius: 4px;
    font-size: 11px; font-weight: 600;
  }}
  .metr-cell .cell-stats {{
    position: absolute; top: 4px; right: 6px; z-index: 10;
    background: rgba(0,0,0,0.7); color: white; padding: 2px 8px; border-radius: 4px;
    font-size: 10px;
  }}
  .iframe-wrap {{
    width: 100%; overflow: hidden;
    aspect-ratio: 1100 / 650;
  }}
  .iframe-wrap iframe {{
    border: none;
    transform-origin: 0 0;
    width: 1100px; height: 650px;
  }}

  .models-section {{ padding: 0 24px 24px; }}
  .model-row {{
    background: #fff; border-radius: 8px; border: 1px solid #ddd;
    margin-bottom: 8px; overflow: hidden;
  }}
  .model-row-header {{
    padding: 10px 16px; font-weight: 600; font-size: 15px;
    background: #f9f9f9; border-bottom: 1px solid #eee; cursor: pointer;
    display: flex; justify-content: space-between; align-items: center;
  }}
  .model-row-header:hover {{ background: #f0f0f0; }}
  .model-row-body {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }}
  .model-row-body .iframe-wrap {{ height: 500px; }}
  .model-row-body .iframe-wrap iframe {{ width: 1100px; height: 605px; }}
  .model-row.collapsed .model-row-body {{ display: none; }}

  .expand-controls {{ padding: 0 24px 8px; display: flex; gap: 8px; }}
  .expand-btn {{
    padding: 4px 12px; border: 1px solid #aaa; border-radius: 4px;
    background: #fff; cursor: pointer; font-size: 12px;
  }}
  .expand-btn:hover {{ background: #eee; }}

  .tables-section {{ padding: 0 24px 16px; }}
  .tables-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .table-wrap {{ overflow-x: auto; }}
  .table-wrap h3 {{ font-size: 14px; margin-bottom: 6px; color: #1b5e20; }}
  .cmp-table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
  .cmp-table th, .cmp-table td {{ padding: 4px 8px; border: 1px solid #ddd; text-align: right; white-space: nowrap; }}
  .cmp-table th {{ background: #1b5e20; color: white; font-weight: 600; position: sticky; top: 0; }}
  .cmp-table th:first-child {{ text-align: left; }}
  .cmp-table td:first-child {{ text-align: left; font-weight: 600; background: #f9f9f9; }}
  .cmp-table tr:hover td {{ background: #e8f5e9; }}
  .cmp-table .ci {{ font-size: 10px; color: #888; display: block; }}
</style>
</head>
<body>

<header>
  <h1>METR Time Horizon: Logistic vs Isotonic</h1>
  <p>K-fold cross-validated fit quality | m-out-of-n bootstrap for isotonic | G[T] integral metric</p>
</header>

<div class="method-bar" id="methodBar">
  <label>Fit Method:</label>
</div>

<div class="stats-row" id="statsRow"></div>

<div class="section-title">Trend Charts</div>
<div class="section-desc" id="fitDescription"></div>
<div class="metr-grid" id="metrGrid"></div>

<div class="section-title">Model Comparison Tables</div>
<div class="tables-section">
  <div class="tables-grid">
    <div class="table-wrap"><h3>P50 Threshold Crossing (minutes)</h3><div id="p50Table"></div></div>
    <div class="table-wrap"><h3>G[T] Geometric Mean (minutes)</h3><div id="gtTable"></div></div>
  </div>
</div>

<div class="section-title">Per-Model Fits</div>
<div class="expand-controls">
  <button class="expand-btn" onclick="toggleAll(true)">Expand All</button>
  <button class="expand-btn" onclick="toggleAll(false)">Collapse All</button>
</div>
<div class="models-section" id="modelsSection"></div>

<script>
const D = {viewer_data};

function chartSrc(key) {{
  return "data:text/html;base64," + D.charts[key];
}}

let currentFit = null;

function selectFit(fitId) {{
  currentFit = fitId;
  document.querySelectorAll('.method-btn').forEach(b => {{
    b.classList.toggle('active', b.dataset.fitId === fitId);
  }});
  renderStats();
  renderMetrGrid();
  renderModels();
  requestAnimationFrame(scaleIframes);
}}

function renderStats() {{
  const f = D.fits[currentFit];
  const q = f.fit_quality;
  const tp = f.trends.p50;
  const tg = f.trends["G[T]"];
  const bt = f.bootstrap_type;
  document.getElementById('statsRow').innerHTML = `
    <div class="stat-card"><div class="label">Avg CV-Brier</div><div class="value">${{q.avg_cv_brier.toFixed(4)}}</div><div class="sub">5-fold cross-validated</div></div>
    <div class="stat-card"><div class="label">Avg CV-LogLoss</div><div class="value">${{q.avg_cv_log_loss.toFixed(3)}}</div><div class="sub">5-fold cross-validated</div></div>
    <div class="stat-card"><div class="label">Avg InSample Brier</div><div class="value">${{q.avg_insample_brier.toFixed(4)}}</div><div class="sub">for reference only</div></div>
    <div class="stat-card"><div class="label">Doubling (G[T])</div><div class="value">${{tg.doubling_time_days}}d</div><div class="sub">R&sup2; = ${{tg.r_squared.toFixed(4)}}</div></div>
    <div class="stat-card"><div class="label">Doubling (p50)</div><div class="value">${{tp.doubling_time_days}}d</div><div class="sub">R&sup2; = ${{tp.r_squared.toFixed(4)}}</div></div>
    <div class="stat-card"><div class="label">Bootstrap</div><div class="value">${{bt === 'm_out_of_n' ? 'm-of-n' : 'standard'}}</div><div class="sub">${{bt === 'm_out_of_n' ? 'consistent for isotonic' : 'n-from-n'}}</div></div>
  `;
  document.getElementById('fitDescription').textContent = f.description;
}}

function renderMetrGrid() {{
  const dir = currentFit;
  const tp = D.fits[currentFit].trends.p50;
  const tg = D.fits[currentFit].trends["G[T]"];
  document.getElementById('metrGrid').innerHTML = `
    <div class="metr-cell">
      <div class="cell-label">P50 Threshold</div>
      <div class="cell-stats">${{tp.doubling_time_days}}d | R&sup2;=${{tp.r_squared.toFixed(3)}}</div>
      <div class="iframe-wrap"><iframe src="${{chartSrc(dir + '/metr_trend_p50.html')}}"></iframe></div>
    </div>
    <div class="metr-cell">
      <div class="cell-label">G[T] Geometric Mean</div>
      <div class="cell-stats">${{tg.doubling_time_days}}d | R&sup2;=${{tg.r_squared.toFixed(3)}}</div>
      <div class="iframe-wrap"><iframe src="${{chartSrc(dir + '/metr_trend_GT.html')}}"></iframe></div>
    </div>
  `;
}}

function renderModels() {{
  const dir = currentFit;
  let html = '';
  for (let i = 0; i < D.models.length; i++) {{
    const [alias, short, date] = D.models[i];
    const safe = D.safe_names[i];
    const fitSrc = chartSrc(dir + '/per_model/' + safe + '_fit.html');
    const binnedKey = dir + '/per_model/' + safe + '_binned.html';
    const hasBinned = D.charts[binnedKey] != null;
    const binnedSrc = hasBinned ? chartSrc(binnedKey) : '';
    html += `<div class="model-row" id="row-${{i}}">
      <div class="model-row-header" onclick="toggleModel(${{i}})">
        <span>${{short}} <span style="font-weight:normal;color:#888;font-size:12px">(${{date}})</span></span>
        <span class="toggle" style="font-size:12px;color:#888">Click to collapse</span>
      </div>
      <div class="model-row-body">
        <div class="iframe-wrap"><iframe src="${{fitSrc}}"></iframe></div>
        ${{hasBinned ? `<div class="iframe-wrap"><iframe src="${{binnedSrc}}"></iframe></div>` : ''}}
      </div>
    </div>`;
  }}
  document.getElementById('modelsSection').innerHTML = html;
}}

function toggleModel(i) {{
  const row = document.getElementById('row-' + i);
  const collapsed = row.classList.toggle('collapsed');
  row.querySelector('.toggle').textContent = collapsed ? 'Click to expand' : 'Click to collapse';
  if (!collapsed) requestAnimationFrame(scaleIframes);
}}

function toggleAll(expand) {{
  for (let i = 0; i < D.models.length; i++) {{
    const row = document.getElementById('row-' + i);
    if (!row) continue;
    const isCollapsed = row.classList.contains('collapsed');
    if (expand && isCollapsed) toggleModel(i);
    if (!expand && !isCollapsed) toggleModel(i);
  }}
}}

function scaleIframes() {{
  document.querySelectorAll('.iframe-wrap').forEach(wrap => {{
    const iframe = wrap.querySelector('iframe');
    if (!iframe) return;
    const nativeW = parseInt(iframe.style.width || getComputedStyle(iframe).width);
    const nativeH = parseInt(iframe.style.height || getComputedStyle(iframe).height);
    const wrapW = wrap.clientWidth;
    const wrapH = wrap.clientHeight;
    if (!wrapW || !wrapH || !nativeW || !nativeH) return;
    const scale = Math.min(wrapW / nativeW, wrapH / nativeH);
    iframe.style.transform = `scale(${{scale}})`;
  }});
}}

const _resizeObserver = new ResizeObserver(() => scaleIframes());
_resizeObserver.observe(document.body);

function fmtMin(v) {{
  if (v == null) return 'N/A';
  if (v < 1) return (v*60).toFixed(0) + 's';
  if (v < 60) return v.toFixed(1) + 'm';
  return (v/60).toFixed(1) + 'h';
}}

function buildTable(data, trendData, containerId) {{
  const fids = D.fit_ids;
  const labs = D.fit_labels;
  let html = '<table class="cmp-table"><thead><tr><th>Model</th>';
  for (const fid of fids) html += `<th>${{labs[fid]}}</th>`;
  html += '</tr></thead><tbody>';

  for (const [alias, short, date] of D.models) {{
    html += `<tr><td>${{short}}</td>`;
    for (const fid of fids) {{
      const entry = data[fid]?.[alias];
      if (!entry || entry[0] == null) {{
        html += '<td>N/A</td>';
        continue;
      }}
      const [val, lo, hi] = entry;
      const ciLo = lo != null ? Math.min(lo, hi) : null;
      const ciHi = lo != null ? Math.max(lo, hi) : null;
      const ciStr = ciLo != null ? `<span class="ci">(${{fmtMin(ciLo)}}\\u2013${{fmtMin(ciHi)}})</span>` : '';
      html += `<td>${{fmtMin(val)}}${{ciStr}}</td>`;
    }}
    html += '</tr>';
  }}

  html += '<tr style="border-top:2px solid #1b5e20;font-weight:700;background:#e8f5e9"><td>Doubling time</td>';
  for (const fid of fids) {{
    const [dt, r2] = trendData[fid];
    html += `<td>${{dt.toFixed(0)}}d<span class="ci">R\\u00b2=${{r2.toFixed(3)}}</span></td>`;
  }}
  html += '</tr>';
  html += '</tbody></table>';
  document.getElementById(containerId).innerHTML = html;
}}

// Init
const bar = document.getElementById('methodBar');
for (const fid of D.fit_ids) {{
  const btn = document.createElement('button');
  btn.className = 'method-btn';
  btn.dataset.fitId = fid;
  btn.textContent = D.fit_labels[fid];
  btn.onclick = () => selectFit(fid);
  bar.appendChild(btn);
}}

buildTable(D.p50, D.p50_trend, 'p50Table');
buildTable(D.gt, D.gt_trend, 'gtTable');

selectFit(D.fit_ids[0]);
</script>
</body>
</html>"""

    viewer_path = os.path.join(BASE_DIR, "viewer.html")
    with open(viewer_path, "w") as f:
        f.write(html)
    print(f"\n  Viewer saved to {viewer_path}")


if __name__ == "__main__":
    main()
