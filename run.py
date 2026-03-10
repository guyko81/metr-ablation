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
from pipeline import run_ablation, plot_metr_trend_comparison

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
          f"{'p50 Dbl(d)':<12} {'p50 R2':<10} {'p80 Dbl(d)':<12} {'p80 R2':<10} "
          f"{'G[T] Dbl(d)':<12} {'G[T] R2':<10} {'Bootstrap'}")
    print("-" * 150)
    for fit_id, _, _ in FITS:
        s = all_results[fit_id]
        fq = s["fit_quality"]
        t_p50 = s["trends"].get("p50", {})
        t_p80 = s["trends"].get("p80", {})
        t_gt = s["trends"].get("G[T]", {})
        bt = s.get("bootstrap_type", "standard")
        print(f"{s['model_name']:<22} "
              f"{fq['avg_cv_brier']:<12.4f} "
              f"{fq['avg_cv_log_loss']:<12.3f} "
              f"{fq['avg_insample_brier']:<12.4f} "
              f"{t_p50.get('doubling_time_days', 'N/A'):<12} "
              f"{t_p50.get('r_squared', 'N/A'):<10} "
              f"{t_p80.get('doubling_time_days', 'N/A'):<12} "
              f"{t_p80.get('r_squared', 'N/A'):<10} "
              f"{t_gt.get('doubling_time_days', 'N/A'):<12} "
              f"{t_gt.get('r_squared', 'N/A'):<10} "
              f"{bt}")

    # Generate comparison charts
    generate_comparison_charts(all_results)

    # Generate viewer
    generate_viewer(all_results)


def generate_comparison_charts(all_results):
    """Generate logistic-vs-isotonic comparison charts and update index.html."""
    import base64
    import matplotlib.pyplot as plt

    charts_b64 = {}
    for metric_label, point_col, lo_col, hi_col in [
        ("p50", "p50_minutes", "p50_ci_lo", "p50_ci_hi"),
        ("G[T]", "integral_minutes", "integral_ci_lo", "integral_ci_hi"),
    ]:
        fig = plot_metr_trend_comparison(all_results, metric_label, point_col, lo_col, hi_col)
        safe = metric_label.replace("[", "").replace("]", "")
        path = os.path.join(BASE_DIR, f"comparison_{safe}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        with open(path, "rb") as f:
            charts_b64[safe] = base64.b64encode(f.read()).decode("ascii")
        print(f"  Comparison chart saved: {path}")

    # Update index.html with embedded charts
    index_path = os.path.join(BASE_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Insert or replace comparison section
    comparison_html = f'''  <section>
    <h2>Logistic vs Isotonic: Direct Comparison</h2>
    <p style="font-size:14px; margin-bottom: 12px;">Both methods plotted on the same axes. Blue diamonds = logistic, green circles = isotonic (bootstrap median crossings).</p>
    <div style="display: flex; flex-direction: column; gap: 18px;">
      <div style="background: #fff; border: 1px solid #ddd; border-radius: 10px; overflow: hidden; padding: 8px;">
        <img src="data:image/png;base64,{charts_b64['p50']}" style="width: 100%; height: auto;" alt="p50 comparison">
      </div>
      <div style="background: #fff; border: 1px solid #ddd; border-radius: 10px; overflow: hidden; padding: 8px;">
        <img src="data:image/png;base64,{charts_b64['GT']}" style="width: 100%; height: auto;" alt="G[T] comparison">
      </div>
    </div>
  </section>'''

    # Check if comparison section already exists
    marker_start = '<!-- COMPARISON_CHARTS_START -->'
    marker_end = '<!-- COMPARISON_CHARTS_END -->'
    comparison_block = f"{marker_start}\n{comparison_html}\n{marker_end}"

    if marker_start in html:
        import re
        html = re.sub(
            f'{marker_start}.*?{marker_end}',
            comparison_block,
            html,
            flags=re.DOTALL
        )
    else:
        # Insert before the viewer-promo section
        html = html.replace(
            '  <div class="viewer-promo">',
            f'{comparison_block}\n\n  <div class="viewer-promo">'
        )

    # Remove old iframe scaling script if present
    if 'scaleComparisonIframes' in html:
        import re
        html = re.sub(
            r'<script>\s*function scaleComparisonIframes\(\).*?</script>\s*',
            '',
            html,
            flags=re.DOTALL
        )

    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Index page updated with comparison charts")


def generate_viewer(all_results):
    """Generate self-contained HTML viewer with embedded PNG charts."""
    import base64
    import numpy as np
    import matplotlib.pyplot as plt
    from io import BytesIO
    from pipeline import (SHORT_NAMES, plot_metr_trend_comparison,
                          plot_per_model_fit_compare, plot_per_model_binned_compare)

    fit_ids = [f[0] for f in FITS]
    fit_labels = {f[0]: f[0].replace("_", " ").title() for f in FITS}

    model_list = []
    for entry in all_results[fit_ids[0]]["per_model_results"]:
        alias = entry["alias"]
        short = SHORT_NAMES.get(alias, alias)
        date = entry["release_date"]
        model_list.append([alias, short, date])

    p50_data = {}
    p80_data = {}
    gt_data = {}
    for fid in fit_ids:
        p50_data[fid] = {}
        p80_data[fid] = {}
        gt_data[fid] = {}
        for entry in all_results[fid]["per_model_results"]:
            alias = entry["alias"]
            p50_data[fid][alias] = [entry["p50_minutes"], entry["p50_ci_lo"], entry["p50_ci_hi"]]
            p80_data[fid][alias] = [entry["p80_minutes"], entry["p80_ci_lo"], entry["p80_ci_hi"]]
            gt_data[fid][alias] = [entry["integral_minutes"], entry["integral_ci_lo"], entry["integral_ci_hi"]]

    p50_trend = {}
    p80_trend = {}
    gt_trend = {}
    for fid in fit_ids:
        t = all_results[fid]["trends"]
        p50_trend[fid] = [t["p50"]["doubling_time_days"], t["p50"]["r_squared"]]
        p80_trend[fid] = [t["p80"]["doubling_time_days"], t["p80"]["r_squared"]]
        gt_trend[fid] = [t["G[T]"]["doubling_time_days"], t["G[T]"]["r_squared"]]

    safe_names = []
    for entry in all_results[fit_ids[0]]["per_model_results"]:
        alias = entry["alias"]
        safe = alias.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "_")
        safe_names.append(safe)

    # Read and base64-encode all pre-computed PNG chart files
    charts = {}
    for fid in fit_ids:
        fid_dir = os.path.join(BASE_DIR, fid)
        for chart_name in ["metr_trend_p50.png", "metr_trend_p80.png", "metr_trend_GT.png"]:
            path = os.path.join(fid_dir, chart_name)
            with open(path, "rb") as cf:
                content = cf.read()
            key = f"{fid}/{chart_name}"
            charts[key] = base64.b64encode(content).decode("ascii")
        for safe in safe_names:
            for suffix in ["_fit.png", "_binned.png"]:
                path = os.path.join(fid_dir, "per_model", f"{safe}{suffix}")
                if not os.path.exists(path):
                    continue
                with open(path, "rb") as cf:
                    content = cf.read()
                key = f"{fid}/per_model/{safe}{suffix}"
                charts[key] = base64.b64encode(content).decode("ascii")

    # Generate "compare" trend charts (both methods overlaid)
    for metric_label, point_col, lo_col, hi_col in [
        ("p50", "p50_minutes", "p50_ci_lo", "p50_ci_hi"),
        ("p80", "p80_minutes", "p80_ci_lo", "p80_ci_hi"),
        ("G[T]", "integral_minutes", "integral_ci_lo", "integral_ci_hi"),
    ]:
        fig = plot_metr_trend_comparison(all_results, metric_label, point_col, lo_col, hi_col)
        safe_metric = metric_label.replace("[", "").replace("]", "")
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        charts[f"compare/metr_trend_{safe_metric}.png"] = base64.b64encode(buf.read()).decode("ascii")

    # Generate per-model comparison charts (both fits overlaid)
    for safe in safe_names:
        # Load curve data from both methods
        curve_files = {}
        for fid in fit_ids:
            path = os.path.join(BASE_DIR, fid, "per_model", f"{safe}_curves.npz")
            if os.path.exists(path):
                curve_files[fid] = dict(np.load(path))
        if len(curve_files) < 2:
            continue
        scatter_data = {
            "human_minutes": curve_files[fit_ids[0]]["human_minutes"],
            "score_binarized": curve_files[fit_ids[0]]["score_binarized"],
            "weights": curve_files[fit_ids[0]]["weights"],
        }
        alias = None
        for m in model_list:
            s = m[0].replace(" ", "_").replace("(", "").replace(")", "").replace(".", "_")
            if s == safe:
                alias = m[0]
                break
        if alias is None:
            continue

        # Comparison fit chart
        fig_cmp = plot_per_model_fit_compare(alias, scatter_data,
                                             curve_files["logistic"], curve_files["isotonic"])
        buf = BytesIO()
        fig_cmp.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig_cmp)
        buf.seek(0)
        charts[f"compare/per_model/{safe}_fit.png"] = base64.b64encode(buf.read()).decode("ascii")

        # Comparison binned chart
        log_primary = curve_files["logistic"]["y_full"]
        iso_primary = curve_files["isotonic"]["ci_median"]
        fig_bin = plot_per_model_binned_compare(
            alias, scatter_data,
            curve_files["logistic"]["x_grid"], log_primary,
            curve_files["isotonic"]["x_grid"], iso_primary,
        )
        if fig_bin is not None:
            buf = BytesIO()
            fig_bin.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig_bin)
            buf.seek(0)
            charts[f"compare/per_model/{safe}_binned.png"] = base64.b64encode(buf.read()).decode("ascii")

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
        "p80": p80_data,
        "gt": gt_data,
        "p50_trend": p50_trend,
        "p80_trend": p80_trend,
        "gt_trend": gt_trend,
        "charts": charts,
    }, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Detailed Comparison: Logistic vs Isotonic</title>
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
    grid-template-columns: 1fr 1fr 1fr;
    gap: 8px;
    padding: 0 24px 16px;
  }}
  .metr-cell {{
    background: #fff; border-radius: 8px; border: 1px solid #ddd; overflow: hidden; position: relative;
  }}
  .metr-cell .cell-label {{
    position: absolute; top: 4px; left: 6px; z-index: 10;
    background: rgba(27,94,32,0.9); color: white; padding: 2px 8px; border-radius: 4px;
    font-size: 10px; font-weight: 600;
  }}
  .metr-cell .cell-stats {{
    position: absolute; top: 4px; right: 6px; z-index: 10;
    background: rgba(0,0,0,0.7); color: white; padding: 2px 8px; border-radius: 4px;
    font-size: 9px;
  }}
  .metr-cell img {{ width: 100%; height: auto; display: block; }}

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
  .model-row-body {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px; padding: 4px; }}
  .model-row-body img {{ width: 100%; height: auto; }}
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
  <h1>Detailed Comparison: Logistic vs Isotonic</h1>
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
  <div class="tables-grid" style="grid-template-columns: 1fr 1fr 1fr;">
    <div class="table-wrap"><h3>P50 Threshold Crossing (minutes)</h3><div id="p50Table"></div></div>
    <div class="table-wrap"><h3>P80 Threshold Crossing (minutes)</h3><div id="p80Table"></div></div>
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

function imgSrc(key) {{
  return "data:image/png;base64," + D.charts[key];
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
}}

function renderStats() {{
  if (currentFit === 'compare') {{
    // Show stats for both methods side by side
    let cards = '';
    for (const fid of D.fit_ids) {{
      const f = D.fits[fid];
      const q = f.fit_quality;
      const lab = D.fit_labels[fid];
      cards += `
        <div class="stat-card"><div class="label">${{lab}} CV-Brier</div><div class="value">${{q.avg_cv_brier.toFixed(4)}}</div></div>
        <div class="stat-card"><div class="label">${{lab}} Dbl(p50)</div><div class="value">${{f.trends.p50.doubling_time_days}}d</div><div class="sub">R&sup2;=${{f.trends.p50.r_squared.toFixed(3)}}</div></div>
        <div class="stat-card"><div class="label">${{lab}} Dbl(G[T])</div><div class="value">${{f.trends["G[T]"].doubling_time_days}}d</div><div class="sub">R&sup2;=${{f.trends["G[T]"].r_squared.toFixed(3)}}</div></div>
      `;
    }}
    document.getElementById('statsRow').innerHTML = cards;
    document.getElementById('fitDescription').textContent = 'Both logistic and isotonic fits overlaid for direct comparison.';
    return;
  }}
  const f = D.fits[currentFit];
  const q = f.fit_quality;
  const tp = f.trends.p50;
  const t8 = f.trends.p80;
  const tg = f.trends["G[T]"];
  const bt = f.bootstrap_type;
  document.getElementById('statsRow').innerHTML = `
    <div class="stat-card"><div class="label">Avg CV-Brier</div><div class="value">${{q.avg_cv_brier.toFixed(4)}}</div><div class="sub">5-fold cross-validated</div></div>
    <div class="stat-card"><div class="label">Avg CV-LogLoss</div><div class="value">${{q.avg_cv_log_loss.toFixed(3)}}</div><div class="sub">5-fold cross-validated</div></div>
    <div class="stat-card"><div class="label">Avg InSample Brier</div><div class="value">${{q.avg_insample_brier.toFixed(4)}}</div><div class="sub">for reference only</div></div>
    <div class="stat-card"><div class="label">Doubling (p50)</div><div class="value">${{tp.doubling_time_days}}d</div><div class="sub">R&sup2; = ${{tp.r_squared.toFixed(4)}}</div></div>
    <div class="stat-card"><div class="label">Doubling (p80)</div><div class="value">${{t8.doubling_time_days}}d</div><div class="sub">R&sup2; = ${{t8.r_squared.toFixed(4)}}</div></div>
    <div class="stat-card"><div class="label">Doubling (G[T])</div><div class="value">${{tg.doubling_time_days}}d</div><div class="sub">R&sup2; = ${{tg.r_squared.toFixed(4)}}</div></div>
    <div class="stat-card"><div class="label">Bootstrap</div><div class="value">${{bt === 'm_out_of_n' ? 'm-of-n' : 'standard'}}</div><div class="sub">${{bt === 'm_out_of_n' ? 'consistent for isotonic' : 'n-from-n'}}</div></div>
  `;
  document.getElementById('fitDescription').textContent = f.description;
}}

function renderMetrGrid() {{
  if (currentFit === 'compare') {{
    document.getElementById('metrGrid').innerHTML = `
      <div class="metr-cell">
        <div class="cell-label">P50 Comparison</div>
        <img src="${{imgSrc('compare/metr_trend_p50.png')}}">
      </div>
      <div class="metr-cell">
        <div class="cell-label">P80 Comparison</div>
        <img src="${{imgSrc('compare/metr_trend_p80.png')}}">
      </div>
      <div class="metr-cell">
        <div class="cell-label">G[T] Comparison</div>
        <img src="${{imgSrc('compare/metr_trend_GT.png')}}">
      </div>
    `;
    return;
  }}
  const dir = currentFit;
  const tp = D.fits[currentFit].trends.p50;
  const t8 = D.fits[currentFit].trends.p80;
  const tg = D.fits[currentFit].trends["G[T]"];
  document.getElementById('metrGrid').innerHTML = `
    <div class="metr-cell">
      <div class="cell-label">P50 Threshold</div>
      <div class="cell-stats">${{tp.doubling_time_days}}d | R&sup2;=${{tp.r_squared.toFixed(3)}}</div>
      <img src="${{imgSrc(dir + '/metr_trend_p50.png')}}">
    </div>
    <div class="metr-cell">
      <div class="cell-label">P80 Threshold</div>
      <div class="cell-stats">${{t8.doubling_time_days}}d | R&sup2;=${{t8.r_squared.toFixed(3)}}</div>
      <img src="${{imgSrc(dir + '/metr_trend_p80.png')}}">
    </div>
    <div class="metr-cell">
      <div class="cell-label">G[T] Geometric Mean</div>
      <div class="cell-stats">${{tg.doubling_time_days}}d | R&sup2;=${{tg.r_squared.toFixed(3)}}</div>
      <img src="${{imgSrc(dir + '/metr_trend_GT.png')}}">
    </div>
  `;
}}

function renderModels() {{
  if (currentFit === 'compare') {{
    // In compare mode, show overlaid comparison charts per model
    let html = '';
    for (let i = 0; i < D.models.length; i++) {{
      const [alias, short, date] = D.models[i];
      const safe = D.safe_names[i];
      const fitKey = 'compare/per_model/' + safe + '_fit.png';
      const binnedKey = 'compare/per_model/' + safe + '_binned.png';
      const hasFit = D.charts[fitKey] != null;
      const hasBinned = D.charts[binnedKey] != null;
      if (!hasFit) continue;
      html += `<div class="model-row" id="row-${{i}}">
        <div class="model-row-header" onclick="toggleModel(${{i}})">
          <span>${{short}} <span style="font-weight:normal;color:#888;font-size:12px">(${{date}})</span></span>
          <span class="toggle" style="font-size:12px;color:#888">Click to collapse</span>
        </div>
        <div class="model-row-body">
          <img src="${{imgSrc(fitKey)}}">
          ${{hasBinned ? `<img src="${{imgSrc(binnedKey)}}">` : ''}}
        </div>
      </div>`;
    }}
    document.getElementById('modelsSection').innerHTML = html;
    return;
  }}
  const dir = currentFit;
  let html = '';
  for (let i = 0; i < D.models.length; i++) {{
    const [alias, short, date] = D.models[i];
    const safe = D.safe_names[i];
    const fitKey = dir + '/per_model/' + safe + '_fit.png';
    const binnedKey = dir + '/per_model/' + safe + '_binned.png';
    const hasBinned = D.charts[binnedKey] != null;
    html += `<div class="model-row" id="row-${{i}}">
      <div class="model-row-header" onclick="toggleModel(${{i}})">
        <span>${{short}} <span style="font-weight:normal;color:#888;font-size:12px">(${{date}})</span></span>
        <span class="toggle" style="font-size:12px;color:#888">Click to collapse</span>
      </div>
      <div class="model-row-body">
        <img src="${{imgSrc(fitKey)}}">
        ${{hasBinned ? `<img src="${{imgSrc(binnedKey)}}">` : ''}}
      </div>
    </div>`;
  }}
  document.getElementById('modelsSection').innerHTML = html;
}}

function toggleModel(i) {{
  const row = document.getElementById('row-' + i);
  const collapsed = row.classList.toggle('collapsed');
  row.querySelector('.toggle').textContent = collapsed ? 'Click to expand' : 'Click to collapse';
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

// Add Compare button
const cmpBtn = document.createElement('button');
cmpBtn.className = 'method-btn';
cmpBtn.dataset.fitId = 'compare';
cmpBtn.textContent = 'Compare';
cmpBtn.onclick = () => selectFit('compare');
bar.appendChild(cmpBtn);

buildTable(D.p50, D.p50_trend, 'p50Table');
buildTable(D.p80, D.p80_trend, 'p80Table');
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
