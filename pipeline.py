"""
Ablation v2 pipeline: logistic vs isotonic vs smoothed isotonic.

Key differences from v1:
1. K-fold cross-validated fit quality (Brier, LogLoss) — no more in-sample cheating
2. m-out-of-n bootstrap for isotonic models (fixes bootstrap inconsistency)
3. G[T] integral metric alongside p50 crossing
4. Focused comparison of three methods only
"""

import copy
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALIAS_TO_BENCH = {
    "GPT-4 0314":                        "gpt_4",
    "GPT-4 1106 (Inspect)":              "gpt_4_1106_inspect",
    "Claude 3 Opus (Inspect)":           "claude_3_opus_inspect",
    "GPT-4 Turbo (Inspect)":             "gpt_4_turbo_inspect",
    "GPT-4o (Inspect)":                  "gpt_4o_inspect",
    "Claude 3.5 Sonnet (Old) (Inspect)": "claude_3_5_sonnet_20240620_inspect",
    "o1-preview":                         "o1_preview",
    "Claude 3.5 Sonnet (New) (Inspect)": "claude_3_5_sonnet_20241022_inspect",
    "o1 (Inspect)":                       "o1_inspect",
    "Claude 3.7 Sonnet (Inspect)":       "claude_3_7_sonnet_inspect",
    "o3 (Inspect)":                       "o3_inspect",
    "Claude 4 Opus (Inspect)":           "claude_4_opus_inspect",
    "Claude 4.1 Opus (Inspect)":         "claude_4_1_opus_inspect",
    "GPT-5 (Inspect)":                   "gpt_5_2025_08_07_inspect",
    "Gemini 3 Pro":                       "gemini_3_pro",
    "GPT-5.1-Codex-Max (Inspect)":       "gpt_5_1_codex_max_inspect",
    "Claude Opus 4.5 (Inspect)":         "claude_opus_4_5_inspect",
    "GPT-5.2":                            "gpt_5_2",
}

SHORT_NAMES = {
    "GPT-4 0314": "GPT-4",
    "GPT-4 1106 (Inspect)": "GPT-4 Turbo",
    "Claude 3 Opus (Inspect)": "Claude 3 Opus",
    "GPT-4 Turbo (Inspect)": "GPT-4 Turbo (Apr)",
    "GPT-4o (Inspect)": "GPT-4o",
    "Claude 3.5 Sonnet (Old) (Inspect)": "Claude 3.5 Sonnet",
    "o1-preview": "o1-preview",
    "Claude 3.5 Sonnet (New) (Inspect)": "Claude 3.5 Sonnet (New)",
    "o1 (Inspect)": "o1",
    "Claude 3.7 Sonnet (Inspect)": "Claude 3.7 Sonnet",
    "o3 (Inspect)": "o3",
    "Claude 4 Opus (Inspect)": "Claude 4 Opus",
    "Claude 4.1 Opus (Inspect)": "Claude 4.1 Opus",
    "GPT-5 (Inspect)": "GPT-5",
    "Gemini 3 Pro": "Gemini 3 Pro",
    "GPT-5.1-Codex-Max (Inspect)": "GPT-5.1 Codex",
    "Claude Opus 4.5 (Inspect)": "Claude Opus 4.5",
    "GPT-5.2": "GPT-5.2",
}

WEIGHT_KEY = "invsqrt_task_weight"

THRESHOLDS = [0.50]

BAR_TIMES = [1/6, 0.5, 1, 2, 4, 8, 15, 30, 60, 2*60, 4*60, 8*60, 16*60, 24*60]


def fmt_time(minutes):
    if minutes < 1:
        return f"{minutes*60:.0f}s"
    if minutes < 60:
        return f"{minutes:.0f}m"
    if minutes < 24 * 60:
        return f"{minutes/60:.1f}h"
    return f"{minutes/60/24:.1f}d"


# ---------------------------------------------------------------------------
# K-fold cross-validated fit quality
# ---------------------------------------------------------------------------

def kfold_fit_quality(X_log2, y, weights, model_factory, n_folds=5, seed=42):
    """Compute k-fold cross-validated Brier score and log-loss.

    Stratified by outcome (0/1) to ensure each fold has both classes.
    Returns dict with cv_brier, cv_log_loss (out-of-sample averages).
    """
    rng = np.random.default_rng(seed)
    n = len(X_log2)

    # Stratified fold assignment
    idx_1 = np.where(y == 1)[0]
    idx_0 = np.where(y == 0)[0]
    rng.shuffle(idx_1)
    rng.shuffle(idx_0)

    folds = np.empty(n, dtype=int)
    for i, idx in enumerate(idx_1):
        folds[idx] = i % n_folds
    for i, idx in enumerate(idx_0):
        folds[idx] = i % n_folds

    oof_brier_num, oof_brier_den = 0.0, 0.0
    oof_ll_num, oof_ll_den = 0.0, 0.0

    for fold in range(n_folds):
        train_mask = folds != fold
        test_mask = folds == fold

        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        # Need both classes in train
        y_train = y[train_mask]
        if len(np.unique(y_train.astype(int))) < 2:
            continue

        m = copy.deepcopy(model_factory)
        try:
            m.fit(X_log2[train_mask], y[train_mask], weights[train_mask])
        except Exception:
            continue

        p_test = m.predict(X_log2[test_mask])
        p_test = np.clip(p_test, 1e-8, 1 - 1e-8)
        y_test = y[test_mask]
        w_test = weights[test_mask]

        oof_brier_num += np.sum(w_test * (p_test - y_test) ** 2)
        oof_brier_den += w_test.sum()

        oof_ll_num += -np.sum(w_test * (y_test * np.log(p_test) + (1 - y_test) * np.log(1 - p_test)))
        oof_ll_den += w_test.sum()

    cv_brier = oof_brier_num / oof_brier_den if oof_brier_den > 0 else float("nan")
    cv_log_loss = oof_ll_num / oof_ll_den if oof_ll_den > 0 else float("nan")

    return {
        "cv_brier": round(float(cv_brier), 6),
        "cv_log_loss": round(float(cv_log_loss), 6),
        "n_folds": n_folds,
    }


# ---------------------------------------------------------------------------
# In-sample fit quality (for reference, clearly labeled)
# ---------------------------------------------------------------------------

def insample_fit_quality(X_log2, y, weights, x_grid, y_pred):
    """In-sample Brier and log-loss from an already-fitted curve."""
    p_at_obs = np.interp(X_log2, x_grid, y_pred)
    p_at_obs = np.clip(p_at_obs, 1e-8, 1 - 1e-8)
    brier = np.sum(weights * (p_at_obs - y) ** 2) / weights.sum()
    log_loss = -np.sum(weights * (y * np.log(p_at_obs) + (1 - y) * np.log(1 - p_at_obs))) / weights.sum()
    return {
        "insample_brier": round(float(brier), 6),
        "insample_log_loss": round(float(log_loss), 6),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_crossing(x_grid_log2, curve, threshold):
    """Find minutes where a descending curve crosses threshold."""
    above = curve >= threshold
    if not above.any() or above.all():
        return None
    last_above = np.where(above)[0][-1]
    if last_above >= len(x_grid_log2) - 1:
        return 2 ** x_grid_log2[-1]
    x1, x2 = x_grid_log2[last_above], x_grid_log2[last_above + 1]
    y1, y2 = curve[last_above], curve[last_above + 1]
    x_cross = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
    return 2 ** x_cross


def compute_integral_metric(x_grid_log2, curve):
    """Compute geometric mean capability time G[T].

    Treats q(u) = 1 - p(u) as a CDF of capability boundary in log2-space.
    E[U] = u_min + integral(p(u) du), then G[T] = 2^E[U].
    """
    area = np.trapezoid(curve, x_grid_log2)
    u_min = x_grid_log2[0]
    return 2 ** (u_min + area)


def get_time_ticks(minutes_arr):
    tick_minutes = [1, 2, 4, 8, 15, 30, 60, 2*60, 4*60, 8*60, 16*60, 24*60]
    lo, hi = np.min(minutes_arr), np.max(minutes_arr)
    return [t for t in tick_minutes if t >= lo * 0.5 and t <= hi * 2]


# ---------------------------------------------------------------------------
# Bootstrap with correct method selection
# ---------------------------------------------------------------------------

def bootstrap_curves_and_crossings(
    X_log2, y, weights, model_factory, x_grid, n_boot=200, seed=42, thresholds=None
):
    """Bootstrap with method-appropriate resampling.

    For models with bootstrap_type == "m_out_of_n":
        Uses m-out-of-n bootstrap (m = n^(2/3)) which is consistent for
        isotonic regression at the n^(1/3) rate.
        Reference: Léger & MacGibbon (2006), Sen & Xu (2015).

    For models with bootstrap_type == "standard":
        Standard nonparametric bootstrap (resample n from n with replacement).
    """
    if thresholds is None:
        thresholds = THRESHOLDS

    rng = np.random.default_rng(seed)
    n = len(X_log2)
    n_grid = len(x_grid)

    use_m_of_n = getattr(model_factory, "bootstrap_type", "standard") == "m_out_of_n"
    m = int(np.ceil(n ** (2.0 / 3.0))) if use_m_of_n else n

    boot_curves = np.zeros((n_boot, n_grid))
    boot_crossings = {thr: [] for thr in thresholds}
    boot_integrals = []

    for b in range(n_boot):
        idx = rng.choice(n, size=m, replace=True)
        m_b = copy.deepcopy(model_factory)
        try:
            m_b.fit(X_log2[idx], y[idx], weights[idx])
            curve_b = m_b.predict(x_grid)
            boot_curves[b] = curve_b
            for thr in thresholds:
                c = find_crossing(x_grid, curve_b, thr)
                if c is not None:
                    boot_crossings[thr].append(c)
            boot_integrals.append(compute_integral_metric(x_grid, curve_b))
        except Exception:
            boot_curves[b] = np.nan

    ci_lo = np.nanpercentile(boot_curves, 5, axis=0)
    ci_hi = np.nanpercentile(boot_curves, 95, axis=0)

    crossings_dict = {}
    for thr in thresholds:
        arr = boot_crossings[thr]
        crossings_dict[thr] = {
            "ci_lo": np.percentile(arr, 5) if len(arr) >= 10 else None,
            "ci_hi": np.percentile(arr, 95) if len(arr) >= 10 else None,
        }

    integral_ci = {
        "ci_lo": np.percentile(boot_integrals, 5) if len(boot_integrals) >= 10 else None,
        "ci_hi": np.percentile(boot_integrals, 95) if len(boot_integrals) >= 10 else None,
    }

    boot_method = f"m-out-of-n (m={m})" if use_m_of_n else f"standard (n={n})"

    return ci_lo, ci_hi, crossings_dict, integral_ci, boot_method


# ---------------------------------------------------------------------------
# Per-model plot: scatter + curve + bootstrap CI
# ---------------------------------------------------------------------------

def plot_per_model_fit(alias, df_model, model_factory, n_boot=200, n_grid=500, seed=42):
    """Fit model, bootstrap CI, return (fig, results_dict, x_grid, y_full)."""
    X = np.log2(df_model["human_minutes"].values)
    y = df_model["score_binarized"].values.astype(float)
    w = df_model[WEIGHT_KEY].values

    x_grid = np.linspace(X.min() - 0.5, X.max() + 0.5, n_grid)
    minutes_grid = 2 ** x_grid

    # Full-data fit
    model = copy.deepcopy(model_factory)
    model.fit(X, y, w)
    y_full = model.predict(x_grid)

    # Point estimates
    p50_point = find_crossing(x_grid, y_full, 0.50)
    integral_point = compute_integral_metric(x_grid, y_full)

    # Bootstrap
    ci_lo, ci_hi, crossings_dict, integral_ci, boot_method = bootstrap_curves_and_crossings(
        X, y, w, model_factory, x_grid, n_boot=n_boot, seed=seed
    )

    # K-fold CV
    cv_stats = kfold_fit_quality(X, y, w, model_factory, n_folds=5, seed=seed)

    # In-sample stats
    is_stats = insample_fit_quality(X, y, w, x_grid, y_full)

    # Build figure
    rng_j = np.random.default_rng(42)
    jitter = rng_j.uniform(-0.03, 0.03, len(df_model))
    colors = df_model["score_binarized"].map({1: "royalblue", 0: "salmon"}).values
    ws = df_model[WEIGHT_KEY].values
    marker_size = 4 + 20 * (ws / ws.max())
    visible_ticks = get_time_ticks(minutes_grid)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_model["human_minutes"].values, y=df_model["score_binarized"].values + jitter,
        mode="markers", name="Runs",
        marker=dict(size=marker_size, color=colors, opacity=0.4,
                    line=dict(width=0.5, color="white")),
    ))

    ci_label = f"90% CI ({boot_method})"
    fig.add_trace(go.Scatter(
        x=np.concatenate([minutes_grid, minutes_grid[::-1]]),
        y=np.concatenate([ci_hi, ci_lo[::-1]]),
        fill="toself", fillcolor="rgba(100,100,200,0.2)",
        line=dict(width=0), name=ci_label, hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=minutes_grid, y=y_full,
        mode="lines", name="Fit",
        line=dict(color="black", width=2.5),
    ))

    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.4)

    fig.update_layout(
        title=(
            f"{alias} - {model_factory.name}<br>"
            f"<span style='font-size:11px;color:#666'>"
            f"CV Brier: {cv_stats['cv_brier']:.4f} | "
            f"CV LogLoss: {cv_stats['cv_log_loss']:.3f} | "
            f"InSample Brier: {is_stats['insample_brier']:.4f} | "
            f"{ci_label}</span>"
        ),
        xaxis=dict(title="Task length (human time)", type="log",
                   tickvals=visible_ticks, ticktext=[fmt_time(t) for t in visible_ticks]),
        yaxis=dict(title="P(success)", range=[-0.05, 1.05]),
        template="plotly_white", width=1000, height=550,
        legend=dict(x=0.01, y=0.01, xanchor="left", yanchor="bottom"),
    )

    results = {
        "p50_minutes": p50_point,
        "p50_ci_lo": crossings_dict.get(0.50, {}).get("ci_lo"),
        "p50_ci_hi": crossings_dict.get(0.50, {}).get("ci_hi"),
        "integral_minutes": integral_point,
        "integral_ci_lo": integral_ci["ci_lo"],
        "integral_ci_hi": integral_ci["ci_hi"],
        "boot_method": boot_method,
        **cv_stats,
        **is_stats,
    }

    return fig, results, x_grid, y_full


# ---------------------------------------------------------------------------
# METR trend chart
# ---------------------------------------------------------------------------

def plot_metr_trend(res_df, model_name_str,
                    point_col="p50_minutes", ci_lo_col="p50_ci_lo", ci_hi_col="p50_ci_hi",
                    metric_label="p50", avg_cv_brier=None, avg_cv_log_loss=None):
    """METR-style trend chart: metric vs release date with exponential fit."""
    res_df = res_df.dropna(subset=[point_col]).sort_values("release_date").copy()

    dates = res_df["release_date"].values
    vals = res_df[point_col].values
    ci_lo = res_df[ci_lo_col].values if ci_lo_col in res_df.columns else vals
    ci_hi = res_df[ci_hi_col].values if ci_hi_col in res_df.columns else vals
    labels = res_df["alias"].values

    date_nums = np.array(
        [(pd.Timestamp(d) - pd.Timestamp("1970-01-01")).days for d in dates], dtype=float
    )

    log_vals = np.log(vals)
    valid = np.isfinite(log_vals) & (vals > 0)

    slope, intercept, r_value, _, _ = linregress(date_nums[valid], log_vals[valid])
    doubling_time_days = np.log(2) / slope

    pad_before, pad_after = 180, 365
    date_range = np.linspace(date_nums.min() - pad_before, date_nums.max() + pad_after, 500)
    dates_range_ts = pd.to_datetime(date_range, unit="D", origin="1970-01-01")

    trend = np.exp(intercept + slope * date_range)
    residuals = log_vals[valid] - (intercept + slope * date_nums[valid])
    res_std = np.std(residuals)
    band_1lo = np.exp(intercept + slope * date_range - res_std)
    band_1hi = np.exp(intercept + slope * date_range + res_std)
    band_2lo = np.exp(intercept + slope * date_range - 1.96 * res_std)
    band_2hi = np.exp(intercept + slope * date_range + 1.96 * res_std)

    fig = go.Figure()

    # 2-sigma band
    fig.add_trace(go.Scatter(
        x=np.concatenate([dates_range_ts, dates_range_ts[::-1]]),
        y=np.concatenate([band_2hi, band_2lo[::-1]]),
        fill="toself", fillcolor="rgba(46,125,50,0.08)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))

    # 1-sigma band
    fig.add_trace(go.Scatter(
        x=np.concatenate([dates_range_ts, dates_range_ts[::-1]]),
        y=np.concatenate([band_1hi, band_1lo[::-1]]),
        fill="toself", fillcolor="rgba(46,125,50,0.12)",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))

    # Trend line
    fig.add_trace(go.Scatter(
        x=dates_range_ts, y=trend,
        mode="lines", name="Exponential trend",
        line=dict(color="#2e7d32", width=2), hoverinfo="skip",
    ))

    # Error bars
    for i in range(len(res_df)):
        lo = ci_lo[i] if ci_lo[i] is not None and not np.isnan(ci_lo[i]) else vals[i]
        hi = ci_hi[i] if ci_hi[i] is not None and not np.isnan(ci_hi[i]) else vals[i]
        d = pd.Timestamp(dates[i])
        fig.add_trace(go.Scatter(
            x=[d, d], y=[lo, hi],
            mode="lines", line=dict(color="#2e7d32", width=2),
            showlegend=False, hoverinfo="skip",
        ))

    # Points
    point_dates = [pd.Timestamp(d) for d in dates]
    hover_text = [
        f"<b>{SHORT_NAMES.get(labels[i], labels[i])}</b><br>"
        f"{metric_label}: {fmt_time(vals[i])}<br>"
        f"Released: {pd.Timestamp(dates[i]).strftime('%Y-%m-%d')}"
        for i in range(len(res_df))
    ]

    fig.add_trace(go.Scatter(
        x=point_dates, y=vals,
        mode="markers+text",
        marker=dict(color="#2e7d32", size=9, line=dict(width=1, color="white")),
        text=[SHORT_NAMES.get(l, l) for l in labels],
        textposition="top center",
        textfont=dict(size=9, color="#333"),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_text,
        name="Models",
    ))

    ytick_minutes = [0.5, 1, 2, 4, 8, 15, 30, 60, 2*60, 4*60, 8*60, 16*60, 32*60, 64*60]
    ytick_labels = []
    for m in ytick_minutes:
        if m < 1:
            ytick_labels.append(f"{m*60:.0f} sec")
        elif m < 60:
            ytick_labels.append(f"{m:.0f} min")
        else:
            ytick_labels.append(f"{m/60:.0f} hr")

    stats_parts = []
    if avg_cv_brier is not None:
        stats_parts.append(f"Avg CV-Brier: {avg_cv_brier:.4f}")
    if avg_cv_log_loss is not None:
        stats_parts.append(f"Avg CV-LogLoss: {avg_cv_log_loss:.3f}")
    stats_str = (" | " + " | ".join(stats_parts)) if stats_parts else ""

    fig.update_layout(
        title=dict(
            text=(
                f"Length of tasks AI agents have been able to complete autonomously<br>"
                f"<span style='font-size:12px;color:#666'>"
                f"{model_name_str} fit - {metric_label} - "
                f"Doubling time: {doubling_time_days:.0f} days - "
                f"R\u00b2 = {r_value**2:.2f}{stats_str}</span>"
            ),
            x=0.5,
        ),
        xaxis=dict(title="Model release date"),
        yaxis=dict(
            title=f"Task time ({metric_label})",
            type="log",
            tickvals=ytick_minutes,
            ticktext=ytick_labels,
            range=[np.log10(max(min(vals[vals > 0]), 0.1) * 0.3), np.log10(max(vals) * 5)],
        ),
        template="plotly_white",
        width=1100, height=650,
        showlegend=False,
        hovermode="closest",
    )

    return fig, doubling_time_days, r_value ** 2


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_ablation(
    model_factory,
    output_dir,
    runs_path,
    bench_path,
    n_boot=200,
    seed=42,
):
    """Run full analysis for one curve fitting model. Returns summary dict."""
    import yaml

    os.makedirs(output_dir, exist_ok=True)

    runs = pd.read_json(runs_path, lines=True)
    runs = runs[runs["alias"] != "human"]

    with open(bench_path) as f:
        bench = yaml.safe_load(f)

    bench_dates = {}
    for key, info in bench["results"].items():
        if info.get("release_date"):
            bench_dates[key] = pd.to_datetime(info["release_date"])

    results = []
    per_model_dir = os.path.join(output_dir, "per_model")
    os.makedirs(per_model_dir, exist_ok=True)

    aliases = sorted(runs["alias"].unique())

    for alias in aliases:
        bench_key = ALIAS_TO_BENCH.get(alias)
        if bench_key is None or bench_key not in bench_dates:
            print(f"  Skipping {alias} (no release date)")
            continue

        release_date = bench_dates[bench_key]
        df_m = runs[runs["alias"] == alias].copy()
        print(f"  Fitting {alias} ({len(df_m)} runs)...")

        fig_fit, model_results, x_grid, y_full = plot_per_model_fit(
            alias, df_m, model_factory, n_boot=n_boot, seed=seed
        )

        safe_name = alias.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "_")
        pio.write_html(fig_fit, os.path.join(per_model_dir, f"{safe_name}_fit.html"), include_plotlyjs="cdn")

        row = {"alias": alias, "release_date": release_date}
        row.update(model_results)
        results.append(row)

        print(f"    CV-Brier={model_results['cv_brier']:.4f}  "
              f"CV-LogLoss={model_results['cv_log_loss']:.3f}  "
              f"InSample-Brier={model_results['insample_brier']:.4f}  "
              f"boot={model_results['boot_method']}")

    res_df = pd.DataFrame(results)

    avg_cv_brier = res_df["cv_brier"].mean()
    avg_cv_log_loss = res_df["cv_log_loss"].mean()
    avg_is_brier = res_df["insample_brier"].mean()

    # Trend charts: p50 and G[T]
    trend_results = {}
    for metric_label, point_col, lo_col, hi_col in [
        ("p50", "p50_minutes", "p50_ci_lo", "p50_ci_hi"),
        ("G[T]", "integral_minutes", "integral_ci_lo", "integral_ci_hi"),
    ]:
        fig_trend, doubling_days, r_sq = plot_metr_trend(
            res_df, model_factory.name,
            point_col=point_col, ci_lo_col=lo_col, ci_hi_col=hi_col,
            metric_label=metric_label,
            avg_cv_brier=avg_cv_brier, avg_cv_log_loss=avg_cv_log_loss,
        )
        safe_metric = metric_label.replace("[", "").replace("]", "")
        pio.write_html(
            fig_trend,
            os.path.join(output_dir, f"metr_trend_{safe_metric}.html"),
            include_plotlyjs="cdn",
        )
        trend_results[metric_label] = {
            "doubling_time_days": round(doubling_days, 1),
            "r_squared": round(r_sq, 4),
        }

    # Summary
    summary = {
        "model_name": model_factory.name,
        "model_params": model_factory.params,
        "bootstrap_type": getattr(model_factory, "bootstrap_type", "standard"),
        "fit_quality": {
            "avg_cv_brier": round(avg_cv_brier, 6),
            "avg_cv_log_loss": round(avg_cv_log_loss, 6),
            "avg_insample_brier": round(avg_is_brier, 6),
        },
        "trends": trend_results,
        "n_boot": n_boot,
        "n_models": len(res_df),
        "per_model_results": [],
    }
    for _, row in res_df.iterrows():
        entry = {
            "alias": row["alias"],
            "release_date": str(row["release_date"].date()) if pd.notna(row["release_date"]) else None,
            "cv_brier": row["cv_brier"],
            "cv_log_loss": row["cv_log_loss"],
            "insample_brier": row["insample_brier"],
            "p50_minutes": round(row["p50_minutes"], 2) if row["p50_minutes"] else None,
            "p50_ci_lo": round(row["p50_ci_lo"], 2) if row["p50_ci_lo"] else None,
            "p50_ci_hi": round(row["p50_ci_hi"], 2) if row["p50_ci_hi"] else None,
            "integral_minutes": round(row["integral_minutes"], 2),
            "integral_ci_lo": round(row["integral_ci_lo"], 2) if row["integral_ci_lo"] else None,
            "integral_ci_hi": round(row["integral_ci_hi"], 2) if row["integral_ci_hi"] else None,
            "boot_method": row["boot_method"],
        }
        summary["per_model_results"].append(entry)

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Done! {model_factory.name}")
    print(f"  CV-Brier={avg_cv_brier:.4f}  CV-LogLoss={avg_cv_log_loss:.3f}  "
          f"InSample-Brier={avg_is_brier:.4f}")
    for k, v in trend_results.items():
        print(f"    {k}: doubling={v['doubling_time_days']:.0f}d, R²={v['r_squared']:.4f}")
    print(f"  Outputs in: {output_dir}")

    return summary
