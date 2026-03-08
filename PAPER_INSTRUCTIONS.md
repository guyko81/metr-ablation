# Paper Instructions

## Tone: Constructive Improvement, Not Critique

This is NOT a takedown of METR. Their benchmark is valuable and their data collection is
excellent. The paper proposes concrete, drop-in improvements to three specific methodology
choices. Frame everything as "here's what we found works better and why" rather than
"here's what METR got wrong."

## Title Direction

Something like:
- "Improving Curve Fitting for AI Capability Time Horizons"
- "Better Estimates of AI Agent Capability Growth Rates"

Avoid words like "mistakes", "flaws", "problems" in the title.

## Structure

### 1. Introduction (1 page)
- METR's time horizon benchmark is the most important tracking effort for AI agent capabilities
- The headline number (doubling time ~4 months) informs policy and safety decisions
- Small methodology choices in the pipeline can materially affect this number
- We propose three improvements, validate them with cross-validation, and show the
  headline conclusion (doubling time ~120-130 days) is robust to methodology

Key message: "The good news is the trend is real. The improvements make the estimate
more defensible, not radically different."

### 2. Background (0.5 page)
- Brief description of METR's pipeline: binary outcomes, weighted by task, curve fitting,
  threshold crossing, trend regression
- Cite METR's paper directly

### 3. Three Improvements (core of the paper, ~4 pages)

#### 3.1 Curve Fitting: Isotonic Regression vs Logistic

**The issue**: Logistic regression assumes a symmetric sigmoid. Real success-vs-difficulty
curves are often asymmetric (quick saturation on easy tasks, slow decay on hard ones).

**The fix**: Isotonic regression (monotone decreasing) makes no parametric assumption.
Smoothed isotonic adds a Gaussian kernel in log2-space for continuity.

**Evidence**:
- 5-fold cross-validated Brier score: isotonic 0.1115 vs logistic 0.1212
- 5-fold cross-validated log-loss: smoothed isotonic 0.363 vs logistic 0.376
- The advantage holds OUT OF SAMPLE, not just in-sample

**Important nuance**: The in-sample Brier gap (0.1085 vs 0.1204) is larger than the CV gap
(0.1115 vs 0.1212). This confirms isotonic overfits slightly more than logistic, but
the improvement survives cross-validation. Report BOTH numbers and discuss this.

**Bootstrap fix**: Standard bootstrap is inconsistent for isotonic regression (converges
at n^{1/3}, not n^{1/2}). Use m-out-of-n bootstrap with m = n^{2/3}.
Cite: Kosorok (2008), Sen & Xu (2015), Leger & MacGibbon (2006).

#### 3.2 Summary Metric: G[T] Integral vs P50 Crossing

**The issue**: The 50% crossing point is a single-point statistic that's extremely sensitive
to local curve shape near 50%. For step-function curves (isotonic), it can jump discontinuously.

**The fix**: G[T] = geometric mean capability time. Treat q(u) = 1 - p(u) as a CDF in
log2-space, compute E[U] = u_min + integral(p(u) du), return 2^{E[U]}.

**Evidence**:
- G[T] R-squared: 0.916-0.921 across all three methods (nearly identical)
- P50 R-squared: 0.811-0.916 (swings wildly by method)
- G[T] doubling time: 128-130 days (stable)
- P50 doubling time: 118-121 days (varies with method)

**Why it works**: The integral uses the entire curve shape, averaging over the bulk where
all methods agree. P50 depends on a single point where methods diverge most.

**Acknowledge the limitation**: G[T]'s stability could be seen as insensitivity to tails.
But for a headline trend metric, stability across reasonable methodology choices is a
feature — it means the doubling time estimate is robust.

#### 3.3 This section is optional / future work: Trend Regression

Mention briefly that linear regression on log(metric) vs date assumes homoscedastic
noise and perfect exponential growth. With only 18 data points, more sophisticated
methods (Bayesian regression, NGBoost) give distributional uncertainty but are hard
to validate. Note this as future work, not a current recommendation.

### 4. Results Summary (1 page)

Present the comparison table:

| Method           | CV-Brier | CV-LogLoss | p50 Dbl | p50 R² | G[T] Dbl | G[T] R² |
|------------------|----------|------------|---------|--------|----------|---------|
| Logistic (METR)  | 0.1212   | 0.376      | 121d    | 0.916  | 128d     | 0.916   |
| Isotonic         | 0.1115   | 0.368      | 118d    | 0.811  | 130d     | 0.921   |
| Smoothed Isotonic| 0.1167   | 0.363      | 121d    | 0.848  | 130d     | 0.921   |

Key takeaways:
1. Isotonic wins on CV-Brier, smoothed isotonic wins on CV-LogLoss
2. G[T] makes the doubling time estimate robust to curve fitting choice
3. The doubling time is approximately 4 months regardless of methodology
4. Recommendation: use smoothed isotonic + G[T] for the most defensible estimate

### 5. Interactive Demo (0.5 page)
- Link to GitHub Pages viewer where readers can inspect every per-model fit
- This is the repo — all code is reproducible with `pip install -r requirements.txt && python run.py`

### 6. Discussion (0.5 page)
- The trend is real and robust
- Methodology improvements make the estimate more defensible for policy use
- Future work: trend regression, weighting scheme sensitivity, leave-one-model-out

## Figures to Include
1. Side-by-side: logistic vs smoothed isotonic fit on ONE model (pick o3 or Claude Opus 4.5
   where the asymmetry is most visible)
2. The G[T] trend chart (smoothed isotonic) — this is the "money figure"
3. Table of results (above)
4. Optional: p50 trend chart comparison showing instability across methods

## What NOT to Include
- The full 7-method ablation from v1 (keep it focused on three)
- NGBoost curve fitting (interesting but tangential)
- Extensive mathematical derivations (keep it practical)
- The old paper's tone of "METR made mistakes" — reframe as improvements

## References
- METR Time Horizon paper (cite their specific version/URL)
- Barlow et al. (1972) — isotonic regression foundations
- Kosorok (2008) — bootstrap inconsistency for isotonic estimators
- Sen & Xu (2015) — m-out-of-n bootstrap for shape-constrained estimators
- Leger & MacGibbon (2006) — smoothed bootstrap alternatives
- Robertson, Wright & Dykstra (1988) — Order Restricted Statistical Inference

## Repo Structure for GitHub
```
README.md           — overview + "how to run" + link to GitHub Pages demo
requirements.txt    — Python dependencies
run.py              — reproduce everything
curve_models.py     — the three model classes
pipeline.py         — fitting, CV, bootstrap, plotting
data/
  runs.jsonl        — METR's raw run data
  benchmark_results_1_1.yaml — model metadata + release dates
logistic/           — pre-computed outputs
isotonic/           — pre-computed outputs
smoothed_isotonic/  — pre-computed outputs
viewer.html         — interactive comparison (serve via GitHub Pages)
all_results.json    — summary of all results
```

The viewer.html + output folders should work directly from GitHub Pages with no build step.
