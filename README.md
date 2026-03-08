# Improving Curve Fitting for AI Capability Time Horizons

Three drop-in improvements to [METR's Time Horizon benchmark](https://metr.org/blog/2025-03-19-measuring-ai-agent-time-horizons/) methodology, validated with 5-fold cross-validation on 18 frontier AI models.

**[Interactive Viewer](https://guyko81.github.io/metr-ablation/)** | **[Paper (arXiv)](https://arxiv.org/abs/[ARXIV-ID])**

## Key Result

The headline finding — AI agent capabilities double roughly every **~4 months** — is **robust** across all methodology choices we tested. Our improvements make the estimate more defensible, not radically different.

| Method | CV-Brier | CV-LogLoss | p50 Doubling | p50 R² | G[T] Doubling | G[T] R² |
|--------|----------|------------|-------------|--------|--------------|---------|
| Logistic (METR) | 0.1212 | 0.376 | 121 d | 0.916 | 128 d | 0.916 |
| Isotonic | **0.1115** | 0.368 | 118 d | 0.811 | 130 d | **0.921** |
| Smoothed Isotonic | 0.1167 | **0.363** | 121 d | 0.848 | 130 d | **0.921** |

**Recommendation:** Use smoothed isotonic regression + G[T] integral metric for the most defensible estimate (~130 day doubling time, R² = 0.921).

## The Three Improvements

1. **Curve fitting:** Replace logistic regression with smoothed isotonic regression — avoids the symmetric-sigmoid assumption, improves cross-validated log-loss by 3.5%.

2. **Summary metric:** Replace p50 threshold crossing with G[T] (geometric mean capability time) — an integral metric that uses the entire fitted curve instead of depending on a single point, yielding R² = 0.921 across all methods.

3. **Bootstrap:** Use m-out-of-n bootstrap (m = n^{2/3}) for isotonic estimators — theoretically consistent for shape-constrained inference (Kosorok 2008, Sen & Xu 2015).

## Quick Start

```bash
pip install -r requirements.txt
python run.py
```

This regenerates all results: 54 per-model fit plots, 6 trend charts, comparison tables, and the interactive viewer.

## Repository Structure

```
paper.tex               — LaTeX source of the paper
index.html              — GitHub Pages landing page
viewer.html             — Interactive comparison viewer
run.py                  — Main execution script (reproduce everything)
curve_models.py         — Three model classes (logistic, isotonic, smoothed isotonic)
pipeline.py             — Fitting, cross-validation, bootstrap, plotting
requirements.txt        — Python dependencies
data/
  runs.jsonl            — METR benchmark run data
  benchmark_results_1_1.yaml — Model metadata and release dates
all_results.json        — Comprehensive results summary
logistic/               — Pre-computed logistic regression outputs
isotonic/               — Pre-computed isotonic regression outputs
smoothed_isotonic/      — Pre-computed smoothed isotonic outputs
```

## Models Evaluated

18 frontier models from GPT-4 (Mar 2023) through GPT-5.2 (Dec 2025), including Claude 3–4.5, o1/o3, and Gemini 3 Pro. Approximately 1,000–1,400 task runs evaluated per model.

## Citation

```bibtex
@article{gulyas2026improving,
  title={Improving Curve Fitting for AI Capability Time Horizons: Three Drop-In Improvements to METR's Methodology},
  author={Gulyas, Gabor},
  journal={arXiv preprint arXiv:[ARXIV-ID]},
  year={2026}
}
```

## License

Data sourced from [METR's Time Horizon Benchmark v1.1](https://metr.org/blog/2025-03-19-measuring-ai-agent-time-horizons/).
