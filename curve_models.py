"""
Curve fitting models for ablation v2: logistic vs isotonic vs smoothed isotonic.

Each model is a callable class with:
    .fit(X_log2, y_binary, weights) -> self
    .predict(x_grid_log2) -> p_success array
    .name -> str
    .params -> dict
    .bootstrap_type -> "standard" | "m_out_of_n"
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class LogisticModel:
    """Sklearn LogisticRegression on log2(human_minutes) -> binary score.

    METR's original approach. C = 1/regularization.
    """

    name = "logistic"
    bootstrap_type = "standard"

    def __init__(self, regularization=0.1):
        self.regularization = regularization
        self.params = {"regularization": regularization}
        self._model = None

    def fit(self, X_log2, y, weights):
        y_int = y.astype(int)
        self._model = LogisticRegression(
            C=1.0 / self.regularization,
            solver="lbfgs",
            max_iter=1000,
        )
        self._model.fit(X_log2.reshape(-1, 1), y_int, sample_weight=weights)
        return self

    def predict(self, x_grid_log2):
        return self._model.predict_proba(x_grid_log2.reshape(-1, 1))[:, 1]


class IsotonicModel:
    """Weighted isotonic regression (decreasing) on log2(human_minutes) -> score."""

    name = "isotonic"
    bootstrap_type = "m_out_of_n"

    def __init__(self):
        self.params = {}
        self._iso = None

    def fit(self, X_log2, y, weights):
        self._iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        self._iso.fit(X_log2, y, sample_weight=weights)
        return self

    def predict(self, x_grid_log2):
        return self._iso.predict(x_grid_log2)


class SmoothedIsotonicModel:
    """Isotonic regression + Gaussian kernel smooth.

    Fits isotonic (decreasing), then applies Gaussian smooth to the
    step-function output. sigma is in log2-space units (not grid-index).
    """

    name = "smoothed_isotonic"
    bootstrap_type = "m_out_of_n"

    def __init__(self, sigma_log2=0.5):
        self.sigma_log2 = sigma_log2
        self.params = {"sigma_log2": sigma_log2}
        self._iso = None
        self._x_grid = None
        self._smoothed = None

    def fit(self, X_log2, y, weights):
        self._iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        self._iso.fit(X_log2, y, sample_weight=weights)
        # Build fine grid and smooth
        self._x_grid = np.linspace(X_log2.min() - 0.5, X_log2.max() + 0.5, 1000)
        dx = self._x_grid[1] - self._x_grid[0]  # grid spacing in log2 units
        sigma_grid = self.sigma_log2 / dx  # convert log2 sigma to grid-index sigma
        iso_pred = self._iso.predict(self._x_grid)
        self._smoothed = gaussian_filter1d(iso_pred, sigma=sigma_grid)
        self._smoothed = np.clip(self._smoothed, 0.0, 1.0)
        return self

    def predict(self, x_grid_log2):
        return np.interp(x_grid_log2, self._x_grid, self._smoothed)
