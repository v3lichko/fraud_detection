import numpy as np
from sklearn.metrics import confusion_matrix

from fraud_detection.evaluate import find_best_threshold_by_cost


def _brute_force_best_threshold(y_true, y_proba, cost_fp, cost_fn):
    best_threshold, best_cost = 0.5, np.inf
    for t in np.unique(y_proba):
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = fp * cost_fp + fn * cost_fn
        if cost < best_cost:
            best_cost, best_threshold = cost, t
    return best_threshold, best_cost


def test_vectorized_threshold_matches_brute_force():
    rng = np.random.default_rng(0)
    y_true = rng.choice([0, 1], size=300, p=[0.9, 0.1])
    y_proba = np.clip(y_true * 0.5 + rng.normal(0.3, 0.25, size=300), 0, 1)

    fast_threshold = find_best_threshold_by_cost(y_true, y_proba, cost_fp=1.0, cost_fn=50.0)
    slow_threshold, slow_cost = _brute_force_best_threshold(y_true, y_proba, 1.0, 50.0)

    def cost_at(t):
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return fp * 1.0 + fn * 50.0

    assert cost_at(fast_threshold) == cost_at(slow_threshold) == slow_cost


def test_threshold_is_within_probability_range():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7])
    threshold = find_best_threshold_by_cost(y_true, y_proba)
    assert y_proba.min() <= threshold <= y_proba.max()
