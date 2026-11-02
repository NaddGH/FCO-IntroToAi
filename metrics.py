"""Classification metrics implemented with NumPy/pandas (no scikit-learn needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def confusion_counts(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return float((y_true == y_pred).mean())


def precision_recall_f1(y_true, y_pred) -> tuple[float, float, float]:
    c = confusion_counts(y_true, y_pred)
    prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 0.0
    rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def roc_auc(y_true, score) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney U) identity.

    Ties receive average ranks. Returns NaN if only one class is present.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def cross_entropy(y_true, p, eps: float = 1e-15) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def full_report(y_true, y_pred, score, proba=None) -> dict:
    """Bundle of metrics. ``score`` is any ranking score; ``proba`` (optional) is P(y=1)."""
    prec, rec, f1 = precision_recall_f1(y_true, y_pred)
    out = {
        "accuracy": accuracy(y_true, y_pred),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": roc_auc(y_true, score),
        "log_loss": cross_entropy(y_true, proba) if proba is not None else float("nan"),
    }
    out.update(confusion_counts(y_true, y_pred))
    return out
