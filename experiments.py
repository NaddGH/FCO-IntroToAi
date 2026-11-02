"""Experiments: learning-rate sweep, model selection, baselines and verification."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data_prep import FEATURE_SETS, TARGETS, Standardizer
from .logreg import LogisticRegressionGD, fit_newton, smoothness_constant
from .metrics import full_report

LR_GRID = (1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
N_ITER = 2000
EPS_EXCESS = 1e-3          # "converged" = excess loss J(theta_k) - J(theta*) <= EPS_EXCESS
SPLITS = ("train", "val", "test")


@dataclass
class SweepResult:
    target: str
    fset: str
    features: list
    L: float                       # smoothness constant of the training loss
    bound: float                   # 2 / L  (sufficient condition for monotone descent)
    loss_star: float               # optimal training loss (Newton)
    theta_star: np.ndarray         # [intercept, coef...] from Newton
    table: pd.DataFrame            # one row per learning rate
    excess: dict                   # lr -> array of J(theta_k) - J(theta*)
    best_lr: float
    model: LogisticRegressionGD
    scaler: Standardizer
    data: dict                     # split -> (X, y)
    metrics: dict = field(default_factory=dict)   # split -> metric dict (selected model)
    verification: dict = field(default_factory=dict)


def _matrices(df: pd.DataFrame, target: str, fset: str):
    feats = FEATURE_SETS[fset]
    scaler = Standardizer(feats).fit(df[df["split"] == "train"])
    data = {}
    for s in SPLITS:
        part = df[df["split"] == s]
        data[s] = (scaler.transform(part), part[target].to_numpy(dtype=float))
    return feats, scaler, data


def run_sweep(df: pd.DataFrame, target: str, fset: str,
              lr_grid=LR_GRID, n_iter: int = N_ITER) -> SweepResult:
    feats, scaler, data = _matrices(df, target, fset)
    Xtr, ytr = data["train"]
    L = smoothness_constant(Xtr)
    w_star, b_star, loss_star = fit_newton(Xtr, ytr)
    theta_star = np.r_[b_star, w_star]

    rows, excess = [], {}
    for lr in lr_grid:
        m = LogisticRegressionGD(lr=lr, n_iter=n_iter).fit(Xtr, ytr)
        hist = m.loss_history_
        ex = np.maximum(hist - loss_star, 0.0)
        excess[lr] = ex
        hit = np.flatnonzero(ex <= EPS_EXCESS)
        theta = np.r_[m.intercept_, m.coef_]
        row = {
            "lr": lr,
            "final_train_loss": float(hist[-1]),
            "final_excess": float(ex[-1]),
            "monotone": bool(np.all(np.diff(hist) <= 1e-12)),
            "iters_to_eps": int(hit[0]) if hit.size else np.nan,
            "max_abs_dtheta": float(np.max(np.abs(theta - theta_star))),
        }
        for s in SPLITS:
            X, y = data[s]
            row[f"{s}_acc"] = float((m.predict(X) == y).mean())
            row[f"{s}_loss"] = m.loss(X, y)
        rows.append(row)
    table = pd.DataFrame(rows)

    # model selection on the VALIDATION loss (ties -> smaller learning rate)
    best_lr = float(table.loc[table["val_loss"].idxmin(), "lr"])
    model = LogisticRegressionGD(lr=best_lr, n_iter=n_iter).fit(Xtr, ytr)

    res = SweepResult(target, fset, feats, L, 2.0 / L, loss_star, theta_star, table,
                      excess, best_lr, model, scaler, data)
    for s in SPLITS:
        X, y = data[s]
        proba = model.predict_proba(X)
        res.metrics[s] = full_report(y, (proba >= 0.5).astype(int), proba, proba)
    res.verification = verify_against_references(res)
    return res


def verify_against_references(res: SweepResult) -> dict:
    """Compare the GD solution with Newton's method and (if installed) scikit-learn."""
    theta = np.r_[res.model.intercept_, res.model.coef_]
    out = {
        "gd_vs_newton_max_abs_diff": float(np.max(np.abs(theta - res.theta_star))),
        "gd_excess_train_loss": float(res.model.loss_history_[-1] - res.loss_star),
        "sklearn_available": False,
    }
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:                                   # scikit-learn is optional
        return out
    Xtr, ytr = res.data["train"]
    Xte, yte = res.data["test"]
    skl = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kwargs in ({"C": np.inf}, {"penalty": None}, {"C": 1e12}):   # unregularised
            try:
                skl = LogisticRegression(max_iter=5000, tol=1e-10, **kwargs).fit(Xtr, ytr)
                break
            except Exception:
                skl = None
    if skl is None:
        return out
    skl_theta = np.r_[skl.intercept_[0], skl.coef_[0]]
    out.update({
        "sklearn_available": True,
        "gd_vs_sklearn_max_abs_diff": float(np.max(np.abs(theta - skl_theta))),
        "test_pred_agreement_gd_vs_sklearn": float(
            (res.model.predict(Xte) == skl.predict(Xte)).mean()),
        "sklearn_test_accuracy": float((skl.predict(Xte) == yte).mean()),
    })
    return out


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def baseline_metrics(df: pd.DataFrame, target: str) -> dict:
    """Two reference classifiers evaluated on the test split.

    majority : always predicts the most frequent class of the training split.
    rule     : predicts 'positive' iff the qualifying position is <= the target cut-off
               (top 10 for ``top10``, top 3 for ``podium``) - a no-learning heuristic.
    """
    cutoff = TARGETS[target]
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    y = test[target].to_numpy(dtype=float)
    majority_class = int(train[target].mean() >= 0.5)
    maj_pred = np.full(len(test), majority_class)
    maj = full_report(y, maj_pred, np.zeros(len(test)))
    rule_pred = (test["q_pos"].to_numpy() <= cutoff).astype(int)
    rule = full_report(y, rule_pred, -test["q_pos"].to_numpy())
    return {"majority": maj, "rule": rule, "majority_class": majority_class}


def per_qualifying_position(df: pd.DataFrame, target: str, results: dict,
                            max_pos: int = 20, min_count: int = 5) -> pd.DataFrame:
    """Empirical positive rate and mean predicted probability by qualifying position (test split)."""
    test = df[df["split"] == "test"].copy()
    out = test.groupby("q_pos")[target].agg(["mean", "size"]).rename(
        columns={"mean": "empirical", "size": "count"})
    for fset, res in results.items():
        X, _ = res.data["test"]
        test[f"pred_{fset}"] = res.model.predict_proba(X)
        out[f"pred_{fset}"] = test.groupby("q_pos")[f"pred_{fset}"].mean()
    out = out[(out["count"] >= min_count) & (out.index <= max_pos)]
    return out.reset_index()
