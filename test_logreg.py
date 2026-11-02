import numpy as np
import pytest

from f1logreg.logreg import (LogisticRegressionGD, fit_newton, log_loss_from_logits,
                             sigmoid, smoothness_constant)
from f1logreg.metrics import accuracy, roc_auc, precision_recall_f1


def _toy(n=400, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    true_w = np.array([1.5, -2.0, 0.5])
    p = 1 / (1 + np.exp(-(X @ true_w + 0.3)))
    y = (rng.random(n) < p).astype(float)
    return X, y


def test_sigmoid_is_stable_and_correct():
    z = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    s = sigmoid(z)
    assert np.all(np.isfinite(s))
    assert s[0] == pytest.approx(0.0, abs=1e-12) and s[-1] == pytest.approx(1.0)
    assert s[2] == pytest.approx(0.5)
    assert s[1] + s[3] == pytest.approx(1.0)


def test_loss_matches_textbook_formula():
    X, y = _toy()
    z = X @ np.array([0.2, -0.1, 0.4]) + 0.1
    p = sigmoid(z)
    textbook = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    assert log_loss_from_logits(y, z) == pytest.approx(textbook)


def test_gradient_matches_finite_differences():
    X, y = _toy(n=50)
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    theta = np.array([0.1, 0.3, -0.2, 0.05])
    analytic = Xb.T @ (sigmoid(Xb @ theta) - y) / n
    numeric = np.zeros_like(theta)
    h = 1e-6
    for j in range(len(theta)):
        e = np.zeros_like(theta); e[j] = h
        numeric[j] = (log_loss_from_logits(y, Xb @ (theta + e)) -
                      log_loss_from_logits(y, Xb @ (theta - e))) / (2 * h)
    assert np.allclose(analytic, numeric, atol=1e-7)


def test_gd_converges_to_newton_solution():
    X, y = _toy()
    L = smoothness_constant(X)
    m = LogisticRegressionGD(lr=1.0 / L, n_iter=20000).fit(X, y)
    w, b, loss_star = fit_newton(X, y)
    assert np.allclose(m.coef_, w, atol=1e-3)
    assert m.intercept_ == pytest.approx(b, abs=1e-3)
    assert m.loss_history_[-1] - loss_star < 1e-6


def test_loss_decreases_monotonically_below_2_over_L():
    X, y = _toy()
    L = smoothness_constant(X)
    for frac in (0.1, 0.5, 0.99):
        m = LogisticRegressionGD(lr=frac * 2.0 / L, n_iter=300).fit(X, y)
        assert np.all(np.diff(m.loss_history_) <= 1e-12), frac


def test_early_stopping_with_tolerance():
    X, y = _toy()
    m = LogisticRegressionGD(lr=1.0, n_iter=100000, tol=1e-6).fit(X, y)
    assert m.n_iter_run_ < 100000
    assert m.grad_norm_history_[-1] < 1e-6


def test_input_validation():
    X, y = _toy(n=20)
    with pytest.raises(ValueError):
        LogisticRegressionGD().fit(X, y + 1)             # labels not in {0,1}
    with pytest.raises(ValueError):
        LogisticRegressionGD(lr=-1).fit(X, y)
    X_bad = X.copy(); X_bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        LogisticRegressionGD().fit(X_bad, y)


def test_metrics():
    y = np.array([1, 1, 0, 0, 1, 0])
    pred = np.array([1, 0, 0, 1, 1, 0])
    assert accuracy(y, pred) == pytest.approx(4 / 6)
    p, r, f = precision_recall_f1(y, pred)
    assert p == pytest.approx(2 / 3) and r == pytest.approx(2 / 3) and f == pytest.approx(2 / 3)
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)
    assert roc_auc([0, 0, 1, 1], [0.4, 0.3, 0.2, 0.1]) == pytest.approx(0.0)
    assert roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
    assert np.isnan(roc_auc([1, 1], [0.1, 0.2]))
