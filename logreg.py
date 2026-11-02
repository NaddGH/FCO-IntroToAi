"""Logistic regression implemented from scratch with NumPy.

Model
-----
For a feature vector x in R^d, with weights w in R^d and bias b,

    z = w^T x + b,        p(y = 1 | x) = sigmoid(z) = 1 / (1 + exp(-z)).

Loss (mean binary cross-entropy, optional L2 penalty on w only)

    J(w, b) = (1/n) * sum_i [ log(1 + exp(z_i)) - y_i z_i ] + (lambda / 2) * ||w||^2

Gradient

    grad_w = (1/n) X^T (p - y) + lambda * w,       grad_b = (1/n) sum_i (p_i - y_i)

Batch gradient descent update with learning rate eta:   theta <- theta - eta * grad J(theta).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# Numerically stable building blocks
# --------------------------------------------------------------------------- #
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function (no overflow for large |z|)."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def log_loss_from_logits(y: np.ndarray, z: np.ndarray) -> float:
    """Mean binary cross-entropy computed directly from logits z.

    Uses log(1 + e^z) - y z = logaddexp(0, z) - y z, which never overflows.
    """
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def add_bias_column(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones so that theta = [b, w_1, ..., w_d]."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def smoothness_constant(X: np.ndarray) -> float:
    """Upper bound L on the Lipschitz constant of grad J (unregularised).

    The Hessian of the mean logistic loss is X_b^T S X_b / n with
    S = diag(p_i (1 - p_i)) <= (1/4) I, hence  L = lambda_max(X_b^T X_b / n) / 4.
    Gradient descent with a constant step eta < 2 / L is guaranteed to decrease
    the loss at every iteration (descent lemma).
    """
    Xb = add_bias_column(X)
    gram = Xb.T @ Xb / Xb.shape[0]
    return float(np.linalg.eigvalsh(gram).max() / 4.0)


# --------------------------------------------------------------------------- #
# Gradient-descent logistic regression
# --------------------------------------------------------------------------- #
@dataclass
class LogisticRegressionGD:
    """Binary logistic regression trained by full-batch gradient descent.

    Parameters
    ----------
    lr : float
        Learning rate (step size) eta.
    n_iter : int
        Maximum number of gradient steps.
    l2 : float
        L2 penalty strength lambda on the weights (not on the bias). 0 = none.
    tol : float or None
        If given, stop early when the Euclidean norm of the gradient is < tol.
        Use None (default) to always run exactly ``n_iter`` steps, which is
        what the learning-rate sweep needs for a fair comparison.
    """

    lr: float = 0.1
    n_iter: int = 2000
    l2: float = 0.0
    tol: float | None = None

    # learned state
    coef_: np.ndarray = field(default=None, init=False, repr=False)   # shape (d,)
    intercept_: float = field(default=0.0, init=False, repr=False)
    loss_history_: np.ndarray = field(default=None, init=False, repr=False)
    grad_norm_history_: np.ndarray = field(default=None, init=False, repr=False)
    n_iter_run_: int = field(default=0, init=False, repr=False)

    def _objective(self, z: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        return log_loss_from_logits(y, z) + 0.5 * self.l2 * float(w @ w)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have different numbers of rows")
        if not np.isin(y, (0.0, 1.0)).all():
            raise ValueError("y must contain only 0/1 labels")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values")
        if self.lr <= 0:
            raise ValueError("learning rate must be positive")

        n, d = X.shape
        Xb = add_bias_column(X)
        theta = np.zeros(d + 1)                    # theta = [b, w]
        penalty = np.r_[0.0, np.full(d, self.l2)]  # no penalty on the bias

        losses = np.empty(self.n_iter + 1)
        gnorms = np.empty(self.n_iter + 1)
        steps_done = 0

        for k in range(self.n_iter + 1):
            z = Xb @ theta
            p = sigmoid(z)
            grad = Xb.T @ (p - y) / n + penalty * theta
            losses[k] = self._objective(z, y, theta[1:])
            gnorms[k] = np.linalg.norm(grad)
            if k == self.n_iter:
                break
            if self.tol is not None and gnorms[k] < self.tol:
                break
            theta = theta - self.lr * grad
            steps_done += 1

        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:].copy()
        self.loss_history_ = losses[: steps_done + 1]
        self.grad_norm_history_ = gnorms[: steps_done + 1]
        self.n_iter_run_ = steps_done
        return self

    # ---- prediction helpers -------------------------------------------------
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(y = 1 | x) as a 1-D array."""
        return sigmoid(self.decision_function(X))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Mean cross-entropy (without the L2 term) on the given data."""
        return log_loss_from_logits(np.asarray(y, dtype=float), self.decision_function(X))


# --------------------------------------------------------------------------- #
# Reference optimum (Newton-Raphson / IRLS)
# --------------------------------------------------------------------------- #
def fit_newton(X: np.ndarray, y: np.ndarray, l2: float = 0.0, max_iter: int = 100,
               tol: float = 1e-10) -> tuple[np.ndarray, float, float]:
    """Solve the same problem with Newton's method (iteratively reweighted LS).

    Returns (coef, intercept, optimal_loss). Used only as a *reference* to
    (i) check that gradient descent converges to the right answer and
    (ii) measure the excess loss J(theta_k) - J(theta*) along the GD path.
    A tiny ridge (1e-10) is added to the Hessian purely for numerical safety.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, d = X.shape
    Xb = add_bias_column(X)
    penalty = np.r_[0.0, np.full(d, l2)]
    theta = np.zeros(d + 1)
    for _ in range(max_iter):
        z = Xb @ theta
        p = sigmoid(z)
        grad = Xb.T @ (p - y) / n + penalty * theta
        s = p * (1.0 - p)
        H = (Xb * s[:, None]).T @ Xb / n + np.diag(penalty) + 1e-10 * np.eye(d + 1)
        step = np.linalg.solve(H, grad)
        theta = theta - step
        if np.linalg.norm(step) < tol:
            break
    z = Xb @ theta
    loss = log_loss_from_logits(y, z) + 0.5 * l2 * float(theta[1:] @ theta[1:])
    return theta[1:].copy(), float(theta[0]), float(loss)
