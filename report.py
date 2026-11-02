"""Create figures and the LaTeX fragments (tables + macros) that the paper includes.

Every number that appears in the paper's results sections is written here from the
computed results, so the paper can never disagree with the code.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data_prep import FEATURE_LABELS  # noqa: E402
from .experiments import EPS_EXCESS, N_ITER  # noqa: E402

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3,
})


# --------------------------------------------------------------------------- #
# Formatting helpers (LaTeX-safe)
# --------------------------------------------------------------------------- #
def lr_math(lr: float) -> str:
    """Learning rate as a math-mode string WITHOUT surrounding dollar signs."""
    if lr < 0.01:
        exp = int(round(np.log10(lr)))
        if abs(lr - 10.0 ** exp) < 1e-12 * max(1.0, abs(lr)):
            return f"10^{{{exp}}}"
        return f"{lr:.0e}"
    return f"{lr:g}"


def lr_cell(lr: float) -> str:
    return f"${lr_math(lr)}$"


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}\\%"


def f3(x: float) -> str:
    return "--" if x is None or not np.isfinite(x) else f"{x:.3f}"


def sci(x: float) -> str:
    """Scientific notation for LaTeX, e.g. 3.2e-05 -> 3.2\\times10^{-5} (math mode content)."""
    if x == 0:
        return "0"
    mant, exp = f"{x:.1e}".split("e")
    return f"{mant}\\times 10^{{{int(exp)}}}"


def sci_cell(x: float) -> str:
    return f"${sci(x)}$" if x > 0 else "$0$"


def years_text(y0: int, y1: int) -> str:
    return f"{y0}" if y0 == y1 else f"{y0}--{y1}"


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save(fig, figdir: Path, name: str) -> None:
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / f"{name}.pdf")
    fig.savefig(figdir / f"{name}.png", dpi=200)
    plt.close(fig)


def fig_probability_by_qpos(by_q: pd.DataFrame, figdir: Path, target_label: str) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(by_q["q_pos"], by_q["empirical"], "ko-", ms=4, lw=1, label="Observed rate (test seasons)")
    if "pred_A" in by_q:
        ax.plot(by_q["q_pos"], by_q["pred_A"], "s--", color="tab:blue", ms=3.5, lw=1,
                label="Model A: mean predicted probability")
    if "pred_B" in by_q:
        ax.plot(by_q["q_pos"], by_q["pred_B"], "^--", color="tab:red", ms=3.5, lw=1,
                label="Model B: mean predicted probability")
    ax.set_xlabel("Qualifying position")
    ax.set_ylabel(f"P({target_label})")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(range(1, int(by_q["q_pos"].max()) + 1, 2))
    ax.legend(loc="upper right")
    _save(fig, figdir, "fig_prob_by_qpos")


def fig_loss_curves(res, figdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    cmap = plt.get_cmap("viridis")
    lrs = list(res.excess.keys())
    for i, lr in enumerate(lrs):
        ex = np.maximum(res.excess[lr], 1e-12)
        ax.loglog(np.arange(1, len(ex)), ex[1:], color=cmap(i / max(1, len(lrs) - 1)),
                  lw=1.1, label=f"$\\eta={lr_math(lr)}$")
    ax.axhline(EPS_EXCESS, color="gray", ls=":", lw=1)
    ax.text(1.2, EPS_EXCESS * 1.25, f"tolerance {EPS_EXCESS:g}", color="gray", fontsize=7)
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel("Excess training loss $J(\\theta_k)-J(\\theta^\\ast)$")
    ax.legend(ncol=2, loc="lower left")
    _save(fig, figdir, "fig_loss_curves")


def fig_accuracy_vs_lr(results: dict, baselines: dict, figdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), sharey=True)
    for ax, fset in zip(axes, ("A", "B")):
        res = results[fset]
        t = res.table
        ax.semilogx(t["lr"], t["train_acc"], "o-", ms=3.5, lw=1, label="Train")
        ax.semilogx(t["lr"], t["val_acc"], "s-", ms=3.5, lw=1, label="Validation")
        ax.semilogx(t["lr"], t["test_acc"], "^-", ms=3.5, lw=1, label="Test")
        ax.axvline(res.bound, color="gray", ls="--", lw=1)
        ax.text(res.bound * 1.08, 0.92, "$2/L$", color="gray", fontsize=8,
                transform=ax.get_xaxis_transform())
        ax.axhline(baselines["majority"]["accuracy"], color="k", ls=":", lw=1,
                   label="Majority class (test)")
        ax.set_title(f"Model {fset}: {', '.join(['qualifying'] if fset == 'A' else ['qualifying', 'pit stops'])}")
        ax.set_xlabel("Learning rate $\\eta$ (log scale)")
    axes[0].set_ylabel(f"Accuracy after {N_ITER} iterations")
    axes[0].legend(loc="center left")
    _save(fig, figdir, "fig_accuracy_vs_lr")


def fig_loss_vs_lr(results: dict, figdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for fset, style, color in (("A", "o--", "tab:blue"), ("B", "s-", "tab:red")):
        t = results[fset].table
        ax.semilogx(t["lr"], t["val_loss"], style, color=color, ms=3.5, lw=1, label=f"Model {fset} (validation)")
    ax.set_xlabel("Learning rate $\\eta$ (log scale)")
    ax.set_ylabel("Validation cross-entropy loss")
    ax.legend()
    _save(fig, figdir, "fig_val_loss_vs_lr")


def fig_confusion(res, figdir: Path) -> None:
    m = res.metrics["test"]
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    ax.set_xticks([0, 1], labels=["Pred. no", "Pred. yes"])
    ax.set_yticks([0, 1], labels=["Actual no", "Actual yes"])
    _save(fig, figdir, "fig_confusion_B")


# --------------------------------------------------------------------------- #
# LaTeX tables
# --------------------------------------------------------------------------- #
def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def table_dataset(df: pd.DataFrame, gen: Path) -> None:
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Split & Seasons & Races & Rows & Top-10 rate & Podium rate \\", r"\midrule"]
    for s, label in (("train", "Training"), ("val", "Validation"), ("test", "Test")):
        part = df[df["split"] == s]
        lines.append(f"{label} & {years_text(int(part['year'].min()), int(part['year'].max()))} & "
                     f"{part['raceid'].nunique():,} & {len(part):,} & "
                     f"{pct(part['top10'].mean())} & {pct(part['podium'].mean())} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(gen / "tab_dataset.tex", "\n".join(lines) + "\n")


def table_sweep(res, gen: Path, name: str) -> None:
    lines = [r"\begin{tabular}{rrcrrrr}", r"\toprule",
             r"$\eta$ & Final train loss & Monotone & Iters to tol. & Train acc. & Val. acc. & Test acc. \\",
             r"\midrule"]
    for _, r in res.table.iterrows():
        it = "not reached" if not np.isfinite(r["iters_to_eps"]) else f"{int(r['iters_to_eps'])}"
        mark = r"\textbf{" if r["lr"] == res.best_lr else ""
        end = "}" if mark else ""
        lines.append(f"{mark}{lr_cell(r['lr'])}{end} & {f3(r['final_train_loss'])} & "
                     f"{'yes' if r['monotone'] else 'no'} & {it} & "
                     f"{pct(r['train_acc'])} & {pct(r['val_acc'])} & {pct(r['test_acc'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(gen / name, "\n".join(lines) + "\n")


def _metric_row(label: str, m: dict, bold: bool = False) -> str:
    b, e = (r"\textbf{", "}") if bold else ("", "")
    ll = f3(m["log_loss"])
    return (f"{b}{label}{e} & {pct(m['accuracy'])} & {f3(m['precision'])} & {f3(m['recall'])} & "
            f"{f3(m['f1'])} & {f3(m['auc'])} & {ll} \\\\")


def table_metrics(results: dict, baselines: dict, gen: Path, name: str, rule_label: str) -> None:
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Model & Accuracy & Precision & Recall & F1 & AUC & Log-loss \\", r"\midrule",
             _metric_row("Majority class", baselines["majority"]),
             _metric_row(rule_label, baselines["rule"]),
             _metric_row("Logistic regression A (qualifying)", results["A"].metrics["test"]),
             _metric_row("Logistic regression B (qual. + pit stops)", results["B"].metrics["test"], bold=True),
             r"\bottomrule", r"\end{tabular}"]
    _write(gen / name, "\n".join(lines) + "\n")


def table_coefficients(res, gen: Path, name: str) -> None:
    lines = [r"\begin{tabular}{lrr}", r"\toprule",
             r"Feature (standardised) & Coefficient & Odds ratio per $+1$ SD \\", r"\midrule",
             f"Intercept & {res.model.intercept_:+.3f} & -- \\\\"]
    for f, w in zip(res.features, res.model.coef_):
        lines.append(f"{FEATURE_LABELS[f]} & {w:+.3f} & {np.exp(w):.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(gen / name, "\n".join(lines) + "\n")


def table_confusion(results: dict, gen: Path) -> None:
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Model & TP & FP & FN & TN \\", r"\midrule"]
    for fset, label in (("A", "A (qualifying)"), ("B", "B (qual. + pit stops)")):
        m = results[fset].metrics["test"]
        lines.append(f"{label} & {m['tp']:,} & {m['fp']:,} & {m['fn']:,} & {m['tn']:,} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(gen / "tab_confusion.tex", "\n".join(lines) + "\n")


def table_verification(results: dict, gen: Path) -> None:
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Model & $\max|\theta_{\mathrm{GD}}-\theta_{\mathrm{Newton}}|$ & "
             r"$\max|\theta_{\mathrm{GD}}-\theta_{\mathrm{sklearn}}|$ & Same test predictions \\",
             r"\midrule"]
    for fset in ("A", "B"):
        v = results[fset].verification
        skl = sci_cell(v["gd_vs_sklearn_max_abs_diff"]) if v.get("sklearn_available") else "n/a"
        agree = pct(v["test_pred_agreement_gd_vs_sklearn"], 2) if v.get("sklearn_available") else "n/a"
        lines.append(f"Model {fset} & {sci_cell(v['gd_vs_newton_max_abs_diff'])} & {skl} & {agree} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(gen / "tab_verification.tex", "\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Macros and data-driven sentences
# --------------------------------------------------------------------------- #
def _lr_list_math(lrs: list) -> str:
    items = [f"${lr_math(x)}$" for x in lrs]
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + (", and " if len(items) > 2 else " and ") + items[-1]


def observation_sentences(results: dict, podium: dict, baselines: dict, base_pod: dict) -> dict:
    """Sentences whose content depends on the numbers. Everything is computed, nothing hard-coded."""
    obs = {}
    A, B = results["A"], results["B"]

    # --- stability / monotonicity (model B) ------------------------------------
    t = B.table
    mono = list(t.loc[t["monotone"], "lr"])
    nonmono = list(t.loc[~t["monotone"], "lr"])
    if not nonmono:
        s = (f"The training loss decreased at every iteration for all {len(t)} learning rates tested "
             f"(up to $\\eta={lr_math(max(t['lr']))}$).")
    elif not mono:
        s = "The training loss was non-monotone for every learning rate tested."
    else:
        prefix = set(t.loc[t["lr"] <= max(mono), "lr"]) == set(mono)      # monotone set is an initial segment?
        head = (f"$\\eta\\le {lr_math(max(mono))}$" if prefix else _lr_list_math(mono))
        s = (f"The training loss decreased at every iteration for {head}, "
             f"whereas it was non-monotone for {_lr_list_math(nonmono)}.")
    above = [x for x in t["lr"] if x > B.bound]
    beyond = [x for x in mono if x > B.bound]
    below_nonmono = [x for x in nonmono if x <= B.bound]
    if below_nonmono:
        s += (" Some learning rates below the sufficient bound $2/L$ were non-monotone, which should not "
              "happen in exact arithmetic and points to a numerical problem.")
    elif not above:
        s += (f" All tested learning rates lie below the sufficient bound $2/L_B={B.bound:.2f}$.")
    elif beyond:
        s += (f" Monotone descent therefore persisted for learning rates above the sufficient bound "
              f"$2/L_B={B.bound:.2f}$ (up to $\\eta={lr_math(max(beyond))}$), which shows that the bound is "
              f"conservative: the true curvature $p(1-p)$ is smaller than the worst case $1/4$ used to define $L$.")
    else:
        s += (f" This is consistent with the theory: the guarantee only covers $\\eta<2/L_B={B.bound:.2f}$, "
              f"and every tested learning rate above that value was non-monotone.")
    obs["ObsStability"] = s

    # --- slow end -------------------------------------------------------------------
    t0 = t.iloc[0]
    obs["ObsSlow"] = (f"With the smallest step $\\eta={lr_math(t0['lr'])}$ the excess training loss after "
                      f"{N_ITER:,} iterations was {sci_cell(t0['final_excess'])} and the test accuracy was "
                      f"{pct(t0['test_acc'])}.")

    # --- fast end --------------------------------------------------------------------
    reached = t[np.isfinite(t["iters_to_eps"])]
    if len(reached):
        fastest = reached.loc[reached["iters_to_eps"].idxmin()]
        slowest = reached.loc[reached["iters_to_eps"].idxmax()]
        obs["ObsSpeed"] = (f"Among the learning rates that reached the tolerance {EPS_EXCESS:g}, the number of "
                           f"iterations needed ranged from {int(fastest['iters_to_eps'])} "
                           f"($\\eta={lr_math(fastest['lr'])}$) to {int(slowest['iters_to_eps'])} "
                           f"($\\eta={lr_math(slowest['lr'])}$).")
    else:
        obs["ObsSpeed"] = f"No learning rate reached the tolerance {EPS_EXCESS:g} within {N_ITER:,} iterations."

    # --- accuracy plateau -------------------------------------------------------------
    best_test = t["test_acc"].max()
    near = list(t.loc[t["test_acc"] >= best_test - 0.005, "lr"])
    worst = t.loc[t["test_acc"].idxmin()]
    obs["ObsPlateau"] = (f"Test accuracy was within 0.5 percentage points of its maximum "
                         f"({pct(best_test)}) for {_lr_list_math(near)}; the lowest test accuracy, "
                         f"{pct(worst['test_acc'])}, occurred at $\\eta={lr_math(worst['lr'])}$.")

    # --- A vs B ----------------------------------------------------------------------
    mA, mB = A.metrics["test"], B.metrics["test"]
    dacc = 100 * (mB["accuracy"] - mA["accuracy"])
    dauc = mB["auc"] - mA["auc"]
    if abs(dacc) < 0.05 and abs(dauc) < 0.0005:
        verdict = "Adding the pit-stop variables changed neither accuracy nor AUC."
    else:
        word = "improved" if dacc > 0 else "reduced"
        verdict = (f"Adding the pit-stop variables {word} test accuracy by {abs(dacc):.1f} percentage points "
                   f"({pct(mA['accuracy'])} to {pct(mB['accuracy'])}) and changed the AUC by "
                   f"{dauc:+.3f} ({f3(mA['auc'])} to {f3(mB['auc'])}).")
    obs["ObsAblation"] = verdict

    # --- baselines -------------------------------------------------------------------
    rule = baselines["rule"]
    obs["ObsBaseline"] = (f"Model B reached {pct(mB['accuracy'])} test accuracy against "
                          f"{pct(baselines['majority']['accuracy'])} for always predicting the majority class and "
                          f"{pct(rule['accuracy'])} for the rule ``qualified in the top ten''; "
                          f"the AUC values were {f3(mB['auc'])} and {f3(rule['auc'])}, respectively.")

    # --- coefficients -----------------------------------------------------------------
    phrase = {"q_pos": "qualifying position", "q1_gap_pct": "the Q1 gap to the fastest time",
              "n_stops": "the number of pit stops", "median_pit_s": "the median pit-stop time"}
    parts = [f"a one-standard-deviation increase in {phrase[f]} multiplies the odds of a top-10 finish by "
             f"{np.exp(w):.2f}" for f, w in zip(B.features, B.model.coef_)]
    k = int(np.argmax(np.abs(B.model.coef_)))
    obs["ObsDominant"] = (f"On the standardised scale the coefficient of largest magnitude is that of "
                          f"{phrase[B.features[k]]} ({B.model.coef_[k]:+.2f}).")
    obs["ObsCoef"] = ("Holding the other features fixed, " + "; ".join(parts[:-1]) + "; and " + parts[-1] + ".")

    # --- podium ------------------------------------------------------------------------
    pA, pB = podium["A"].metrics["test"], podium["B"].metrics["test"]
    obs["ObsPodium"] = (f"For the podium target the majority-class rule already reaches "
                        f"{pct(base_pod['majority']['accuracy'])} accuracy, so accuracy is uninformative; "
                        f"model B obtained F1 = {f3(pB['f1'])} and AUC = {f3(pB['auc'])} "
                        f"(model A: F1 = {f3(pA['f1'])}, AUC = {f3(pA['auc'])}), compared with F1 = "
                        f"{f3(base_pod['rule']['f1'])} and AUC = {f3(base_pod['rule']['auc'])} for the "
                        f"rule ``qualified in the top three''.")
    return obs


def write_macros(path: Path, macros: dict) -> None:
    lines = ["% AUTO-GENERATED by run_all.py - do not edit by hand."]
    for k, v in macros.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    _write(path, "\n".join(lines) + "\n")
