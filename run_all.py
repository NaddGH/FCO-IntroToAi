#!/usr/bin/env python
"""Run the whole study: data -> experiments -> results/*.csv -> figures/*.pdf -> paper/generated/*.tex

Usage
-----
    python run_all.py                       # real data in data/raw  (Kaggle: binayas/f1-dataset)
    python run_all.py --data-dir path/to/f1 # real data somewhere else
    python run_all.py --demo                # SYNTHETIC data, only to test the pipeline

After a real run, compile the paper:   cd paper && pdflatex main && pdflatex main
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from f1logreg import report  # noqa: E402
from f1logreg.data_prep import FEATURE_SETS, TARGETS, build_dataset, split_by_year  # noqa: E402
from f1logreg.experiments import (EPS_EXCESS, LR_GRID, N_ITER, baseline_metrics,  # noqa: E402
                                  per_qualifying_position, run_sweep)
from f1logreg.logreg import LogisticRegressionGD, fit_newton  # noqa: E402
from f1logreg.make_synthetic import make_synthetic  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "raw"),
                    help="folder containing the Kaggle CSV files (searched recursively)")
    ap.add_argument("--demo", action="store_true",
                    help="use SYNTHETIC data (NOT real F1 data) to test the pipeline")
    ap.add_argument("--min-year", type=int, default=None, help="optional first season to keep")
    ap.add_argument("--test-years", type=int, default=2, help="number of newest seasons used for testing")
    ap.add_argument("--val-years", type=int, default=2, help="number of seasons before them used for validation")
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--figures-dir", default=str(ROOT / "figures"))
    ap.add_argument("--paper-gen-dir", default=str(ROOT / "paper" / "generated"))
    args = ap.parse_args()

    results_dir, figdir, gen = Path(args.results_dir), Path(args.figures_dir), Path(args.paper_gen_dir)
    for d in (results_dir, figdir, gen):
        d.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir
    if args.demo:
        data_dir = str(ROOT / "data" / "synthetic")
        make_synthetic(data_dir)
        print("*** DEMO MODE: using SYNTHETIC data. These numbers are NOT real. ***")

    # ------------------------------------------------------------------ data
    df, info = build_dataset(data_dir, min_year=args.min_year)
    df = split_by_year(df, test_years=args.test_years, val_years=args.val_years)
    print(f"Modelling table: {info['n_rows']:,} driver-race rows, {info['n_races']:,} races, "
          f"seasons {info['year_min']}-{info['year_max']}")
    split_summary = df.groupby("split").agg(rows=("year", "size"), first_year=("year", "min"),
                                            last_year=("year", "max"), races=("raceid", "nunique"),
                                            top10_rate=("top10", "mean"), podium_rate=("podium", "mean"))
    print(split_summary.loc[["train", "val", "test"]].to_string())

    # ----------------------------------------------------------- experiments
    results = {f: run_sweep(df, "top10", f) for f in FEATURE_SETS}
    podium = {f: run_sweep(df, "podium", f) for f in FEATURE_SETS}
    base = baseline_metrics(df, "top10")
    base_pod = baseline_metrics(df, "podium")
    by_q = per_qualifying_position(df, "top10", results)

    # ------------------------------------------------------- save raw results
    df.to_csv(results_dir / "modelling_table.csv", index=False)
    split_summary.to_csv(results_dir / "dataset_split_summary.csv")
    for tgt, group in (("top10", results), ("podium", podium)):
        for f, r in group.items():
            r.table.to_csv(results_dir / f"sweep_{tgt}_{f}.csv", index=False)
    rows = []
    for tgt, group, b in (("top10", results, base), ("podium", podium, base_pod)):
        rows.append({"target": tgt, "model": "majority", **b["majority"]})
        rows.append({"target": tgt, "model": "rule", **b["rule"]})
        for f, r in group.items():
            for s in ("train", "val", "test"):
                rows.append({"target": tgt, "model": f"logreg_{f}", "split": s,
                             "lr": r.best_lr, **r.metrics[s]})
    pd.DataFrame(rows).to_csv(results_dir / "metrics_all.csv", index=False)
    by_q.to_csv(results_dir / "top10_rate_by_qualifying_position.csv", index=False)
    meta = {
        "mode": "SYNTHETIC DEMO" if args.demo else "REAL DATA",
        "data_dir": str(data_dir),
        "cleaning": info,
        "n_iter": N_ITER, "lr_grid": list(LR_GRID), "eps_excess": EPS_EXCESS,
        "smoothness_L": {f: r.L for f, r in results.items()},
        "stability_bound_2_over_L": {f: r.bound for f, r in results.items()},
        "selected_lr": {f"{t}_{f}": r.best_lr for t, g in (("top10", results), ("podium", podium))
                        for f, r in g.items()},
        "verification": {f"{t}_{f}": r.verification for t, g in (("top10", results), ("podium", podium))
                         for f, r in g.items()},
        "coefficients_top10_B": dict(zip(["intercept"] + results["B"].features,
                                         [results["B"].model.intercept_] + list(results["B"].model.coef_))),
    }
    (results_dir / "run_summary.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")

    # --------------------------------------------------------------- figures
    report.fig_probability_by_qpos(by_q, figdir, "top-10 finish")
    report.fig_loss_curves(results["B"], figdir)
    report.fig_accuracy_vs_lr(results, base, figdir)
    report.fig_loss_vs_lr(results, figdir)
    report.fig_confusion(results["B"], figdir)

    # ---------------------------------------------------------------- tables
    report.table_dataset(df, gen)
    report.table_sweep(results["B"], gen, "tab_sweep_B.tex")
    report.table_sweep(results["A"], gen, "tab_sweep_A.tex")
    report.table_metrics(results, base, gen, "tab_metrics_top10.tex", "Rule: qualified top 10")
    report.table_metrics(podium, base_pod, gen, "tab_metrics_podium.tex", "Rule: qualified top 3")
    report.table_coefficients(results["B"], gen, "tab_coef_B.tex")
    report.table_coefficients(results["A"], gen, "tab_coef_A.tex")
    report.table_confusion(results, gen)
    report.table_verification(results, gen)

    # ---------------------------------------------------------------- timing
    Xtr_B, ytr_B = results["B"].data["train"]
    t0 = time.perf_counter()
    LogisticRegressionGD(lr=results["B"].best_lr, n_iter=N_ITER).fit(Xtr_B, ytr_B)
    time_gd = time.perf_counter() - t0
    t0 = time.perf_counter()
    fit_newton(Xtr_B, ytr_B)
    time_newton = time.perf_counter() - t0
    tB = results["B"].table

    # ---------------------------------------------------------------- macros
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
    mA, mB = results["A"].metrics["test"], results["B"].metrics["test"]
    macros = {
        "DataNote": (r"\textcolor{red}{\textbf{DEMO RUN: SYNTHETIC DATA, NOT REAL F1 RESULTS}}"
                     if args.demo else ""),
        "NRows": f"{len(df):,}".replace(",", "{,}"),
        "NRaces": f"{df['raceid'].nunique():,}".replace(",", "{,}"),
        "NTrain": f"{len(tr):,}".replace(",", "{,}"),
        "NVal": f"{len(va):,}".replace(",", "{,}"),
        "NTest": f"{len(te):,}".replace(",", "{,}"),
        "YearsAll": report.years_text(info["year_min"], info["year_max"]),
        "YearsTrain": report.years_text(int(tr["year"].min()), int(tr["year"].max())),
        "YearsVal": report.years_text(int(va["year"].min()), int(va["year"].max())),
        "YearsTest": report.years_text(int(te["year"].min()), int(te["year"].max())),
        "NIter": f"{N_ITER:,}".replace(",", "{,}"),
        "EpsTol": f"{EPS_EXCESS:g}",
        "NLR": str(len(LR_GRID)),
        "LRGrid": ",\\,".join(report.lr_math(x) for x in LR_GRID),
        "LRMin": f"${report.lr_math(min(LR_GRID))}$",
        "LRMax": f"${report.lr_math(max(LR_GRID))}$",
        "LipA": f"{results['A'].L:.3f}", "LipB": f"{results['B'].L:.3f}",
        "BoundA": f"{results['A'].bound:.2f}", "BoundB": f"{results['B'].bound:.2f}",
        "BestLRA": f"${report.lr_math(results['A'].best_lr)}$",
        "BestLRB": f"${report.lr_math(results['B'].best_lr)}$",
        "TopRateTrain": report.pct(tr["top10"].mean()),
        "PodRateTrain": report.pct(tr["podium"].mean()),
        "AccMaj": report.pct(base["majority"]["accuracy"]),
        "AccRule": report.pct(base["rule"]["accuracy"]),
        "AccA": report.pct(mA["accuracy"]), "AccB": report.pct(mB["accuracy"]),
        "AUCA": report.f3(mA["auc"]), "AUCB": report.f3(mB["auc"]),
        "FoneA": report.f3(mA["f1"]), "FoneB": report.f3(mB["f1"]),
        "LossStarB": f"{results['B'].loss_star:.4f}",
        "AccLRMinB": report.pct(tB["test_acc"].min()), "AccLRMaxB": report.pct(tB["test_acc"].max()),
        "NMonoB": str(int(tB["monotone"].sum())),
        "TimeGD": f"{1000 * time_gd:.0f}", "TimeNewton": f"{1000 * time_newton:.0f}",
        "NDropQual": f"{info['n_dropped_missing_qualifying']:,}".replace(",", "{,}"),
        "NDropOutlier": f"{info['n_dropped_q1_outlier']:,}".replace(",", "{,}"),
        "NPitStops": f"{info['n_pit_stops_total']:,}".replace(",", "{,}"),
        "NPitOverCap": f"{info['n_pit_stops_over_cap']:,}".replace(",", "{,}"),
        "NRacesPit": f"{info['n_races_with_pit_data']:,}".replace(",", "{,}"),
    }
    macros.update(report.observation_sentences(results, podium, base, base_pod))
    report.write_macros(gen / "results.tex", macros)

    # --------------------------------------------------------------- console
    print("\nSelected learning rates (by validation loss):",
          {f: r.best_lr for f, r in results.items()})
    print(f"Test accuracy  A: {mA['accuracy']:.4f}   B: {mB['accuracy']:.4f}   "
          f"majority: {base['majority']['accuracy']:.4f}   rule: {base['rule']['accuracy']:.4f}")
    print("Verification (model B):", results["B"].verification)
    print(f"\nDone. Results -> {results_dir}, figures -> {figdir}, LaTeX fragments -> {gen}")


if __name__ == "__main__":
    main()
