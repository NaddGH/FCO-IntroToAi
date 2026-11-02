import numpy as np
import pandas as pd
import pytest

from f1logreg.data_prep import (FEATURE_SETS, Standardizer, build_dataset, parse_time_to_seconds,
                                split_by_year)
from f1logreg.experiments import run_sweep
from f1logreg.make_synthetic import make_synthetic


@pytest.fixture(scope="module")
def synthetic_dir(tmp_path_factory):
    return make_synthetic(tmp_path_factory.mktemp("synth"), first_year=2011, last_year=2020,
                          races_per_year=10, seed=3)


def test_parse_time():
    assert parse_time_to_seconds("1:27.456") == pytest.approx(87.456)
    assert parse_time_to_seconds("23.456") == pytest.approx(23.456)
    assert parse_time_to_seconds(12.5) == 12.5
    for bad in ("\\N", "", None, float("nan"), "abc", "0:00.000"):
        assert np.isnan(parse_time_to_seconds(bad))


def test_build_dataset_and_split(synthetic_dir):
    df, info = build_dataset(synthetic_dir)
    assert set(["q_pos", "q1_gap_pct", "n_stops", "median_pit_s", "top10", "podium"]) <= set(df.columns)
    assert (df["q1_gap_pct"] >= 0).all()
    assert df.groupby("raceid")["q1_gap_pct"].min().max() == pytest.approx(0.0, abs=1e-9)
    assert ((df["top10"] == 1) == (df["positionorder"] <= 10)).all()
    assert ((df["podium"] == 1) == (df["positionorder"] <= 3)).all()
    assert (df["median_pit_s"].dropna() <= 60).all()           # red-flag stops excluded
    assert df.loc[df["n_stops"] == 0, "median_pit_s"].isna().all()
    sp = split_by_year(df)
    # chronological, no overlap in seasons
    tr, va, te = (sp[sp["split"] == s]["year"] for s in ("train", "val", "test"))
    assert tr.max() < va.min() and va.max() < te.min()


def test_standardizer_uses_training_statistics_only(synthetic_dir):
    df, _ = build_dataset(synthetic_dir)
    sp = split_by_year(df)
    train = sp[sp["split"] == "train"]
    sc = Standardizer(FEATURE_SETS["B"]).fit(train)
    Xtr = sc.transform(train)
    assert np.allclose(Xtr.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(Xtr.std(axis=0), 1, atol=1e-9)
    assert np.isfinite(sc.transform(sp[sp["split"] == "test"])).all()


def test_sweep_agrees_with_newton(synthetic_dir):
    df, _ = build_dataset(synthetic_dir)
    sp = split_by_year(df)
    res = run_sweep(sp, "top10", "B", lr_grid=(0.01, 0.5, 1.0), n_iter=1500)
    assert res.verification["gd_vs_newton_max_abs_diff"] < 1e-4
    assert res.table["monotone"].all()                          # all below 2/L
    assert res.bound > 1.0


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="results"):
        build_dataset(tmp_path)
