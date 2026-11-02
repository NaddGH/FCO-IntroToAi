"""Load the Kaggle F1 archive (Ergast/Jolpica schema) and build a modelling table.

One row of the final table = one driver in one race.

Targets (binary)
    top10   : 1 if the driver's final classification (``positionOrder``) is 1..10
              (the points-scoring positions since 2010)
    podium  : 1 if the final classification is 1..3

Features
    q_pos        qualifying position (1 = pole)
    q1_gap_pct   (driver's Q1 time - fastest Q1 time of that race) / fastest Q1 time * 100
    n_stops      number of pit stops made in the race               (in-race information)
    median_pit_s median pit-stop duration in seconds                (in-race information)

Feature set A = qualifying only (known before the race).
Feature set B = qualifying + pit-stop data (known only during/after the race).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_SETS = {
    "A": ["q_pos", "q1_gap_pct"],
    "B": ["q_pos", "q1_gap_pct", "n_stops", "median_pit_s"],
}
FEATURE_LABELS = {
    "q_pos": "Qualifying position",
    "q1_gap_pct": "Q1 gap to fastest (\\%)",
    "n_stops": "Number of pit stops",
    "median_pit_s": "Median pit-stop time (s)",
}
TARGETS = {"top10": 10, "podium": 3}

# A Q1 time more than this far (in %) behind the fastest Q1 time of the same race is
# treated as a data error / non-representative lap and the row is dropped.
MAX_Q1_GAP_PCT = 20.0

# Pit-lane times longer than this (seconds) are red-flag stops, repairs or data errors,
# not ordinary service stops; they are excluded from the median pit time (but still
# counted in n_stops).
MAX_PIT_S = 60.0

_FILE_ALIASES = {
    "results": ["results"],
    "races": ["races"],
    "qualifying": ["qualifying", "qualifyings"],
    "pit_stops": ["pit_stops", "pitstops", "pit_stop", "pit_stop_records"],
}
_REQUIRED_COLUMNS = {
    "results": ["raceid", "driverid", "positionorder"],
    "races": ["raceid", "year"],
    "qualifying": ["raceid", "driverid", "position", "q1"],
    "pit_stops": ["raceid", "driverid"],
}


# --------------------------------------------------------------------------- #
# Reading helpers
# --------------------------------------------------------------------------- #
def find_csv(data_dir: str | Path, table: str) -> Path:
    """Find ``<table>.csv`` anywhere below ``data_dir`` (case-insensitive)."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data folder '{data_dir}' does not exist. Download the Kaggle dataset "
            "(binayas/f1-dataset) and unzip it there - see data/README.md."
        )
    aliases = _FILE_ALIASES[table]
    hits = [p for p in data_dir.rglob("*") if p.suffix.lower() == ".csv" and p.stem.lower() in aliases]
    if not hits:
        found = sorted(p.name for p in data_dir.rglob("*.csv"))
        raise FileNotFoundError(
            f"Could not find a file named {aliases[0]}.csv under '{data_dir}'. "
            f"CSV files found: {found if found else 'none'}"
        )
    return sorted(hits, key=lambda p: (len(p.parts), str(p)))[0]


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV; the Ergast archive encodes missing values as the two characters \\N."""
    df = pd.read_csv(path, na_values=["\\N"], keep_default_na=True, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def check_columns(df: pd.DataFrame, table: str) -> None:
    missing = [c for c in _REQUIRED_COLUMNS[table] if c not in df.columns]
    if missing:
        raise KeyError(
            f"Table '{table}' is missing required column(s) {missing}. "
            f"Columns present: {list(df.columns)}"
        )


def parse_time_to_seconds(x) -> float:
    """'1:27.456' -> 87.456 ; '23.456' -> 23.456 ; numbers pass through ; junk -> NaN."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x)
        return v if np.isfinite(v) and v > 0 else np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "\\n"}:
        return np.nan
    try:
        parts = s.split(":")
        if len(parts) == 1:
            secs = float(parts[0])
        elif len(parts) == 2:
            secs = float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            secs = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        else:
            return np.nan
    except ValueError:
        return np.nan
    return secs if np.isfinite(secs) and secs > 0 else np.nan


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #
def build_dataset(data_dir: str | Path, min_year: int | None = None,
                  max_year: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (modelling table, info dict with row counts at each cleaning step)."""
    tables = {}
    for name in _FILE_ALIASES:
        tables[name] = read_table(find_csv(data_dir, name))
        check_columns(tables[name], name)
    results, races, qual, pits = (tables["results"], tables["races"],
                                  tables["qualifying"], tables["pit_stops"])
    info: dict = {}

    # ---- results: the label -------------------------------------------------
    res = results[["raceid", "driverid", "positionorder"]].copy()
    for c in res.columns:
        res[c] = pd.to_numeric(res[c], errors="coerce")
    res = res.dropna().astype(int).drop_duplicates(["raceid", "driverid"])
    info["n_result_rows"] = len(res)

    # ---- races: season of each race ----------------------------------------
    rc_cols = ["raceid", "year"] + [c for c in ("round",) if c in races.columns]
    rc = races[rc_cols].copy()
    for c in rc.columns:
        rc[c] = pd.to_numeric(rc[c], errors="coerce")
    rc = rc.dropna(subset=["raceid", "year"]).astype({"raceid": int, "year": int})

    # ---- qualifying: position and Q1 gap ------------------------------------
    q = qual[["raceid", "driverid", "position", "q1"]].copy()
    q["raceid"] = pd.to_numeric(q["raceid"], errors="coerce")
    q["driverid"] = pd.to_numeric(q["driverid"], errors="coerce")
    q["q_pos"] = pd.to_numeric(q["position"], errors="coerce")
    q["q1_s"] = q["q1"].map(parse_time_to_seconds)
    q = q.dropna(subset=["raceid", "driverid"]).astype({"raceid": int, "driverid": int})
    q = q.drop_duplicates(["raceid", "driverid"])
    fastest = q.groupby("raceid")["q1_s"].transform("min")
    q["q1_gap_pct"] = (q["q1_s"] / fastest - 1.0) * 100.0
    q = q[["raceid", "driverid", "q_pos", "q1_gap_pct"]]

    # ---- pit stops: per driver per race --------------------------------------
    p = pits.copy()
    p["raceid"] = pd.to_numeric(p["raceid"], errors="coerce")
    p["driverid"] = pd.to_numeric(p["driverid"], errors="coerce")
    p = p.dropna(subset=["raceid", "driverid"]).astype({"raceid": int, "driverid": int})
    secs = pd.Series(np.nan, index=p.index)
    if "milliseconds" in p.columns:
        secs = pd.to_numeric(p["milliseconds"], errors="coerce") / 1000.0
    if "duration" in p.columns:
        parsed = p["duration"].map(parse_time_to_seconds)
        secs = secs.where(secs.notna() & (secs > 0), parsed)
    info["n_pit_stops_total"] = int(len(p))
    info["n_pit_stops_over_cap"] = int((secs > MAX_PIT_S).sum())
    p["pit_s"] = secs.where(secs <= MAX_PIT_S)
    agg = p.groupby(["raceid", "driverid"]).agg(n_stops=("pit_s", "size"),
                                                median_pit_s=("pit_s", "median")).reset_index()
    races_with_pit_data = set(p["raceid"].unique())
    info["n_races_with_pit_data"] = len(races_with_pit_data)

    # ---- join ----------------------------------------------------------------
    df = res.merge(rc, on="raceid", how="inner")
    info["n_after_races_join"] = len(df)
    df = df[df["raceid"].isin(races_with_pit_data)]          # pit data must exist for the race
    info["n_in_races_with_pit_data"] = len(df)
    df = df.merge(q, on=["raceid", "driverid"], how="inner")
    info["n_after_qualifying_join"] = len(df)
    df = df.merge(agg, on=["raceid", "driverid"], how="left")
    df["n_stops"] = df["n_stops"].fillna(0).astype(int)      # no record = no stop

    # ---- cleaning ------------------------------------------------------------
    before = len(df)
    df = df.dropna(subset=["q_pos", "q1_gap_pct"])
    info["n_dropped_missing_qualifying"] = before - len(df)
    before = len(df)
    df = df[df["q1_gap_pct"] <= MAX_Q1_GAP_PCT]
    info["n_dropped_q1_outlier"] = before - len(df)
    if min_year is not None:
        df = df[df["year"] >= min_year]
    if max_year is not None:
        df = df[df["year"] <= max_year]

    # ---- targets ---------------------------------------------------------------
    for name, cutoff in TARGETS.items():
        df[name] = (df["positionorder"] <= cutoff).astype(int)

    df = df.sort_values(["year", "raceid", "positionorder"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("The modelling table is empty after cleaning - check the input files.")
    info["n_rows"] = len(df)
    info["n_races"] = int(df["raceid"].nunique())
    info["year_min"] = int(df["year"].min())
    info["year_max"] = int(df["year"].max())
    return df, info


# --------------------------------------------------------------------------- #
# Chronological split and standardisation
# --------------------------------------------------------------------------- #
def split_by_year(df: pd.DataFrame, test_years: int = 2, val_years: int = 2) -> pd.DataFrame:
    """Chronological split: oldest seasons -> train, then validation, newest -> test."""
    years = sorted(df["year"].unique())
    if len(years) < test_years + val_years + 1:
        raise ValueError(
            f"Need at least {test_years + val_years + 1} seasons for the split, found {len(years)}."
        )
    test_set = set(years[-test_years:])
    val_set = set(years[-(test_years + val_years):-test_years])
    out = df.copy()
    out["split"] = np.where(out["year"].isin(test_set), "test",
                            np.where(out["year"].isin(val_set), "val", "train"))
    for t in TARGETS:
        for s in ("train", "val", "test"):
            if out.loc[out["split"] == s, t].nunique() < 2:
                raise ValueError(f"Split '{s}' has only one class for target '{t}'.")
    return out


class Standardizer:
    """Median-impute then z-score, with every statistic estimated on the training rows only."""

    def __init__(self, columns: list[str]):
        self.columns = list(columns)

    def fit(self, train: pd.DataFrame) -> "Standardizer":
        X = train[self.columns].astype(float)
        self.median_ = X.median()
        if self.median_.isna().any():
            bad = list(self.median_[self.median_.isna()].index)
            raise ValueError(f"Column(s) {bad} have no observed values in the training set.")
        filled = X.fillna(self.median_)
        self.mean_ = filled.mean()
        std = filled.std(ddof=0)
        self.std_ = std.where(std > 0, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.columns].astype(float).fillna(self.median_)
        return ((X - self.mean_) / self.std_).to_numpy(dtype=float)
