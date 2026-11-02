"""Generate a SYNTHETIC dataset with the same file layout as the Kaggle F1 archive.

*** This is NOT real Formula 1 data. ***
It exists only so that the pipeline, the unit tests and the paper layout can be
checked without downloading anything. Never report numbers computed from it.

Files written (Ergast schema, missing values written as \\N like the real archive):
    races.csv, results.csv, qualifying.csv, pit_stops.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _fmt_lap(seconds: float) -> str:
    m = int(seconds // 60)
    return f"{m}:{seconds - 60 * m:06.3f}"


def make_synthetic(out_dir: str | Path, first_year: int = 2011, last_year: int = 2025,
                   races_per_year: int = 18, n_drivers: int = 20, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    races, results, qualifying, pits = [], [], [], []
    race_id = 0
    result_id = 0
    for year in range(first_year, last_year + 1):
        # latent car/driver strength (lower = faster), re-drawn mildly every season
        strength = rng.normal(0, 1, n_drivers)
        for rnd in range(1, races_per_year + 1):
            race_id += 1
            races.append({"raceId": race_id, "year": year, "round": rnd,
                          "circuitId": rnd, "name": f"Synthetic GP {rnd}"})
            pace = strength + rng.normal(0, 0.35, n_drivers)             # qualifying pace
            q1 = 85.0 + 1.1 * (pace - pace.min()) + rng.normal(0, 0.08, n_drivers)
            q_order = np.argsort(q1)
            q_pos = np.empty(n_drivers, dtype=int)
            q_pos[q_order] = np.arange(1, n_drivers + 1)

            stops = rng.choice([1, 2, 3], size=n_drivers, p=[0.45, 0.45, 0.10])
            retire = rng.random(n_drivers) < 0.10                        # DNF
            race_pace = pace + rng.normal(0, 0.6, n_drivers) + 0.15 * (stops - 2)
            race_score = np.where(retire, race_pace + 10.0 + rng.random(n_drivers), race_pace)
            finish_order = np.argsort(race_score)
            pos = np.empty(n_drivers, dtype=int)
            pos[finish_order] = np.arange(1, n_drivers + 1)

            for d in range(n_drivers):
                result_id += 1
                results.append({"resultId": result_id, "raceId": race_id, "driverId": d + 1,
                                "constructorId": d // 2 + 1, "grid": int(q_pos[d]),
                                "position": "\\N" if retire[d] else int(pos[d]),
                                "positionOrder": int(pos[d])})
                # a few drivers fail to set a Q1 time -> missing values like the real data
                no_time = rng.random() < 0.01
                qualifying.append({"qualifyId": result_id, "raceId": race_id, "driverId": d + 1,
                                   "constructorId": d // 2 + 1, "number": d + 1,
                                   "position": int(q_pos[d]),
                                   "q1": "\\N" if no_time else _fmt_lap(q1[d]),
                                   "q2": "\\N", "q3": "\\N"})
                n_st = 0 if (retire[d] and rng.random() < 0.5) else int(stops[d])
                for s in range(1, n_st + 1):
                    dur = 22.0 + abs(rng.normal(0, 1.2)) + 0.3 * (pace[d] - pace.mean())
                    if rng.random() < 0.01:
                        dur += 25.0                                         # slow stop
                    if rng.random() < 0.003:
                        dur = 1500.0                                        # red-flag stop
                    shown = f"{dur:.3f}" if dur < 60 else _fmt_lap(dur)     # '1:02.345' style
                    pits.append({"raceId": race_id, "driverId": d + 1, "stop": s,
                                 "lap": 8 * s, "time": "14:30:00", "duration": shown,
                                 "milliseconds": int(round(dur * 1000))})

    pd.DataFrame(races).to_csv(out / "races.csv", index=False)
    pd.DataFrame(results).to_csv(out / "results.csv", index=False)
    pd.DataFrame(qualifying).to_csv(out / "qualifying.csv", index=False)
    pd.DataFrame(pits).to_csv(out / "pit_stops.csv", index=False)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print("Wrote SYNTHETIC data to", make_synthetic(args.out, seed=args.seed))
