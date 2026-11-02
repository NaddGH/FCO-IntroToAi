# Predicting Formula 1 Race Finishing Positions Using Logistic Regression on Qualifying and Pit-Stop Data

Final Course Output - MCS 306 Introduction to Artificial Intelligence, Bulacan State University.

**Question.** Given a driver's qualifying position, Q1 pace and pit-stop record, will the driver finish in the
points-scoring top ten (secondary target: podium)? Logistic regression is trained by **batch gradient descent
implemented from scratch in NumPy**; the study measures how accuracy and convergence depend on the **learning rate**.

## Repository layout

| Path | Contents |
|------|----------|
| `run_all.py` | One command that runs the whole study |
| `src/f1logreg/logreg.py` | Logistic regression + gradient descent, Newton reference solver, smoothness constant |
| `src/f1logreg/data_prep.py` | Loads the Kaggle CSVs, builds features and targets, chronological split, standardisation |
| `src/f1logreg/experiments.py` | Learning-rate sweep, model selection, baselines, verification |
| `src/f1logreg/metrics.py` | Accuracy, precision, recall, F1, AUC, log-loss |
| `src/f1logreg/report.py` | Figures and the LaTeX tables/macros used by the paper |
| `src/f1logreg/make_synthetic.py` | **Synthetic** data generator (for tests / `--demo` only) |
| `tests/` | Unit tests (gradient check, GD vs Newton, metrics, data pipeline) |
| `data/` | Instructions for the Kaggle files (`data/README.md`) |
| `results/` | Generated CSV / JSON results, incl. the cleaned modelling table |
| `figures/` | Generated plots (PDF + PNG) |
| `paper/` | LaTeX paper (`main.tex`) and proposal (`proposal.tex`) |

## How to run

```bash
# 1. environment (Python 3.10+)
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. data: download https://www.kaggle.com/datasets/binayas/f1-dataset
#    and unzip the CSV files into data/raw/   (see data/README.md)

# 3. run the study (about a minute at most)
python run_all.py

# 4. build the paper (twice, for the table of contents)
cd paper && pdflatex main && pdflatex main
```

`python run_all.py --demo` runs the pipeline on **synthetic** data to check that everything works.
A demo run puts a red "DEMO RUN: SYNTHETIC DATA" banner on the paper; never submit that version.
Run the unit tests with `python -m pytest`.

Options: `--data-dir PATH`, `--min-year YEAR`, `--test-years N` (default 2), `--val-years N` (default 2).

## Method in brief

* Unit: one driver in one race, seasons that have pit-stop records. Label: `positionOrder <= 10` (and `<= 3`).
* Features: qualifying position, Q1 gap to the fastest Q1 time (%), number of pit stops, median pit-stop time.
  Model A = qualifying only; Model B = + pit-stop variables (known only during the race, so B is *not* a pre-race forecast).
* Chronological split (oldest seasons train, next two validation, newest two test); standardisation uses training data only.
* Gradient descent: zero initialisation, fixed 2000 iterations, learning-rate grid 1e-4 ... 20; the learning rate with the
  lowest **validation** loss is selected; the test set is used once.
* Checks: GD solution vs Newton's method and vs scikit-learn (unregularised); stability bound 2/L from the smoothness constant.

## Collaborative history

The assignment asks for version history showing collaborative work. Each member should clone the repository, work on a
branch, and commit under their own name, e.g.

```bash
git init && git add . && git commit -m "Initial project skeleton"
git checkout -b member-name/feature && ... && git push origin member-name/feature
```

then merge with pull requests, so the Git log shows who did what.
