# Reservoir Permeability Prediction Using Machine Learning — Rio Del Rey (RDR) Basin, Cameroon

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research%20code-yellow)

This repository contains the code used to predict reservoir permeability from wireline well
logs in the Rio Del Rey (RDR) Basin, Cameroon, using three machine learning regression
models: **Random Forest (RF)**, **Support Vector Regression (SVR)**, and **Extreme Gradient
Boosting (XGBoost)**.

It accompanies the manuscript *"Machine Learning Models Application for Reservoir
Permeability Prediction in the Rio Del Rey (RDR) Basin, Cameroon"*. This README is written to
stand on its own: it explains what the code does, why each part exists, and how to run it,
without requiring you to cross-reference the manuscript's figures or section numbers.

---

## Table of contents

1. [What this project does](#1-what-this-project-does)
2. [Why machine learning for permeability](#2-why-machine-learning-for-permeability)
3. [Repository structure](#3-repository-structure)
4. [Input features](#4-input-features)
5. [Requirements](#5-requirements)
6. [Installation](#6-installation)
7. [How to use this code](#7-how-to-use-this-code)
8. [Quick test (no real data needed)](#8-quick-test-no-real-data-needed)
9. [The three models, explained](#9-the-three-models-explained)
10. [Data availability](#10-data-availability)
11. [Reproducibility notes](#11-reproducibility-notes)
12. [Troubleshooting](#12-troubleshooting)
13. [License](#13-license)
14. [Citation](#14-citation)
15. [Contributing](#15-contributing)
16. [Acknowledgments](#16-acknowledgments)
17. [Contact](#17-contact)

---

## 1. What this project does

Reservoir permeability — how easily fluids flow through rock — is usually measured directly
from physical core samples, which is expensive, slow, and only available at a few depths
along a well. This project instead predicts permeability directly from standard wireline well
logs (measurements taken continuously along the entire wellbore), using machine learning
models trained on wells where both logs and true permeability are known.

Given a well's log readings at any depth, the trained models output a predicted permeability
value, without needing a physical core sample at that depth.

## 2. Why machine learning for permeability

Classical formulas for estimating permeability from logs (e.g., Kozeny-Carman, Wyllie-Rose)
assume simplified, uniform rock properties. Real reservoirs — especially structurally complex
ones like the RDR Basin — have non-linear relationships between log measurements and
permeability that these formulas don't capture well. Machine learning models can learn these
non-linear relationships directly from data instead of assuming a fixed formula, which
generally produces more accurate predictions in heterogeneous reservoirs.

This project trains and compares three models with different strengths:
- **Random Forest** — an ensemble of decision trees, robust to noise and less prone to
  overfitting.
- **Support Vector Regression** — good at handling non-linear relationships via a kernel
  function, but can struggle with extreme outliers.
- **XGBoost** — a gradient-boosted ensemble that iteratively corrects its own errors, often
  the strongest performer on structured, tabular data like well logs.

## 3. Repository structure

```
Permeability-Prediction/
├── data/                          # Not included in this public repo — see Section 10
├── notebooks/
│   ├── permeability_prediction.ipynb   # Main entry point: run this to see the full workflow
│   └── results/
│       └── figures/
│           ├── scatter_plot.png        # Predicted-vs-actual permeability plot
│           └── scatter_plot.pdf        # Same plot, vector format
├── src/
│   ├── __init__.py                # Marks src/ as an importable Python package
│   ├── data_processing.py         # Loads raw well-log files and merges them into one dataset
│   ├── preprocessing.py           # Cleans data: fills missing values, scales features, log-transforms permeability
│   ├── models.py                  # Defines the RF, SVR, and XGBoost model objects
│   ├── training.py                # Fits a given model on the training data
│   ├── tuning.py                  # Searches for the best hyperparameters for each model via cross-validation
│   ├── evaluation.py              # Scores a trained model on unseen test data (R², RMSE, MSE)
│   └── viualisation.py            # Builds the correlation, learning-curve, and scatter-plot figures
├── tests/
│   └── quick_test.py              # Runs the whole pipeline on fake data, to prove the code works
├── pyproject.toml                 # Exact list of dependencies and their versions
├── .gitignore                     # Tells git to skip the real data folder and cache files
├── LICENSE                        # MIT License
└── README.md
```

## 4. Input features

The models take these well-log measurements as input:

| Log mnemonic | What it measures | Unit |
|---|---|---|
| GR | Gamma Ray — natural radioactivity, helps identify shale vs. clean sand | API |
| NPHI | Neutron Porosity — estimates pore space | fraction (v/v) |
| RHOB | Bulk Density — rock density | g/cm³ |
| ILD | Deep Induction Resistivity — how well the rock resists electrical current | ohm·m |
| CAL | Caliper — actual borehole diameter | in |
| DT | Sonic transit time — how fast sound travels through the rock | µs/ft |
| SP | Spontaneous Potential — natural electrical potential in the borehole | mV |
| LITHO | Lithology facies classification (e.g., sandstone, shale) | categorical |

The model's output (target variable) is **logK** — permeability, transformed with log10 to
make its distribution easier for the models to learn from.

## 5. Requirements

- Python ≥ 3.10
- ~500 MB free disk space (dependencies + bundled synthetic test data)
- No GPU required
- Works on Windows, macOS, and Linux

## 6. Installation

**Using `uv` (recommended — fast, reproducible dependency resolution):**
```bash
git clone https://github.com/aryl-shamir/Permeability-Prediction.git
cd Permeability-Prediction
uv sync
```

**Using `pip`:**
```bash
git clone https://github.com/aryl-shamir/Permeability-Prediction.git
cd Permeability-Prediction
python -m venv .venv
source .venv/Scripts/activate       # Windows (Git Bash); .venv\Scripts\activate.bat on cmd.exe
# source .venv/bin/activate         # macOS/Linux
pip install -e .
```

Core dependencies (pinned in `pyproject.toml`): NumPy, Pandas, SciPy, scikit-learn, XGBoost,
Matplotlib, Seaborn, Jupyter.

## 7. How to use this code

The main entry point is the notebook. Launch it with:

```bash
jupyter lab notebooks/permeability_prediction.ipynb
```

Running it from top to bottom will:

1. **Load the data** — read in well-log files and combine them into a single working dataset
   (`src/data_processing.py`).
2. **Clean and prepare it** — fill in missing values, scale the features to comparable
   ranges, and log-transform the permeability target, fitting these steps only on the
   training portion of the data to avoid inflating performance artificially
   (`src/preprocessing.py`).
3. **Explore relationships** — compute how strongly each log correlates with permeability,
   which helps explain which measurements are most informative (`src/viualisation.py`).
4. **Train and tune each model** — search over a range of hyperparameters for RF, SVR, and
   XGBoost, using cross-validation to pick the settings that generalize best rather than
   just memorizing the training data (`src/tuning.py`, `src/training.py`).
5. **Check for overfitting** — plot learning curves (how error changes as more training data
   is used) to confirm each model generalizes well rather than just memorizing.
6. **Evaluate on unseen data** — score each trained model on a held-out test set it never saw
   during training, and plot predicted vs. actual permeability values
   (`src/evaluation.py`).

Expected runtime on a standard laptop CPU: a few minutes; no GPU is required.

## 8. Quick test (no real data needed)

To simply confirm the code runs correctly — without needing the real well-log data — run:

```bash
python tests/quick_test.py
# or, with uv:
uv run python tests/quick_test.py
```

This runs the exact same steps as the main notebook (clean → train → evaluate), but on a
small, made-up dataset that's already bundled in the repository. It finishes in a few seconds
and prints:

```
Quick test completed successfully.
```

Because this uses fake data, the numbers it produces won't match real results — it's only
there to prove the code itself works correctly for anyone who downloads the repository.

## 9. The three models, explained

| Model | How it works | Typical trade-off |
|---|---|---|
| **Random Forest** | Builds many decision trees on random subsets of the data and features, then averages their predictions | Robust and hard to overfit, but can be slightly less accurate than boosting on complex patterns |
| **Support Vector Regression** | Fits a function that tries to keep predictions within a small margin of error, using a kernel to handle non-linear patterns | Sensitive to outliers and can underperform when the data has sharp local changes |
| **XGBoost** | Builds trees one at a time, where each new tree specifically corrects the mistakes of the previous ones | Often the most accurate on structured tabular data, but requires more careful tuning to avoid overfitting |

Each model is trained with the same input features and evaluated with the same metrics, so
their results are directly comparable.

## 10. Data availability

The real well-log data used to train and test these models is proprietary — it was shared by
the operating company under a confidentiality agreement and cannot be redistributed publicly.
For that reason, the `data/` folder is intentionally excluded from this repository (see
`.gitignore`).

This does **not** affect the code itself: everything in `src/`, `notebooks/`, and `tests/` is
complete and runnable. To make sure of that, a small synthetic dataset with the same column
structure is bundled with `tests/quick_test.py`, so anyone can confirm the pipeline works
correctly without needing the real data (see Section 8).

If you need access to the original data for research purposes, contact the corresponding
author (Section 17) — access may be possible through a separate data-use agreement with the
data owner.

## 11. Reproducibility notes

- Random elements (train/test split, model initialization, cross-validation folds) use fixed
  random seeds wherever the underlying libraries support it, so re-running the code on the
  same data produces consistent results.
- Data cleaning and scaling steps are fit only on the training data and then applied to the
  test data, to avoid the model getting an unfair "preview" of test information.
- Exact dependency versions are pinned (`pyproject.toml` / `uv.lock`) so results don't drift
  due to library updates over time.

## 12. Troubleshooting

- **`ModuleNotFoundError`** — make sure your virtual environment is activated
  (`source .venv/Scripts/activate` or `uv run ...`) and that installation finished without
  errors.
- **XGBoost won't install on Apple Silicon (M1/M2/M3)** — try installing via `conda`/`mamba`
  instead of `pip`, or make sure you're using a recent XGBoost build that supports `arm64`.
- **`FileNotFoundError` mentioning `data/`** — expected if you don't have the real dataset;
  use `tests/quick_test.py` instead, which only needs the bundled synthetic data.
- **Jupyter can't find a kernel** — after activating your environment, run
  `python -m ipykernel install --user --name permeability-env`, then select that kernel
  inside Jupyter.

## 13. License

This project is released under the [MIT License](LICENSE), which allows reuse, modification,
and redistribution with attribution.

