# Permeability Prediction Using Machine Learning

This repository contains the source code used for the study on machine-learning-based prediction of permeability from well-log data.

The study uses well-log data from wells P4 and P5 and evaluates three regression algorithms:

* Random Forest
* Support Vector Regression (SVR)
* XGBoost

Permeability is transformed to a base-10 logarithmic scale (`logK`) before model training.

## Repository Structure

```text
Permeability-Prediction/
├── data/              # Well-log datasets
├── notebooks/         # Complete analysis notebook
├── src/               # Research source code
├── tests/             # Quick test
├── pyproject.toml     # Project dependencies
└── README.md
```

## Requirements

The main Python dependencies are:

* NumPy
* Pandas
* SciPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn

The dependencies are specified in `pyproject.toml`.

## Running the Analysis

The complete analysis is provided in:

```text
notebooks/permeability_prediction.ipynb
```

Open the notebook using Jupyter Notebook, JupyterLab, or Visual Studio Code and run the cells sequentially.

The notebook performs data preparation, preprocessing, model training, five-fold cross-validation, learning-curve analysis, hyperparameter tuning, model evaluation, and visualization.

## Quick Test

A quick test is provided in:

```text
tests/quick_test.py
```

From the repository root, run:

```bash
python tests/quick_test.py
```

or, when using `uv`:

```bash
uv run python tests/quick_test.py
```

A successful execution should end with:

```text
Quick test completed successfully.
```

The quick test uses a small synthetic dataset and is intended to verify that the main data-processing, preprocessing, modelling, and evaluation components are functioning correctly.

## Source Code

The source code is organized into individual Python files in the `src/` directory.

The repository is publicly available at:

<https://github.com/aryl-shamir/Permeability-Prediction>

## Citation

If you use this code, please cite the corresponding scientific article.
