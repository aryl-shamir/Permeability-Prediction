from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data_processing import combine_wells, create_logK
from src.preprocessing import create_preprocessor
from src.models import build_models
from src.evaluation import regression_metrics


def main():

    print("Running permeability prediction quick test...\n")

    # ---------------------------------------------------------
    # 1. Create a small synthetic well dataset
    # ---------------------------------------------------------

    np.random.seed(42)

    n_samples = 20

    well_1 = pd.DataFrame({
        "DEPTH": np.arange(1000, 1000 + n_samples),
        "CAL": np.random.uniform(8, 12, n_samples),
        "DT": np.random.uniform(50, 100, n_samples),
        "GR": np.random.uniform(20, 80, n_samples),
        "NPHI": np.random.uniform(0.1, 0.4, n_samples),
        "ILD": np.random.uniform(1, 100, n_samples),
        "RHOB": np.random.uniform(2.0, 2.7, n_samples),
        "SP": np.random.uniform(-50, 50, n_samples),
        "K": np.random.uniform(300, 4500, n_samples),
        "porosite": np.random.uniform(0.1, 0.3, n_samples)
    })

    well_2 = pd.DataFrame({
        "DEPTH": np.arange(2000, 2000 + n_samples),
        "CAL": np.random.uniform(8, 12, n_samples),
        "DT": np.random.uniform(50, 100, n_samples),
        "GR": np.random.uniform(20, 80, n_samples),
        "NPHI": np.random.uniform(0.1, 0.4, n_samples),
        "ILD": np.random.uniform(1, 100, n_samples),
        "RHOB": np.random.uniform(2.0, 2.7, n_samples),
        "SP": np.random.uniform(-50, 50, n_samples),
        "K": np.random.uniform(300, 4500, n_samples),
        "porosite": np.random.uniform(0.1, 0.3, n_samples)
    })

    print("✓ Synthetic datasets created")


    # ---------------------------------------------------------
    # 2. Combine wells
    # ---------------------------------------------------------

    wells = combine_wells(
        [well_1, well_2],
        ["p4", "p5"]
    )

    assert len(wells) == 40
    assert "Well Name" in wells.columns

    print("✓ Well combination works")


    # ---------------------------------------------------------
    # 3. Create logK target
    # ---------------------------------------------------------

    wells_no_depth = create_logK(wells)

    assert "DEPTH" not in wells_no_depth.columns
    assert "logK" in wells_no_depth.columns
    assert wells_no_depth["logK"].notna().all()

    print("✓ logK transformation works")


    # ---------------------------------------------------------
    # 4. Prepare features and target
    # ---------------------------------------------------------

    features = [
        "CAL",
        "DT",
        "GR",
        "NPHI",
        "ILD",
        "RHOB",
        "SP"
    ]

    X = wells_no_depth[features]
    y = wells_no_depth["logK"]

    print("✓ Feature and target preparation works")


    # ---------------------------------------------------------
    # 5. Create preprocessing pipeline
    # ---------------------------------------------------------

    preprocessor = create_preprocessor()

    X_processed = preprocessor.fit_transform(X)

    assert X_processed.shape[0] == len(X)

    print("✓ Preprocessing pipeline works")


    # ---------------------------------------------------------
    # 6. Build machine-learning models
    # ---------------------------------------------------------

    models = build_models(preprocessor)

    assert "RandomForest" in models
    assert "SVR" in models
    assert "XGBoost" in models

    print("✓ Model construction works")


    # ---------------------------------------------------------
    # 7. Quick model training
    # ---------------------------------------------------------

    model = models["RandomForest"]

    model.fit(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert np.isfinite(predictions).all()

    print("✓ Random Forest training and prediction work")


    # ---------------------------------------------------------
    # 8. Test evaluation
    # ---------------------------------------------------------

    metrics = regression_metrics(y, predictions)

    assert "RMSE" in metrics
    assert "MSE" in metrics
    assert "R2" in metrics

    assert np.isfinite(metrics["RMSE"])
    assert np.isfinite(metrics["MSE"])
    assert np.isfinite(metrics["R2"])

    print("✓ Evaluation functions work")


    # ---------------------------------------------------------
    # 9. Final result
    # ---------------------------------------------------------

    print("\nQuick test completed successfully.")
    print("\nTest metrics:")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MSE:  {metrics['MSE']:.4f}")
    print(f"R²:   {metrics['R2']:.4f}")


if __name__ == "__main__":
    main()