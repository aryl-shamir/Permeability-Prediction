import numpy as np

from sklearn.preprocessing import (
    StandardScaler,
    FunctionTransformer
)

from sklearn.pipeline import make_pipeline

from sklearn.impute import KNNImputer

from sklearn.compose import ColumnTransformer


def create_preprocessor():
    """
    Create the preprocessing pipeline used in the
    permeability prediction models.
    """

    log_pipeline = make_pipeline(
        KNNImputer(
            n_neighbors=5
        ),
        FunctionTransformer(
            lambda x: np.log10(x + 1e-6),
            feature_names_out="one-to-one"
        ),
        StandardScaler()
    )

    default_pipeline = make_pipeline(
        KNNImputer(
            n_neighbors=5
        ),
        StandardScaler()
    )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "scale",
                default_pipeline,
                [
                    "CAL",
                    "DT",
                    "GR",
                    "NPHI",
                    "RHOB",
                    "SP"
                ]
            ),
            (
                "log",
                log_pipeline,
                ["ILD"]
            )
        ],
        remainder="drop"
    )

    return preprocessing