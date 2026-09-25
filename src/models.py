from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from xgboost import XGBRegressor


def build_models(preprocessor):
    """
    Build the baseline Random Forest, SVR,
    and XGBoost regression models.
    """

    models = {

        "RandomForest": make_pipeline(
            preprocessor,
            RandomForestRegressor(
                random_state=42
            )
        ),

        "SVR": make_pipeline(
            preprocessor,
            SVR()
        ),

        "XGBoost": make_pipeline(
            preprocessor,
            XGBRegressor(
                objective="reg:squarederror",
                random_state=42
            )
        )

    }

    return models