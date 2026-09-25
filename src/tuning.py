from scipy.stats import loguniform, uniform

from sklearn.model_selection import RandomizedSearchCV

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from xgboost import XGBRegressor


def tune_random_forest(preprocessor):
    """
    Create the RandomizedSearchCV object used
    for Random Forest hyperparameter tuning.
    """

    full_pipeline = Pipeline([
        (
            "preprocessing",
            preprocessor
        ),
        (
            "rf_reg",
            RandomForestRegressor(
                random_state=42
            )
        )
    ])

    param_dist = {
        "rf_reg__n_estimators": [
            60,
            70
        ],

        "rf_reg__max_depth": [
            3,
            4,
            8
        ],

        "rf_reg__min_samples_split": [
            2,
            4,
            5
        ],

        "rf_reg__min_samples_leaf": [
            1,
            2
        ],

        "rf_reg__max_features": [
            "sqrt",
            "log2",
            None
        ],

        "rf_reg__bootstrap": [
            True,
            False
        ]
    }

    random_search = RandomizedSearchCV(
        estimator=full_pipeline,
        param_distributions=param_dist,
        cv=5,
        n_iter=30,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    return random_search


def tune_xgboost(preprocessor):
    """
    Create the RandomizedSearchCV object used
    for XGBoost hyperparameter tuning.
    """

    full_pipeline = Pipeline([
        (
            "preprocessing",
            preprocessor
        ),
        (
            "xgb_reg",
            XGBRegressor(
                objective="reg:squarederror",
                random_state=42
            )
        )
    ])

    param_grid_xgb = {
        "xgb_reg__n_estimators": [
            100,
            200
        ],

        "xgb_reg__max_depth": [
            3,
            4,
            5
        ],

        "xgb_reg__learning_rate": [
            0.01,
            0.05,
            0.1
        ],

        "xgb_reg__subsample": [
            0.8,
            1.0
        ],

        "xgb_reg__colsample_bytree": [
            0.8,
            1.0
        ],

        "xgb_reg__min_child_weight": [
            4,
            5,
            6
        ]
    }

    random_search = RandomizedSearchCV(
        estimator=full_pipeline,
        param_distributions=param_grid_xgb,
        cv=5,
        scoring="neg_mean_squared_error",
        verbose=1,
        n_jobs=-1
    )

    return random_search


def tune_svr(preprocessor):
    """
    Create the RandomizedSearchCV object used
    for SVR hyperparameter tuning.
    """

    full_pipeline = Pipeline([
        (
            "preprocessing",
            preprocessor
        ),
        (
            "svr_reg",
            SVR()
        )
    ])

    param_distrib = {
        "svr_reg__gamma": loguniform(
            0.001,
            0.1
        ),

        "svr_reg__C": uniform(
            1,
            10
        )
    }

    random_search = RandomizedSearchCV(
        estimator=full_pipeline,
        param_distributions=param_distrib,
        cv=5,
        random_state=42,
        n_iter=100
    )

    return random_search