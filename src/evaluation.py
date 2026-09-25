from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    root_mean_squared_error
)


def regression_metrics(y_true, y_pred):
    """
    Calculate regression performance metrics.

    Parameters
    ----------
    y_true : array-like
        Actual target values.

    y_pred : array-like
        Predicted target values.

    Returns
    -------
    dict
        MSE, RMSE, and R2 values.
    """

    return {
        "RMSE": root_mean_squared_error(
            y_true,
            y_pred
        ),

        "MSE": mean_squared_error(
            y_true,
            y_pred
        ),

        "R2": r2_score(
            y_true,
            y_pred
        )
    }