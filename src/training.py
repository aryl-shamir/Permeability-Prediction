from sklearn.model_selection import cross_val_score


def cross_validate_model(model, X, y, cv=5):
    """
    Perform k-fold cross-validation using RMSE.

    Parameters
    ----------
    model : estimator
        Scikit-learn compatible model or pipeline.

    X : pandas.DataFrame
        Training features.

    y : pandas.Series
        Training target.

    cv : int, default=5
        Number of cross-validation folds.

    Returns
    -------
    numpy.ndarray
        RMSE score for each fold.
    """

    scores = -cross_val_score(
        model,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1
    )

    return scores


def train_model(model, X, y):
    """
    Fit a model on the training data.

    Parameters
    ----------
    model : estimator
        Model or pipeline to train.

    X : pandas.DataFrame
        Training features.

    y : pandas.Series
        Training target.

    Returns
    -------
    estimator
        Fitted model.
    """

    model.fit(
        X,
        y
    )

    return model