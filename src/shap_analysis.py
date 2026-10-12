"""
shap_analysis.py

Generates SHAP (SHapley Additive exPlanations) values for the trained
XGBoost model, to explain which features drive its permeability
predictions, and in which direction.

Requires: pip install shap
"""

import shap
import matplotlib.pyplot as plt


def compute_shap_values(model, X_test_transformed):
    """
    Compute SHAP values for a trained tree-based model (XGBoost or RF).

    Parameters
    ----------
    model : fitted estimator
        The trained XGBoost (or RF) model.
    X_test_transformed : np.ndarray or pd.DataFrame
        Preprocessed test features to explain.

    Returns
    -------
    shap_values : np.ndarray
        SHAP values for the test set.
    explainer : shap.TreeExplainer
        The fitted explainer.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_transformed)
    return shap_values, explainer


def plot_summary(shap_values, X_test, save_path=None):
    """
    SHAP summary (beeswarm) plot: shows, for every feature, how much it
    affects the model's predictions (left-right spread) and whether high
    or low values of that feature push the prediction up or down (color).

    Parameters
    ----------
    shap_values : np.ndarray
        Output of compute_shap_values().
    X_test : pd.DataFrame
        The same test features passed to compute_shap_values(), used
        here to label the x-axis values and feature names.
    save_path : str, optional
        If given, saves the figure to this path (e.g. a .png file).
    """
    shap.summary_plot(shap_values, X_test, show=False)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()