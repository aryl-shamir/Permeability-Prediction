import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import learning_curve


def plot_learning_curve(model, X, y, title, cv=5):
    """
    Plot training and validation RMSE as a function
    of the training set size.
    """

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        train_sizes=np.linspace(
            0.1,
            1.0,
            6
        ),
        n_jobs=-1
    )

    train_rmse = -train_scores.mean(
        axis=1
    )

    val_rmse = -val_scores.mean(
        axis=1
    )

    plt.figure(
        figsize=(6, 4)
    )

    plt.plot(
        train_sizes,
        train_rmse,
        marker="o",
        label="Training RMSE"
    )

    plt.plot(
        train_sizes,
        val_rmse,
        marker="s",
        label="Validation RMSE"
    )

    plt.xlabel(
        "Training set size"
    )

    plt.ylabel(
        "RMSE"
    )

    plt.title(
        title
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.show()


def plot_correlation_with_logk(df):
    """
    Plot correlations between numerical features
    and logK.
    """

    corr_matrix = df.corr(
        numeric_only=True
    )

    corr_with_logk = (
        corr_matrix["logK"]
        .sort_values(
            ascending=False
        )
    )

    plt.figure(
        figsize=(8, 5)
    )

    corr_with_logk.plot(
        kind="bar"
    )

    plt.title(
        "Correlation of Features with logK"
    )

    plt.ylabel(
        "Correlation coefficient"
    )

    plt.xticks(
        rotation=45
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.7
    )

    plt.tight_layout()

    plt.show()


def plot_pairplot(df, variables):
    """
    Create a pairplot for selected variables.
    """

    sns.pairplot(
        df,
        vars=variables
    )

    plt.show()


def scatter_plot(
    actual_logk,
    pred_logk_list,
    model_names=None,
    save_path=None
):
    """
    Plot actual versus predicted log permeability
    for one or multiple models.
    """

    actual = actual_logk.to_numpy()

    n_models = len(
        pred_logk_list
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_models,
        figsize=(8 * n_models, 7),
        sharex=True,
        sharey=True,
        dpi=600
    )

    if n_models == 1:
        axes = [axes]

    for i, pred in enumerate(
        pred_logk_list
    ):

        residuals = np.abs(
            actual - pred
        )

        sns.scatterplot(
            x=actual,
            y=pred,
            hue=residuals,
            palette="coolwarm",
            alpha=1.0,
            s=100,
            edgecolors="black",
            linewidths=0.5,
            ax=axes[i],
            legend=False
        )

        min_val = min(
            actual.min(),
            pred.min()
        )

        max_val = max(
            actual.max(),
            pred.max()
        )

        axes[i].plot(
            [min_val, max_val],
            [min_val, max_val],
            color="red",
            linestyle="--",
            linewidth=2
        )

        title = (
            model_names[i]
            if model_names
            else f"Model {i + 1}"
        )

        axes[i].set_title(
            title,
            fontsize=20,
            fontweight="bold",
            pad=15
        )

        axes[i].set_xlabel(
            "Actual log(K)",
            fontsize=22,
            fontweight="bold",
            labelpad=12,
            color="black"
        )

        axes[i].set_ylabel(
            "Predicted log(K)",
            fontsize=22,
            fontweight="bold",
            labelpad=12,
            color="black"
        )

        axes[i].tick_params(
            axis="both",
            labelsize=20,
            width=2,
            length=6
        )

        for label in (
            axes[i].get_xticklabels()
            + axes[i].get_yticklabels()
        ):
            label.set_fontweight(
                "bold"
            )

        axes[i].grid(
            True,
            linewidth=0.6,
            alpha=0.6
        )

        for spine in axes[i].spines.values():
            spine.set_linewidth(
                2
            )

    plt.tight_layout(
        pad=3.0
    )

    if save_path is not None:

        plt.savefig(
            f"{save_path}.png",
            dpi=600,
            bbox_inches="tight"
        )

        plt.savefig(
            f"{save_path}.pdf",
            bbox_inches="tight"
        )

    plt.show()