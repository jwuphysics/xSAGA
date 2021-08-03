"""
John F. Wu
2021-07-21

Scripts for evaluating CNN performance using SAGA data.
"""

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from easyquery import Query
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
results_dir = ROOT / "results/xSAGA"


def load_saga_crossvalidation():
    """Load the SAGA cross validation using the FL-hdxresnet34 model on 144x144 images,
    generated on 2021-08-03. Note that columns are renamed here so as to be consistent
    with the rest of the code base.
    """

    saga_cv_file = results_dir / "cnn-training/saga-validation_FL-hdxresnet34.csv"
    df = pd.read_csv(saga_cv_file, dtype={"OBJID": str})

    # rename columns `SPEC_Z` and `pred_low_z`
    df = df.rename({"SPEC_Z": "Z", "pred_low_z": "p_CNN"}, axis=1)

    return df


def get_top_metrics(df, N=None, p_CNN_threshold=None, z_cutoff=0.03):
    """Return a dictionary of CNN metrics for a cross-validated dataset with a given
    p_CNN threshold (equivalent to the top `N` predictions).

    Parameters
        df : pd.DataFrame
            DataFrame containing the ground-truth redshifts in column `Z` and
            low-z predictions in column `p_CNN`.
        N : int or None
            The number of predictions to evaluate; this directly corresponds to a
            `p_CNN_threshold`. Either `N` or `p_CNN_threshold` must be provided, but a
            given `N` will supercede `p_CNN_threshold`.
        p_CNN_threshold : float in [0, 1] or None
            The `p_CNN` threshold that determines the subset of predictions to
            evaluate. Alternatively, `N = sum(df.p_CNN > p_CNN_threshold)` can also
            be provided.
        z_cutoff : float
            The redshift used to define "low-redshift".

    Returns
        evaluation_metrics : dict
            A dictionary with keys `accuracy`, `completeness`, `purity`, and
            the `geometric_mean` of the purity and completeness.
    """

    if "ranking" not in df.columns:
        df["ranking"] = df.p_CNN.rank(ascending=False)

    q_true = Query(f"Z < {z_cutoff}")

    if N is not None:
        q_pred = Query(f"ranking < {N}")
    elif p_CNN_threshold is not None:
        q_pred = Query(f"p_CNN > {p_CNN_threshold}")
    else:
        raise ValueError("Either `N` or `p_CNN_threshold` must be provided.")

    TP = (q_pred & q_true).count(df)
    TN = (~q_pred & ~q_true).count(df)
    FP = (q_pred & ~q_true).count(df)
    FN = (~q_pred & q_true).count(df)

    evaluation_metrics = {}
    evaluation_metrics["accuracy"] = (TP + TN) / (TP + TN + FP + FN)
    try:
        evaluation_metrics["completeness"] = TP / (TP + FN)
        evaluation_metrics["purity"] = TP / (TP + FP)

        evaluation_metrics["geometric_mean"] = np.sqrt(
            evaluation_metrics["completeness"] * evaluation_metrics["purity"]
        )
    except ZeroDivisionError:
        pass

    return evaluation_metrics


def plot_metrics(
    df, label_surface_density=False, surface_area=None, figname="CNN_metrics.png"
):
    """Plot evaluation metrics for cross-validated dataset that spans a given
    surface area. See `get_top_metrics()` for more details.

    Parameters
        df : pd.DataFrame
            DataFrame containing the ground-truth redshifts in column `Z` and
            low-z predictions in column `p_CNN`.
        label_surface_density : bool
            Whether to use the target surface density as the x-axis, or `p_CNN`.
        surface_area : float [deg^2]
            The area spanned by the data set, in units of square degrees.
        figname : str
            The name of the resulting figure, which will be stored in
            `{results_dir}/plots/`.
    """

    metrics_dict = {}

    if label_surface_density:
        for surface_density in np.arange(1, 200, 1):
            if surface_area is None:
                raise ValueError(
                    """Supply either a surface area, or set `label_surface_density` \
                    to False.
                    """
                )
            N = np.round(surface_density * surface_area).astype(int)
            evaluation_metrics = get_top_metrics(df, N=N)
            metrics_dict[surface_density] = evaluation_metrics
        all_metrics_df = pd.DataFrame.from_dict(
            metrics_dict, orient="index"
        ).rename_axis("surface_density")
    else:
        for p_CNN_threshold in np.arange(0, 1, 0.01):
            evaluation_metrics = get_top_metrics(df, p_CNN_threshold=p_CNN_threshold)
            metrics_dict[p_CNN_threshold] = evaluation_metrics
        all_metrics_df = pd.DataFrame.from_dict(
            metrics_dict, orient="index"
        ).rename_axis("p_CNN_threshold")

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
    ax.plot(
        all_metrics_df.index,
        all_metrics_df.completeness,
        c="#ff6361",
        lw=3,
        label="Completeness",
    )
    ax.plot(
        all_metrics_df.index, all_metrics_df.purity, c="#ffa600", lw=3, label="Purity"
    )

    ax.plot(
        all_metrics_df.index,
        all_metrics_df.geometric_mean,
        c="#003f5c",
        lw=3,
        ls="-",
        label="Geometric mean",
    )

    if label_surface_density:
        ax.set_xlabel("Surface density [deg$^{-2}$]", fontsize=12)
        surface_densities = [1, 2, 3, 5, 10, 15, 20, 30, 50, 100]
        ax.set_xscale("log")
        ax.set_xticks(surface_densities)
        ax.set_xticklabels(surface_densities)
    else:
        ax.set_xlabel(r"$p_{\rm CNN}$", fontsize=12)
        ax.set_xscale("linear")

    ax.set_ylim(0, 1)
    ax.set_ylabel("Evaluation metric")

    ax.grid(alpha=0.15)
    ax.legend(framealpha=0, loc="lower center")

    fig.tight_layout()
    fig.savefig(results_dir / f"plots/cnn-evaluation/{figname}")


def plot_metrics_multi(
    df,
    K=4,
    label_surface_density=False,
    surface_area=None,
    figname="CNN_metrics-multi_k.png",
):
    """Plot evaluation metrics for cross-validated dataset that spans a given
    surface area. See `get_top_metrics()` for more details.

    Parameters
        df : pd.DataFrame
            DataFrame containing the ground-truth redshifts in column `Z` and
            low-z predictions in column `p_CNN`.
        label_surface_density : bool
            Whether to use the target surface density as the x-axis, or `p_CNN`.
        surface_area : float [deg^2]
            The area spanned by the data set, in units of square degrees.
        figname : str
            The name of the resulting figure, which will be stored in
            `{results_dir}/plots/`.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)

    metrics_dict = {}

    for k in range(K):
        k = k + 1
        dfk = Query(f"kfold == {k}").filter(df)

        if label_surface_density:
            for surface_density in np.arange(1, 200, 1):
                if surface_area is None:
                    raise ValueError(
                        """Supply either a surface area, or set `label_surface_density` \
                        to False.
                        """
                    )
                N = np.round(surface_density * surface_area).astype(int)
                evaluation_metrics = get_top_metrics(dfk, N=N)
                metrics_dict[surface_density] = evaluation_metrics
            all_metrics_df = pd.DataFrame.from_dict(
                metrics_dict, orient="index"
            ).rename_axis("surface_density")
        else:
            for p_CNN_threshold in np.arange(0, 1, 0.01):
                evaluation_metrics = get_top_metrics(
                    dfk, p_CNN_threshold=p_CNN_threshold
                )
                metrics_dict[p_CNN_threshold] = evaluation_metrics
            all_metrics_df = pd.DataFrame.from_dict(
                metrics_dict, orient="index"
            ).rename_axis("p_CNN_threshold")

        ax.plot(
            all_metrics_df.index,
            all_metrics_df.completeness,
            c="#ff6361",
            lw=2,
            label="Completeness" if k == K else "",
            zorder=1,
        )
        ax.plot(
            all_metrics_df.index,
            all_metrics_df.purity,
            c="#ffa600",
            lw=2,
            label="Purity" if k == K else "",
            zorder=2,
        )

        ax.plot(
            all_metrics_df.index,
            all_metrics_df.geometric_mean,
            c="#003f5c",
            lw=2,
            ls="-",
            label="Geometric mean" if k == K else "",
            zorder=3,
        )

    if label_surface_density:
        ax.set_xlabel("Surface density [deg$^{-2}$]", fontsize=12)
        surface_densities = [1, 2, 3, 5, 10, 15, 20, 30, 50, 100]
        ax.set_xscale("log")
        ax.set_xticks(surface_densities)
        ax.set_xticklabels(surface_densities)
    else:
        ax.set_xlabel(r"$p_{\rm CNN}$", fontsize=12)
        ax.set_xscale("linear")

    ax.set_ylim(0, 1)
    ax.set_ylabel("Evaluation metric", fontsize=12)

    ax.grid(alpha=0.15)
    ax.legend(framealpha=0, loc="lower center", fontsize=12)

    fig.tight_layout()
    fig.savefig(results_dir / f"plots/cnn-evaluation/{figname}")


def plot_confusion_matrix(
    df,
    p_cnn_thresh=0.5,
    normalize="true",
    z_thresh=0.03,
    cmap="Blues",
    figname="confusion-matrix_0p5.png",
):
    """Save a plot of the confusion matrix for SAGA cross-validation results.

    Some code has been adapted from a previous version of
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """

    if isinstance(normalize, bool):
        normalize = "true" if True else None

    cm = confusion_matrix(
        Query(f"Z < {z_thresh}").mask(df),
        Query(f"p_CNN > {p_cnn_thresh}").mask(df),
        normalize=normalize,
    )

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)

    ax.imshow(cm, interpolation="nearest", cmap=cmap)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            ax.text(
                j,
                i,
                "{:0.4f}".format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
        else:
            ax.text(
                j,
                i,
                "{:,}".format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["high-z", "low-z"])
    ax.set_yticklabels(["high-z", "low-z"])
    ax.set_ylabel("True label", fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    fig.tight_layout()

    fig.savefig(results_dir / f"plots/cnn-evaluation/{figname}")


def compare_north_and_south():
    """Compare p_CNN thresholds in the Legacy Survey North and South, which have
    different imaging systematics.

    """
    pass


def plot_comparison_by_X(
    df, X, X_min, X_max, delta_X, X_label, figname, K=4, z_thresh=0.03, p_cnn_thresh=0.5, count=True
):
    """Compare metrics as a function of column X.
    """

    q_true = Query(f"Z < {z_thresh}")
    q_pred = Query(f"p_CNN > {p_cnn_thresh}")

    X_bins = np.arange(X_min, X_max, delta_X)
    X_queries = [
        Query(f"{X} >= {x1}", f"{X} < {x2}") for x1, x2 in zip(X_bins, X_bins + delta_X)
    ]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=300)

    completenesses = []
    purities = []
    for k in range(K):
        k = k + 1

        dfk = Query(f"kfold == {k}").filter(df)

        # get the True/False mask for labels and predictions for each q (magnitude bin)
        completeness = np.array(
            [
                recall_score(q_true.mask(q.filter(dfk)), q_pred.mask(q.filter(dfk)))
                for q in X_queries
            ]
        )

        purity = np.array(
            [
                precision_score(q_true.mask(q.filter(dfk)), q_pred.mask(q.filter(dfk)))
                for q in X_queries
            ]
        )

        completenesses.append(completeness)
        purities.append(purity)

    completenesses = np.array(completenesses)
    purities = np.array(purities)
    geometric_means = np.sqrt(completenesses * purities)

    ax.plot(
        X_bins + delta_X / 2,
        completenesses.mean(0),
        c="#ff6361",
        lw=2,
        label="Completeness" if k == K else "",
        zorder=1,
    )
    ax.fill_between(
        X_bins + delta_X / 2,
        completenesses.mean(0) - completenesses.std(0),
        completenesses.mean(0) + completenesses.std(0),
        color="#ff6361",
        lw=0,
        zorder=1,
        alpha=0.3,
    )

    ax.plot(
        X_bins + delta_X / 2,
        purities.mean(0),
        c="#ffa600",
        lw=2,
        label="Purity" if k == K else "",
        zorder=2,
    )
    ax.fill_between(
        X_bins + delta_X / 2,
        purities.mean(0) - purities.std(0),
        purities.mean(0) + purities.std(0),
        color="#ffa600",
        lw=0,
        zorder=2,
        alpha=0.3,
    )

    ax.plot(
        X_bins + delta_X / 2,
        geometric_means.mean(0),
        c="#003f5c",
        lw=2,
        label="Geometric mean" if k == K else "",
        zorder=3,
    )
    ax.fill_between(
        X_bins + delta_X / 2,
        geometric_means.mean(0) - geometric_means.std(0),
        geometric_means.mean(0) + geometric_means.std(0),
        color="#003f5c",
        lw=0,
        zorder=3,
        alpha=0.3,
    )
    if count:
        counts = np.array([q_true.count(q.filter(df)) for q in X_queries])
        for x, c in zip(X_bins + delta_X / 2, counts):
            ax.text(x, 1.02, c, rotation=60, fontsize=8, color='k')

    ax.set_xlabel(X_label, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Evaluation metric", fontsize=12)

    ax.grid(alpha=0.15)
    ax.legend(
        framealpha=0,
        loc="lower left" if X not in ("sb_r", 'DEC') else "lower right",
        fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(results_dir / f"plots/cnn-evaluation/{figname}")


if __name__ == "__main__":

    # load SAGA crossvalidation results
    # =================================
    saga_cv = load_saga_crossvalidation()
    saga_cv["ranking"] = saga_cv.p_CNN.rank(ascending=False)

    # N_hosts = 89
    # saga_area = N_hosts * np.pi * 1 ** 2
    # plot_metrics(saga_cv, label_surface_density=True, surface_area=saga_area)

    # plot_metrics(saga_cv, label_surface_density=False, figname="saga_all.png")
    # plot_metrics_multi(
    #     saga_cv, K=4, label_surface_density=False, figname="saga_all-multi_k.png"
    # )

    # plot_metrics(
    #     saga_cv[saga_cv.r_mag < 19],
    #     label_surface_density=False,
    #     figname="saga_r-below-19.png",
    # )
    #
    # plot_metrics(
    #     saga_cv[saga_cv.r_mag > 19],
    #     label_surface_density=False,
    #     figname="saga_r-above-19.png",
    # )

    # plot_confusion_matrix(saga_cv, p_cnn_thresh=0.5, figname="confusion-matrix.png")
    # plot_confusion_matrix(saga_cv, p_cnn_thresh=0.4, figname="confusion-matrix_0p4.png")

    plot_comparison_by_X(
        saga_cv,
        X="r_mag",
        X_min=14,
        X_max=21,
        delta_X=0.5,
        X_label=r"$r_0$ [mag]",
        figname="magnitude-comparison.png",
    )

    plot_comparison_by_X(
        saga_cv,
        X="sb_r",
        X_min=20,
        X_max=26,
        delta_X=0.5,
        X_label=r"$\mu_{r,\rm eff}$ [mag arcsec$^{-2}$]",
        figname="surface_brightness-comparison.png",
    )

    plot_comparison_by_X(
        saga_cv,
        X="gr",
        X_min=0,
        X_max=0.9,
        delta_X=0.1,
        X_label=r"$(g-r)_0$ [mag]",
        figname="gmr-comparison.png",
    )

    # note that these plots will raise warnings because RA/Dec are very unevenly
    # distributed, and the kfolds may contain zero ground truths -- resulting in
    # divide by zero errors (or UndefinedMetricWarning).
    # plot_comparison_by_X(
    #     saga_cv,
    #     X="RA",
    #     X_min=120,
    #     X_max=240,
    #     delta_X=10,
    #     X_label=r"$RA [deg]",
    #     figname="RA-comparison.png",
    # )
    # plot_comparison_by_X(
    #     saga_cv,
    #     X="DEC",
    #     X_min=-30,
    #     X_max=60,
    #     delta_X=10,
    #     X_label=r"$Dec [deg]",
    #     figname="DEC-comparison.png",
    # )
