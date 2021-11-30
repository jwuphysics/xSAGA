"""
John F. Wu (2021)

Scripts for evaluating CNN performance using SAGA data.
"""

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.cosmology import FlatLambdaCDM
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

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def load_saga_crossvalidation(catalogname="saga-validation_FL-hdxresnet34.csv"):
    """Load the SAGA cross validation using the FL-hdxresnet34 model on 144x144 images,
    generated on 2021-08-03. Note that columns are renamed here so as to be consistent
    with the rest of the code base.
    """

    saga_cv_file = results_dir / f"cnn-training/{catalogname}"
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
    df,
    X,
    X_min,
    X_max,
    delta_X,
    X_label,
    figname,
    z_thresh=0.03,
    p_cnn_thresh=0.5,
    N_boot=100,
    label="fraction",
):
    """Compare metrics as a function of column X.

    If N_boot is an integer, then it will bootstrap resample each metric `N_boot` times.

    The param `label` can be either "count" or "fraction", depending on whether the
    total number or fraction of low-z galaxies should be labeled at the top of the
    figure.
    """

    q_true = Query(f"Z < {z_thresh}")
    q_pred = Query(f"p_CNN > {p_cnn_thresh}")

    X_bins = np.arange(X_min, X_max, delta_X)
    X_queries = [
        Query(f"{X} >= {x1}", f"{X} < {x2}") for x1, x2 in zip(X_bins, X_bins + delta_X)
    ]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=300)

    # get the True/False mask for labels and predictions for each q (magnitude bin)
    def completeness_score_bootfunc(x):
        return recall_score(q_true.mask(x), q_pred.mask(x))

    def purity_score_bootfunc(x):
        return precision_score(q_true.mask(x), q_pred.mask(x))

    boot_completeness = np.array(
        [
            [
                completeness_score_bootfunc(q.filter(df).sample(frac=1, replace=True))
                for _ in range(N_boot)
            ]
            for q in X_queries
        ]
    )

    boot_purity = np.array(
        [
            [
                purity_score_bootfunc(q.filter(df).sample(frac=1, replace=True))
                for _ in range(N_boot)
            ]
            for q in X_queries
        ]
    )

    boot_geometric_mean = np.sqrt(boot_completeness * boot_purity)

    ax.plot(
        X_bins + delta_X / 2,
        boot_completeness.mean(1),
        c="#ff6361",
        lw=2,
        label="Completeness",
        zorder=1,
    )
    ax.fill_between(
        X_bins + delta_X / 2,
        *np.quantile(boot_completeness, [0.16, 0.84], axis=1),
        color="#ff6361",
        lw=0,
        zorder=1,
        alpha=0.3,
    )

    ax.plot(
        X_bins + delta_X / 2,
        boot_purity.mean(1),
        c="#ffa600",
        lw=2,
        label="Purity",
        zorder=2,
    )
    ax.fill_between(
        X_bins + delta_X / 2,
        *np.quantile(boot_purity, [0.16, 0.84], axis=1),
        color="#ffa600",
        lw=0,
        zorder=2,
        alpha=0.3,
    )

    ax.plot(
        X_bins + delta_X / 2,
        boot_geometric_mean.mean(1),
        c="#003f5c",
        lw=2,
        label="Geometric mean",
        zorder=3,
    )
    ax.fill_between(
        X_bins + delta_X / 2,
        *np.quantile(boot_geometric_mean, [0.16, 0.84], axis=1),
        color="#003f5c",
        lw=0,
        zorder=3,
        alpha=0.3,
    )
    if label == "fraction":
        fracs = np.array([q_true.count(q.filter(df)) / q.count(df) for q in X_queries])
        for x, l in zip(X_bins + delta_X / 2, fracs):
            ax.text(x, 1.08, f"{l:.3f}", rotation=60, fontsize=10, color="k")
    elif label == "count":
        counts = np.array([q_true.count(q.filter(df)) for q in X_queries])
        for x, l in zip(X_bins + delta_X / 2, counts):
            ax.text(x, 1.08, l, rotation=60, fontsize=10, color="k")
    else:
        raise ValueError("Please specify a valid `label`.")

    ax.set_xlabel(X_label, fontsize=16)
    ax.set_ylim(-0.06, 1.06)
    # ax.set_ylabel("Evaluation metric", fontsize=14)
    ax.tick_params(which="both", labelsize=12)

    ax.grid(alpha=0.15)
    ax.legend(
        framealpha=0,
        loc="lower left" if X not in ("sb_r", "DEC") else "lower right",
        fontsize=14,
    )

    fig.tight_layout()
    fig.savefig(results_dir / f"plots/cnn-evaluation/{figname}")


def plot_roc_curve(
    df,
    K=4,
    color="#003f5c",
    label="FL-hdxresnet34",
    fig=None,
    ax=None,
    figname="roc-curve.png",
):
    """Plot one or more ROC curves.
    """
    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)

    for k in range(K):
        k = k + 1
        q = Query(f"kfold == {k}")

        fpr, tpr, thresholds = roc_curve(q.filter(df).low_z, q.filter(df).p_CNN)
        ax.plot(fpr, tpr, c=color, lw=1, alpha=0.5, zorder=1)

    # plot mean trend and mean scores
    score = roc_auc_score(df.low_z, df.p_CNN)
    fpr, tpr, thresholds = roc_curve(df.low_z, df.p_CNN)

    ax.plot(fpr, tpr, c=color, lw=3, zorder=5, label=f"{label}\n(AUC = {score:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)

    ax.grid(alpha=0.15)
    ax.legend(loc="lower right", fontsize=12)

    if figname is None:
        return fig, ax
    else:
        ax.plot([0, 1], [0, 1], ls="--", lw=1, c="k")
        fig.tight_layout()
        fig.savefig(results_dir / f"plots/cnn-evaluation/{figname}")


if __name__ == "__main__":

    # load SAGA crossvalidation results
    # =================================
    saga_cv = load_saga_crossvalidation()
    saga_cv["ranking"] = saga_cv.p_CNN.rank(ascending=False)
    saga_cv["M_r"] = saga_cv.r_mag - cosmo.distmod(z=saga_cv.Z)

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
        X_label=r"$r$ [mag]",
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
        X_min=-0.05,
        X_max=0.9,
        delta_X=0.1,
        X_label=r"$g-r$ [mag]",
        figname="gmr-comparison.png",
    )

    plot_comparison_by_X(
        saga_cv,
        X="rz",
        X_min=-0.2,
        X_max=0.8,
        delta_X=0.1,
        X_label=r"$r-z$ [mag]",
        figname="rmz-comparison.png",
    )

    plot_comparison_by_X(
        saga_cv,
        X="M_r",
        X_min=-21,
        X_max=-12,
        delta_X=0.5,
        X_label=r"$M_{r}$ [mag]",
        figname="M_r-comparison.png",
    )

    plot_comparison_by_X(
        saga_cv,
        X="Z",
        X_min=0,
        X_max=0.03,
        delta_X=0.005,
        X_label=r"$z$",
        figname="redshift-comparison.png",
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

    # Receiving Operator Characteristic curve
    # ---------------------------------------

    # start with p_sat from SAGA II
    # saga_psat = saga_cv.copy()
    # saga_psat["p_CNN"] = saga_psat.p_sat_approx
    # fig, ax = plot_roc_curve(
    #     saga_psat, color="#7a5195", label=r"SAGA $p_{\rm sat}$", figname=None
    # )
    #
    # # get resnet cross-validation and add to ROC curve
    # saga_resnet_cv = load_saga_crossvalidation(
    #     catalogname="saga-validation_resnet34.csv"
    # )
    # fig, ax = plot_roc_curve(
    #     saga_resnet_cv, color="#ef5675", label="resnet34", fig=fig, ax=ax, figname=None
    # )
    #
    # # finally include highly optimized CNN
    # plot_roc_curve(
    #     saga_cv,
    #     color="#ffa600",
    #     label="FL-hdxresnet34",
    #     fig=fig,
    #     ax=ax,
    #     figname="roc-curve.png",
    # )
