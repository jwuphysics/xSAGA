"""
John F. Wu
2021-07-21

Scripts for evaluating CNN performance using SAGA data.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from easyquery import Query
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
results_dir = ROOT / "results/xSAGA"


def get_top_metrics(df, N=None, p_CNN_threshold=None, z_cutoff=0.03):
    """Return a dictionary of CNN metrics for a cross-validated dataset with a given
    p_CNN threshold (equivalent to the top `N` predictions).

    Parameters
        df : pd.DataFrame
            DataFrame containing the ground-truth redshifts in column `SPEC_Z` and
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

    df = df.sort_values(by="p_CNN", ascending=False).reset_index()

    q_true = Query(f"SPEC_Z < {z_cutoff}")

    if N is not None:
        q_pred = Query(f"index < {N}")
    elif p_CNN_threshold is not None:
        q_pred = Query(f"p_CNN > p_CNN_threshold")
    else:
        raise ValueError("Either `N` or `p_CNN_threshold` must be provided.")

    TP = (q_pred & q_true).count(df)
    TN = (~q_pred & ~q_true).count(df)
    FP = (q_pred & ~q_true).count(df)
    FN = (~q_pred & q_true).count(df)

    evaluation_metrics = {}
    evaluation_metrics["accuracy"] = (TP + TN) / (TP + TN + FP + FN)
    evaluation_metrics["completeness"] = TP / (TP + FN)
    evaluation_metrics["purity"] = TP / (TP + FP)

    evaluation_metrics["geometric_mean"] = np.sqrt(
        evaluation_metrics["completeness"] * evaluation_metrics["purity"]
    )

    return evaluation_metrics


def plot_metrics(df, surface_area, figname="CNN_metrics.png"):
    """Plot evaluation metrics for cross-validated dataset that spans a given
    surface area. See `get_top_metrics()` for more details.

    Parameters
        df : pd.DataFrame
            DataFrame containing the ground-truth redshifts in column `SPEC_Z` and
            low-z predictions in column `p_CNN`.
        surface_area : float [deg^2]
            The area spanned by the data set, in units of square degrees.
        figname : str
            The name of the resulting figure, which will be stored in
            `{results_dir}/plots/`.
    """

    metrics_dict = {}

    for surface_density in np.arange(1, 200, 1):
        N = np.round(surface_density * surface_area).astype(int)
        evaluation_metrics = get_top_metrics(df, N)
        metrics_dict[surface_density] = evaluation_metrics

    all_metrics_df = pd.DataFrame.from_dict(metrics_dict, orient="index").rename_axis(
        "surface_density"
    )

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

    surface_densities = [1, 2, 3, 5, 10, 15, 20, 30, 50, 100]
    ax.set_xticks(surface_densities)
    ax.set_xticklabels(surface_densities)
    ax.set_xscale("log")

    ax.set_ylim(0, 1)
    ax.set_xlabel("Surface density [deg$^{-2}$]", fontsize=12)
    ax.set_ylabel("Evaluation metric")

    ax.grid(alpha=0.15)
    ax.legend(framealpha=0, loc="lower center")

    fig.savefig(results_dir / f"plots/{figname}")
    

def compare_north_and_south():
    """Compare p_CNN thresholds in the Legacy Survey North and South, which have
    different imaging systematics.

    """
    pass


def compare_magnitude(predictions, labels, r):
    """Compare metrics as a function of apparent magnitude.
    """
    pass


def compare_surface_brightness(predictions, labels, mu_eff):
    pass


def compare_color(predictions, labels, gmr):
    pass


if __name__ == "__main__":
    pass
