"""
John F Wu (2021)

Scripts for quantifying the uncertainties, biases, and errors in satellite catalogs.
"""

import numpy as np
import pandas as pd
from easyquery import Query
from pathlib import Path

from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import binned_statistic_dd

from cnn_evaluation import load_saga_crossvalidation

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

ROOT = Path(__file__).resolve().parent.parent
results_dir = ROOT / "results/xSAGA"

# bin choices
r0_range = np.arange(13, 21.5, 1)
sb_range = np.arange(19, 26.5, 1)
gmr_range = np.arange(0.0, 0.95, 0.1)

r0_grid, sb_grid, gmr_grid = np.meshgrid(
    r0_range[:-1] + 0.5, sb_range[:-1] + 0.5, gmr_range[:-1] + 0.05
)

r0_bins = r0_range[:-1] + 0.5
sb_bins = sb_range[:-1] + 0.5
gmr_bins = gmr_range[:-1] + 0.05


def probability_host_alignments(
    hosts, delta_z=0.005, z_min=0.01, different_redshifts=True
):
    """Computes the probability that any two hosts at different (or same) redshifts
    have intersecting halos.

    Parameters
        hosts : pd.DataFrame
            The dataset of hosts
        delta_z : float
            The threshold for a host to be at a "different redshift" from another
        z_min : float
            The minimum redshift of a host. This affects host matching because the
            lowest-z hosts will subtend the largest angular areas.
        different_redshifts : bool
            Whether to return the fraction of hosts that are at different redshifts or
            at the same redshifts. Set this to False in order to estimate the fraction
            of truly overlapping hosts as opposed to the fraction of chance alignments.
    """

    q_cut = Query(f"z_NSA > {z_min}")
    hosts = q_cut.filter(hosts).copy()

    host_coords = SkyCoord(hosts.ra_NSA, hosts.dec_NSA, unit=u.deg)
    idx, angsep, _ = host_coords.match_to_catalog_sky(host_coords, nthneighbor=2)
    sep = (angsep * cosmo.kpc_proper_per_arcmin(hosts.iloc[idx].z_NSA)).to(u.kpc)

    hosts["sep_nearest_host"] = sep
    hosts["mass_nearest_host"] = hosts.iloc[idx].mass_GSE.values
    hosts["z_nearest_host"] = hosts.iloc[idx].z_NSA.values

    q1 = Query("sep_nearest_host < 300")
    q2 = Query("mass_GSE > mass_nearest_host")
    q3 = Query(f"abs(z_NSA - z_nearest_host) > {delta_z}")

    if different_redshifts:
        return (q1 & q2 & q3).count(hosts) / len(hosts)
    else:
        return (q1 & q2 & ~q3).count(hosts) / len(hosts)


def lowz_rate_model(params, r0, mu_eff, gmr):
    """Computes the low-z rate using a logistic model.

    Parameters:
        params : 5-tuple
            The model parameters `b0`, `b1`, `b2`, `b3`, and `Rmax`
        X : 2d array-like of shape (3, N_data)
            The data vector
    """

    b0, b1, b2, b3, Rmax = params

    ell = b0 + b1 * r0 + b2 * mu_eff + b3 * gmr

    return Rmax / (1 + np.exp(-ell))


def lnlikelihood(params, X, y):
    """Estimate the log likelihood for the lowz_rate_model.
    """

    if np.shape(X)[0] == 3:
        r0, mu_eff, gmr = X
    elif np.shape(X)[1] == 3:
        r0, mu_eff, gmr = np.transpose(X)
    else:
        raise ValueError("The data `X` must contain vectors [r0, mu_eff, gmr]")

    rates = lowz_rate_model(params, r0, mu_eff, gmr)

    rates = np.where(rates > 1e-8, rates, 1e-8)
    rates = np.where(rates < 1 - 1e-8, rates, 1 - 1e-8)

    return np.sum(np.where(y, np.log(rates), np.log(1 - rates)))


def fit_logistic_model(cv):
    """
    Fits the logistic regression model given some cross-validated results. The input
    `cv` must be a DataFrame containing the columns:
        r_mag : float
            extinction-corrected r magnitudes
        sb_r : float
            the extinction-corrected surface brightness using the r-band
            half-light radius
        gr : float
            extinction-corrected g-r colors
        low_z : bool
            whether or not the redshift is at low-z (z<0.03)
        p_CNN : float
            CNN probabilities that each source is at low-z
    """

    X = np.array([cv.r_mag, cv.sb_r, cv.gr]).T
    X_scaled = (X - X.mean(0)) / X.std(0)
    y_true = cv.low_z
    y_pred = cv.p_CNN > 0.5

    def _loss_func(*args):
        return -1 * lnlikelihood(*args)

    params_init = [0, 0, 0, 0, 0.8]
    bounds = np.array(4 * [[-np.inf, np.inf]] + [[0, 1]])

    lowz_params = dict()

    lowz_params["rate"] = minimize(
        _loss_func, params_init, args=(X_scaled, y_true), bounds=bounds
    ).x
    lowz_params["completeness"] = minimize(
        _loss_func, params_init, args=(X_scaled[y_true], y_pred[y_true]), bounds=bounds
    ).x
    lowz_params["purity"] = minimize(
        _loss_func, params_init, args=(X_scaled[y_pred], y_true[y_pred]), bounds=bounds
    ).x

    return lowz_params


def plot_model_fit_statistic(cv, params, statistic, figname="model_fit_lowz-stat.png"):
    """Given data, fit parameters, and some desired statistic (`rate`, `purity`,
    `completeness`, makes a plot)
    """

    # scale photometric parameters and corresponding bins
    X = np.array([cv.r_mag, cv.sb_r, cv.gr]).T
    X_scaled = (X - X.mean(0)) / X.std(0)
    y_true = cv.low_z
    y_pred = cv.p_CNN > 0.5

    bins_scaled = [
        (r0_range - X.mean(0)[0]) / X.std(0)[0],
        (sb_range - X.mean(0)[1]) / X.std(0)[1],
        (gmr_range - X.mean(0)[2]) / X.std(0)[2],
    ]

    # prepare plots
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=300, sharey=True)

    if statistic == "rate":
        X_ = X_scaled
        y_ = y_true
        color = "#58508d"
        label = "SAGA"
    elif statistic == "completeness":
        X_ = X_scaled[y_true]
        y_ = y_pred[y_true]
        color = "#ff6361"
        label = "CNN"
    elif statistic == "purity":
        X_ = X_scaled[y_pred]
        y_ = y_true[y_pred]
        color = "#ffa600"
        label = "CNN"
    else:
        raise ValueError("Statistic must be one of 'purity', 'completeness', or 'rate'")

    binned_statistic_result = binned_statistic_dd(X_, y_, bins=bins_scaled)
    lowz_rate = binned_statistic_result.statistic

    model_preds = lowz_rate_model(params, *X_.T)
    binned_statistic_result = binned_statistic_dd(X_, model_preds, bins=bins_scaled)
    model_rate = binned_statistic_result.statistic

    xbins = [r0_bins, sb_bins, gmr_bins]
    xlabels = [
        r"$r$ [mag]",
        r"$\mu_{r, \rm eff}$ [mag arcsec$^{-2}$]",
        r"$g-r$ [mag]",
    ]

    for i, (ax, xbins, xlabel) in enumerate(zip(axes.flat, xbins, xlabels)):
        axis = tuple(j for j in [0, 1, 2] if j != i)

        p_data = np.nanmean(lowz_rate, axis=axis)
        p_model = np.nanmean(model_rate, axis=axis)
        N = np.isfinite(model_rate).sum(axis=axis)

        def _wald_interval(p, N):
            return 2 * [np.sqrt(p * (1 - p) / N)]

        yerr = _wald_interval(p_data, N)
        width = 0.95 * (xbins[1] - xbins[0])

        ax.bar(xbins, p_data, width=width, alpha=0.5, color=color, label=label)
        ax.errorbar(xbins, p_data, yerr=yerr, ls="none", color=color)
        ax.plot(xbins, p_model, c="k", label="Model")

        ax.set_ylim(1e-2, 1)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.grid(color="white", lw=0.5, alpha=0.5)

    axes.flat[0].set_ylabel(f"Low-$z$ {statistic}", fontsize=12)
    axes.flat[0].legend(fontsize=12)
    fig.tight_layout()

    fig.savefig(results_dir / f"plots/cnn-evaluation/{figname}")


def correct_lowz_catalog(lowz, params):
    """Returns correction factor based on purity and completeness models based on
    photometric properties.

    WARNING: this ends up dominated by very low-completeness and low-purity objects, so
    it should not be used!

    Parameters
        lowz : pd.DataFrame
            Dataframe of low-z objects, which must contain the columns `r0`, `mu_eff`,
            or `gmr`. Can be sats or lowz catalog.
        params : dict
            Dictionary of best-fit model params returned by the function
            `fit_logistic_model()``. Must have keys `purity` and `completeness`.
    """

    X = np.array([lowz.r0, lowz.mu_eff, lowz.gmr]).T
    X_scaled = (X - X.mean(0)) / X.std(0)

    P = lowz_rate_model(params["purity"], *X_scaled.T)
    C = lowz_rate_model(params["completeness"], *X_scaled.T)

    P = np.where(P > 0.1, P, 0.1)
    C = np.where(P > 0.1, C, 0.1)

    # F = 1 + C**-1 - P**-1
    F = P / C
    F_err = np.sqrt((1 - C) ** 2 + (1 - P) ** 2)

    return F, F_err


def plot_purity_completeness_radial_trends(
    cv, saga_sats, saga_hosts, params, figname="purity_completeness_radial_trends"
):
    """Given the cross-validated data, SAGA satellite and host catalogs,
    and best-fit model parameters from above, plots the radial and magnitude dependence
    for candidates colored by the photometric model predictions for low-z rate,
    purity, and completeness.
    """
    cv_coords = SkyCoord(cv.RA, cv.DEC, unit="deg")
    s2_coords = SkyCoord(saga_sats.RA, saga_sats.DEC, unit="deg")

    idx, sep, _ = s2_coords.match_to_catalog_sky(cv_coords)

    df = pd.concat(
        [
            saga_sats.reset_index(),
            cv.iloc[idx][["Z", "low_z", "p_CNN", "r_mag", "sb_r", "gr"]].reset_index(),
        ],
        axis=1,
    )
    df = df[sep < 10 * u.arcsec].copy()

    predictions = dict()

    for hostname, host in saga_hosts.set_index("INTERNAL_HOSTID", drop=True).iterrows():
        host_coord = SkyCoord(host.RA, host.DEC, unit="deg")
        sep = host_coord.separation(cv_coords)
        max_angsep = ((0.3 / host.DIST) * u.rad).to(u.arcsec)
        sub_cv = cv[sep < max_angsep].copy()
        sub_cv["D_PROJ"] = sep[sep < max_angsep].to(u.rad).value * host.DIST * 1e3
        predictions[hostname] = sub_cv

    pred_df = pd.concat(predictions)

    # include scaled photometric quantities and compute model predictions
    pred_df["r_mag_scaled"] = (pred_df.r_mag - 19.30038351) / 1.63920353
    pred_df["sb_r_scaled"] = (pred_df.sb_r - 22.23829513) / 1.60764197

    pred_df["gr_scaled"] = (pred_df.gr - 0.97987122) / 1.16822956

    pred_df["p_model_rate"] = lowz_rate_model(
        params["rate"], pred_df.r_mag_scaled, pred_df.sb_r_scaled, pred_df.gr_scaled
    )

    pred_df["p_model_completeness"] = lowz_rate_model(
        params["completeness"],
        pred_df.r_mag_scaled,
        pred_df.sb_r_scaled,
        pred_df.gr_scaled,
    )

    pred_df["p_model_purity"] = lowz_rate_model(
        params["purity"], pred_df.r_mag_scaled, pred_df.sb_r_scaled, pred_df.gr_scaled
    )

    # make plots
    fig, axes = plt.subplots(
        3, 1, sharex=True, constrained_layout=True, figsize=(8, 6), dpi=300
    )

    is_lowz = Query("low_z == True")
    cnn_selected = Query("p_CNN > 0.5")

    TP = (is_lowz & cnn_selected).filter(pred_df)
    FP = (~is_lowz & cnn_selected).filter(pred_df)
    FN = (is_lowz & ~cnn_selected).filter(pred_df)

    for ax, metric, (df1, df2) in zip(
        axes.flat, ["rate", "completeness", "purity"], [(TP, FP), (TP, FN), (TP, FP)]
    ):

        ax.scatter(
            df2.D_PROJ,
            df2.r_mag,
            c=df2[f"p_model_{metric}"],
            marker="x",
            edgecolor="none",
            vmin=0,
            vmax=1,
            cmap="viridis_r",
        )

        sc = ax.scatter(
            df1.D_PROJ,
            df1.r_mag,
            c=df1[f"p_model_{metric}"],
            vmin=0,
            vmax=1,
            cmap="viridis_r",
            edgecolor="k",
            marker="o",
        )

        ax.set_xlim(15, 300)
        ax.set_ylim(12, 22)
        ax.set_ylabel("$r$ [mag]", fontsize=12)
        ax.grid(alpha=0.15)

        ax.text(18, 20.7, f"Modeled {metric}", fontsize=14)
    axes.flat[-1].set_xlabel("$D$ [pkpc]", fontsize=12)

    fig.colorbar(sc, ax=axes.flat, aspect=50, pad=-0.01)
    fig.savefig(results_dir / f"plots/cnn-evaluation/{figname}")


if __name__ == "__main__":

    # estimate halo chance alignments
    # ===============================

    # hosts = pd.read_parquet(results_dir / "hosts-nsa.parquet")
    # hosts_rand = pd.read_parquet(results_dir / "hosts_rand-nsa.parquet")
    #
    # p_host_chance = probability_host_alignments(hosts, different_redshifts=True)
    # p_host_overlap = probability_host_alignments(hosts, different_redshifts=False)
    # print(
    #     f"The fraction of chance overlaps in real hosts is {p_host_chance:.4f} and\n"
    #     f"the fraction same-redshift overlaps is {p_host_overlap:.4f}.\n"
    # )
    #
    # p_host_rand_chance = probability_host_alignments(
    #     hosts_rand, different_redshifts=True
    # )
    # p_host_rand_overlap = probability_host_alignments(
    #     hosts_rand, different_redshifts=False
    # )
    # print(
    #     f"The fraction of chance overlaps in random hosts is {p_host_rand_chance:.4f}\n"
    #     f"and the fraction at similar redshifts is {p_host_rand_overlap:.4f}.\n"
    # )

    # modeling low-z rate, purity, and completeness
    # =============================================
    cv = load_saga_crossvalidation()
    best_fit_params = fit_logistic_model(cv)

    # print(best_fit_params)

    # for statistic in ["rate", "completeness", "purity"]:
    #     plot_model_fit_statistic(
    #         cv,
    #         best_fit_params[statistic],
    #         statistic,
    #         figname=f"model_fit_lowz-{statistic}.png",
    #     )

    # # corrections to low-z
    # # ====================
    # sats = pd.read_parquet(results_dir / "sats-nsa_p0_5.parquet")
    # F, F_err = correct_lowz_catalog(sats, best_fit_params)

    # Checking radial trends
    # ======================

    saga2_sats = pd.read_csv(ROOT / "data/saga_stage2_sats.csv")
    saga2_hosts = pd.read_csv(ROOT / "data/saga_stage2_hosts.csv")
    plot_purity_completeness_radial_trends(cv, saga2_sats, saga2_hosts, best_fit_params)
