"""John F Wu
2021-08-03

Scripts for quantifying the uncertainties, biases, and errors in satellite catalogs.
"""

import numpy as np
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

    X = np.array([cv.r_mag, cv.sb_r, cv.gr]).T
    X_scaled = (X - X.mean(0)) / X.std(0)
    y_true = cv.low_z
    y_pred = cv.p_CNN > 0.5

    bins_scaled = [
        (r0_range - X.mean(0)[0]) / X.std(0)[0],
        (sb_range - X.mean(0)[1]) / X.std(0)[1],
        (gmr_range - X.mean(0)[2]) / X.std(0)[2],
    ]

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
        label = "xSAGA"
    elif statistic == "purity":
        X_ = X_scaled[y_pred]
        y_ = y_true[y_pred]
        color = "#ffa600"
        label = "xSAGA"
    else:
        raise ValueError("Statistic must be one of 'purity', 'completeness', or 'rate'")

    binned_statistic_result = binned_statistic_dd(X_, y_, bins=bins_scaled)
    lowz_rate = binned_statistic_result.statistic

    model_preds = lowz_rate_model(params, *X_.T)
    binned_statistic_result = binned_statistic_dd(X_, model_preds, bins=bins_scaled)
    model_rate = binned_statistic_result.statistic

    xbins = [r0_bins, sb_bins, gmr_bins]
    xlabels = [
        r"$r_0$ [mag]",
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
        ax.plot(xbins, p_model, c="k", label="model")

        ax.set_ylim(1e-2, 1)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.grid(color="white", lw=0.5, alpha=0.5)

    axes.flat[0].set_ylabel(f"Low-$z$ {statistic}", fontsize=12)
    axes.flat[0].legend(fontsize=12)
    fig.tight_layout()

    fig.savefig(results_dir / f"plots/cnn-evaluation/{figname}")


if __name__ == "__main__":
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
    cv = load_saga_crossvalidation()
    best_fit_params = fit_logistic_model(cv)

    print(best_fit_params)

    for statistic in ["rate", "completeness", "purity"]:
        plot_model_fit_statistic(
            cv,
            best_fit_params[statistic],
            statistic,
            figname=f"model_fit_lowz-{statistic}.png",
        )
