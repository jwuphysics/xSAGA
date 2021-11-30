"""
John F. Wu (2021)

Scripts for creating figures of large scale structure.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
from astroML.correlation import bootstrap_two_point_angular
from astropy.cosmology import FlatLambdaCDM
from astropy.stats import bootstrap
import astropy.units as u
import cmasher as cmr
from easyquery import Query
from pathlib import Path

from utils import mass2color, kpc2deg, deg2kpc

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
rng = np.random.RandomState(42)

ROOT = Path(__file__).resolve().parent.parent
results_dir = ROOT / "results/xSAGA"


def angular_two_point_correlation(
    df, bins, q=None, N_boot=30, method="landy-szalay", random_state=42
):
    r"""Returns bootstrapped angular correlation function for a catalog with coordinates.

    Parameters
        df : DataFrame
            Dataframe of galaxy coordinates and possibly other properties.
        bins : array
            Distance bins in the same units as `df` coordinates (degrees). This
            should probably be log-spaced, e.g., `np.logspace(-2, 1, num=20)`
        q : easyquery.Query
            A query object for filtering `df`. For example, the data can be filtered
            by stellar mass, magnitude, redshift, etc.
        N_boot : int
            The number of bootstrap resamples.
        method : str
            Must be `"landy-szalay"` or `"standard"`.
        random_state : int or np.random.RandomState
            Random state or seed integer for generating background.

    Returns:
        angcorr : 1d array
            Angular correlation function, i.e., $\hat w(\theta)$.
        angcorr_err : 1d array
            Uncertainties on the angular correlations estimated via bootstrapping.
        bootstraps : 2d array
            The bootstrapped angular correlations, which is an array of shape
            (`N_boot`, `len(bins)`).
    """

    if q is not None:
        assert isinstance(q, Query), "Must provide easyquery.Query object"
        df = q.filter(df)

    try:
        angcorr, angcorr_err, bootstraps = bootstrap_two_point_angular(
            df["ra"],
            df["dec"],
            bins=bins,
            Nbootstraps=N_boot,
            method=method,
            random_state=random_state,
        )
    except KeyError:
        try:
            angcorr, angcorr_err, bootstraps = bootstrap_two_point_angular(
                df["ra_NSA"],
                df["dec_NSA"],
                bins=bins,
                Nbootstraps=N_boot,
                method=method,
                random_state=random_state,
            )
        except KeyError as e:
            raise e(
                """Neither `ra` nor `ra_NSA` found.
                Please provide a catalog with viable coordinates.
                """
            )

    return angcorr, angcorr_err, bootstraps


def catalog_redshift_clustering_positions():
    """Use clustering tomography to estimate the redshift distribution of lowz and
    SAGA catalogs. Produces intermediate catalog of positions, and the user must
    later get final redshift bins using http://tomographer.org/, which can be plotted
    with `plot_redshift_clustering()`.
    """

    lowz = pd.read_parquet(results_dir / "lowz-p0_5.parquet")
    saga2 = pd.read_csv(ROOT / "results/predictions-dr9.csv")

    lowz_positions_fname = results_dir / "redshift-clustering/lowz-p0_5-positions.csv"
    lowz[["ra", "dec"]].to_csv(lowz_positions_fname, index=False)

    saga2_positions_fname = results_dir / "redshift-clustering/saga2-positions.csv"
    saga2[["ra", "dec"]].to_csv(saga2_positions_fname, index=False)

    return


def plot_redshift_clustering(
    fname="redshift-clustering",
    renormalize_lowz=38.1755,
    xlabel=r"$z$",
    ylabel=r"$dN/dz \times b$ [deg$^{-2}$]",
):
    """Makes a dN/dz plot as a function of redshift given the clustering catalog
    produced by http://tomographer.org (Menard et al. 2013, Chiang et al. 2019).

    Parameters:
        renormalize_lowz : float
            If not none, renormalize dNdz_b for the low-z sample by a given factor
            (by default this is len(saga2)/len(lowz) = 4433095/116124 = 38.1755).
        fname : str
            The filename of the plot (in the dir `results/xSAGA/plots/`)
        xlabel : str
            Label for the x axis, which should be "$z$" or "Redshift" or something
            like that.
        ylabel : str
            Label for the y axis, which is "$dN/dz$", possibly multiplied by some
            bias term, and most likely with units [deg$^{-2}$].
    """

    lowz_clustering_fname = results_dir / "redshift-clustering/lowz-p0_5-clustering.csv"
    lowz_clustering = pd.read_csv(lowz_clustering_fname)

    if renormalize_lowz is not None:
        lowz_clustering.dNdz_b *= renormalize_lowz
        lowz_clustering.dNdz_b_err *= renormalize_lowz

    saga2_clustering_fname = results_dir / "redshift-clustering/saga2-clustering.csv"
    saga2_clustering = pd.read_csv(saga2_clustering_fname)

    fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=300)
    ax.errorbar(
        lowz_clustering.z,
        lowz_clustering.dNdz_b,
        lowz_clustering.dNdz_b_err,
        ls="",
        lw=1,
        marker="o",
        markersize=4,
        markeredgecolor="none",
        c="#003f5c",
        label="xSAGA low-z",
    )
    ax.errorbar(
        saga2_clustering.z,
        saga2_clustering.dNdz_b,
        saga2_clustering.dNdz_b_err,
        ls="",
        lw=1,
        marker="o",
        markersize=4,
        markeredgecolor="none",
        c="#ff6361",
        label="SAGA-II selection",
    )

    ax.axhline(xmin=0, xmax=1, c="k", lw=1)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(alpha=0.15)
    ax.legend(fontsize=12)
    fig.tight_layout()

    fig.savefig(results_dir / f"plots/{fname}.png")

    return


def plot_angular_two_point_lowz(N_boot=10, bins=np.logspace(-2, 1, 15)):
    """Plot angular 2pt functions comparing xSAGA low-z and SAGA-II.
    """

    bins_plotting = np.sqrt(bins[:-1] * bins[1:])

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)

    # low-z (p_cnn_thresh > 0.5)
    lowz = pd.read_parquet(results_dir / "lowz-p0_5.parquet").sample(10000)
    w_lowz, w_err_lowz, _ = angular_two_point_correlation(
        lowz, bins=bins, N_boot=N_boot
    )
    ax.errorbar(
        bins_plotting, w_lowz, w_err_lowz, ls="", marker="o", label="xSAGA $z < 0.03$"
    )

    # all saga 2
    saga = pd.read_csv(ROOT / "results/predictions-dr9.csv").sample(10000)
    q_saga = Query("ra == ra", "dec == dec")
    w_saga, w_err_saga, _ = angular_two_point_correlation(
        saga, bins=bins, N_boot=N_boot, q=q_saga
    )
    ax.errorbar(bins_plotting, w_saga, w_err_saga, ls="", marker="o", label="SAGA II")

    ax.legend(loc="upper right", fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"$\theta$ [deg]", fontsize=12)
    ax.set_ylabel(r"$\hat w(\theta)$", fontsize=12)
    ax.grid(alpha=0.2)

    secax = ax.secondary_xaxis("top", functions=(deg2kpc, kpc2deg))
    secax.set_xlabel("Distance ($z=0.03$) [kpc]")
    fig.tight_layout()

    fig.savefig(results_dir / "plots/angular_2pt_lowz-vs-saga.png")

    return


def plot_angular_two_point_sats_and_hosts(N_boot=10, bins=np.logspace(-2, 1, 15)):
    """Plot angular 2pt functions comparing satellites and hosts.
    """

    bins_plotting = np.sqrt(bins[:-1] * bins[1:])

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)

    sats = pd.read_parquet(results_dir / "sats-nsa_p0_5.parquet")
    w_sats, w_err_sats, _ = angular_two_point_correlation(
        sats, bins=bins, N_boot=N_boot
    )
    ax.errorbar(
        bins_plotting, w_sats, w_err_sats, ls="", marker="o", label="satellites"
    )

    hosts = pd.read_parquet(results_dir / "hosts-nsa.parquet")
    q_hosts = Query("mass_GSE > 9.5")
    w_hosts, w_err_hosts, _ = angular_two_point_correlation(
        hosts, bins=bins, N_boot=N_boot, q=q_hosts
    )
    ax.errorbar(bins_plotting, w_hosts, w_err_hosts, ls="", marker="o", label="hosts")

    ax.legend(loc="upper right", fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"$\theta$ [deg]", fontsize=12)
    ax.set_ylabel(r"$\hat w(\theta)$", fontsize=12)
    ax.grid(alpha=0.2)

    secax = ax.secondary_xaxis("top", functions=(deg2kpc, kpc2deg))
    secax.set_xlabel("Distance ($z=0.03$) [kpc]")
    fig.tight_layout()

    fig.savefig(results_dir / "plots/angular_2pt_sats-and-hosts.png")

    return


def plot_angular_two_point_sats_by_host_mass(
    mass_min=9.5, mass_max=11.0, dmass=0.25, N_boot=10, bins=np.logspace(-2, 1, 15)
):
    """Plot angular 2pt functions for satellites by host mass.
    """

    mass_bins = np.arange(mass_min, mass_max, dmass)
    sats = pd.read_parquet(results_dir / "sats-nsa_p0_5.parquet")

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
    bins_plotting = np.sqrt(bins[:-1] * bins[1:])
    bins_offset = np.linspace(1.03 ** -1, 1.03, len(mass_bins))

    for m1, m2, offset in zip(mass_bins, mass_bins + dmass, bins_offset):
        q = Query(f"mass_GSE > {m1}", f"mass_GSE < {m2}")
        w, w_err, _ = angular_two_point_correlation(sats, bins=bins, N_boot=N_boot, q=q)

        ax.errorbar(
            bins_plotting * offset,
            w,
            w_err,
            c=mass2color((m1 + m2) / 2),
            ls="",
            marker="o",
            label=f"${m1} - {m2}$",
        )

    ax.legend(loc="upper right", fontsize=12, title="Host " r"$\log(M_★/M_\odot)$")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"$\theta$ [deg]", fontsize=12)
    ax.set_ylabel(r"$\hat w(\theta)$", fontsize=12)
    ax.grid(alpha=0.2)

    secax = ax.secondary_xaxis("top", functions=(deg2kpc, kpc2deg))
    secax.set_xlabel("Distance ($z=0.03$) [kpc]")
    fig.tight_layout()

    fig.savefig(results_dir / "plots/angular_2pt_sats-by-host-mass.png")

    return


def plot_angular_two_point_sats_by_apparent_magnitude(
    r_min=14, r_max=20, dr=1, N_boot=10, bins=np.logspace(-2, 1, 15)
):
    """Plot angular 2pt functions for satellites by satellite absolute magnitude.
    """

    r_bins = np.arange(r_min, r_max, dr)
    sats = pd.read_parquet(results_dir / "sats-nsa_p0_5.parquet")

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
    bins_plotting = np.sqrt(bins[:-1] * bins[1:])
    bins_offset = np.linspace(1.03 ** -1, 1.03, len(r_bins))

    for r1, r2, offset in zip(r_bins, r_bins + dr, bins_offset):
        q = Query(f"r0 > {r1}", f"r0 < {r2}")
        w, w_err, _ = angular_two_point_correlation(sats, bins=bins, N_boot=N_boot, q=q)

        ax.errorbar(
            bins_plotting * offset,
            w,
            w_err,
            c=cmr.dusk_r((r2 - 14) / 7),
            ls="",
            marker="o",
            label=f"${r1} < r_0 < {r2}$",
        )

    ax.legend(loc="upper right", fontsize=12, title="Apparent magnitude")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"$\theta$ [deg]", fontsize=12)
    ax.set_ylabel(r"$\hat w(\theta)$", fontsize=12)
    ax.grid(alpha=0.2)

    secax = ax.secondary_xaxis("top", functions=(deg2kpc, kpc2deg))
    secax.set_xlabel("Distance ($z=0.03$) [kpc]")
    fig.tight_layout()

    fig.savefig(results_dir / "plots/angular_2pt_sats-by-r0.png")

    return


def plot_angular_two_point_sats_by_absolute_magnitude(
    M_r_min=-21, M_r_max=-15, dM_r=1, N_boot=10, bins=np.logspace(-2, 1, 15)
):
    """Plot angular 2pt functions for satellites by satellite absolute magnitude.
    """

    M_r_bins = np.arange(M_r_min, M_r_max, dM_r)
    sats = pd.read_parquet(results_dir / "sats-nsa_p0_5.parquet")

    sats["M_r"] = sats.r0 - cosmo.distmod(sats.z_NSA).value

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
    bins_plotting = np.sqrt(bins[:-1] * bins[1:])
    bins_offset = np.linspace(1.08 ** -1, 1.08, len(M_r_bins))

    for M_r1, M_r2, offset in zip(M_r_bins, M_r_bins + dM_r, bins_offset):
        q = Query(f"M_r > {M_r1}", f"M_r < {M_r2}")
        w, w_err, _ = angular_two_point_correlation(sats, bins=bins, N_boot=N_boot, q=q)

        ax.errorbar(
            bins_plotting * offset,
            w,
            w_err,
            c=cmr.dusk_r((M_r2 + 21.0) / 7),
            ls="",
            marker="o",
            label=f"${M_r1} < M_r < {M_r2}$",
        )

    ax.legend(loc="upper right", fontsize=12, title="Absolute magnitude")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"$\theta$ [deg]", fontsize=12)
    ax.set_ylabel(r"$\hat w(\theta)$", fontsize=12)
    ax.grid(alpha=0.2)

    secax = ax.secondary_xaxis("top", functions=(deg2kpc, kpc2deg))
    secax.set_xlabel("Distance ($z=0.03$) [kpc]")
    fig.tight_layout()

    fig.savefig(results_dir / "plots/angular_2pt_sats-by-M_r.png")

    return


def compute_nonsatellite_surface_density(
    lowz,
    N_rand=10000,
    N_boot=100,
    ra_range=[120, 160],
    dec_range=[0, 50],
    radius=0.3 * u.deg,
):
    """Draw random circles on the sky (random hosts) and return the bootstrapped median
    surface density of probable low-z candidates. The default RA and Dec range
    avoid the Coma and Virgo clusters.
    """

    ra_min, ra_max = ra_range
    dec_min, dec_max = dec_range

    ra_rand = (rng.random(size=N_rand)) * (ra_max - ra_min) + ra_min
    dec_rand = np.rad2deg(
        np.pi / 2
        - np.arccos(rng.random(size=N_rand) * np.deg2rad(dec_max - dec_min) + dec_min)
    )

    rand_coords = SkyCoord(ra_rand, dec_rand, unit="deg")
    lowz_coords = SkyCoord(lowz.ra, lowz.dec, unit="deg")

    idx_lowz, _, _, _ = lowz_coords.search_around_sky(rand_coords, radius)

    N_lowz_per_rand = np.array([np.sum(i == idx_lowz) for i in np.arange(N_rand)])

    boot_lowz_surface_density = bootstrap(
        N_lowz_per_rand / (np.pi * (radius.value) ** 2),
        bootfunc=np.mean,
        bootnum=N_boot,
    )

    return boot_lowz_surface_density.mean()


# def plot_image_grid(df, mask, img_dir="", nrows=1, ncols=4, seed=seed):
#     """Plot a grid of images selected randomly.
#     """
#
#     fnames = df[mask].sample(N=nrows*ncols, random_state=seed).OBJID
#
#     images = np.array([np.asarray(imread(fname)) for img in fnames])
#
#     fig = plt.figure(figsize=(4.0, 4.0))
#     grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0.05)
#
#     for ax, im in zip(grid, images):


if __name__ == "__main__":
    # plot_angular_two_point_lowz()
    # plot_angular_two_point_sats_and_hosts()
    # plot_angular_two_point_sats_by_host_mass()
    # plot_angular_two_point_sats_by_apparent_magnitude()
    # plot_angular_two_point_sats_by_absolute_magnitude()

    # catalog_redshift_clustering_positions()
    # plot_redshift_clustering()

    lowz = pd.read_parquet(results_dir / "lowz-p0_5.parquet")
    lowz_surface_density = compute_nonsatellite_surface_density(lowz)
    print(lowz_surface_density)
