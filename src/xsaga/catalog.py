"""
John F. Wu
2021-07-20

Scripts for creating catalogs and figures to be in the first xSAGA paper.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astroML.correlation import (
    bootstrap_two_point_angular,
    two_point_angular,
    bootstrap_two_point,
    two_point,
)
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import cmasher as cmr
from easyquery import Query
from functools import partial
from pathlib import Path

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

MAKE_PLOTS = False
EXT = "png"

ROOT = Path(__file__).resolve().parent.parent.parent
results_dir = ROOT / "results/xSAGA"


def measure_purity():
    pass


def measure_completeness():
    pass


def compare_surface_brightness():
    pass


def compare_color():
    pass


def angular_two_point_correlation(
    df, bins, q=None, N_boot=30, method="landy-szalay", random_state=42
):
    """Returns bootstrapped angular correlation function for a catalog with coordinates.

    Parameters
    ----------
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
            print("Columns `ra` and `dec` not found. Trying NSA coordinates.")
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


def two_point_correlation():
    pass


def redshift_clustering():
    pass


def _deg2kpc(theta, z):
    """Convenience function for converting angular distances [deg] to physical
    distances [kpc] at a given redshift. Used in `ax.secondary_xaxis` for plotting.
    """
    return (theta * u.deg * cosmo.kpc_proper_per_arcmin(z=z)).to("kpc").value


def _kpc2deg(dist, z):
    """Convenience function for converting physical distances [kpc] to angular
    distances [deg] at a given redshift. Used in `ax.secondary_xaxis` for plotting.
    """
    return (dist * u.kpc / cosmo.kpc_proper_per_arcmin(z=z)).to("deg").value


deg2kpc = partial(_deg2kpc, z=0.03)
kpc2deg = partial(_kpc2deg, z=0.03)


def mass2color(mass, cmap=cmr.ember, mass_min=9.5, mass_max=11.5):
    """Convenience function for mapping a stellar mass, normalized to some mass range,
    to a color determined by colormap `cmap`.
    """
    return cmap((mass - mass_min) / (mass_max - mass_min))


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


if __name__ == "__main__":
    plot_angular_two_point_lowz()
    plot_angular_two_point_sats_and_hosts()
    # plot_angular_two_point_sats_by_host_mass()
    # plot_angular_two_point_sats_by_apparent_magnitude()
    # plot_angular_two_point_sats_by_absolute_magnitude()
