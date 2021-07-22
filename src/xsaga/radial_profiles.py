"""
John F. Wu
2021-07-22

Scripts for investigating the radial distribution of satellties around hosts. Output
plots are saved to `{ROOT}/results/xSAGA/plots/profiles/`.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.stats import bootstrap
import cmasher as cmr
from easyquery import Query
from functools import partial
from pathlib import Path

from utils import mass2color

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

ROOT = Path(__file__).resolve().parent.parent.parent
results_dir = ROOT / "results/xSAGA"

# cuts on satellite systems
SAT_QUERY = Query("M_r < -15.0")
HOST_QUERY = Query("mass_GSE > 9.5", "z_NSA >= 0.005", "z_NSA <= 0.03")


def load_hosts_and_sats():
    """Loads the hosts and satellite catalogs, the latter with a `M_r` column.
    """
    hosts = pd.read_parquet(results_dir / "hosts-nsa.parquet")
    sats = pd.read_parquet(results_dir / "sats-nsa_p0_5.parquet")
    sats["M_r"] = sats.r0 - cosmo.distmod(sats.z_NSA).value

    return hosts, sats


def load_random_hosts_and_sats():
    """Loads the random hosts and satellite catalogs, the latter with a `M_r` column.
    """
    hosts_rand = pd.read_parquet(results_dir / "hosts_rand-nsa.parquet")
    sats_rand = pd.read_parquet(results_dir / "sats_rand-nsa_p0_5.parquet")
    sats_rand["M_r"] = sats_rand.r0 - cosmo.distmod(sats_rand.z_NSA).value

    return hosts_rand, sats_rand


def compute_radial_cdf(satellite_separations, radial_bins):
    """Counts the cumulative number of satellites within a list of radial bins.

    Parameters
        satellite_separations : 1d array-like
            A list or array of separations, which should not have a DataFrame index
            and should already be filtered using host mass or morphology.
        radial_bins : 1d array-like
            A list or array of (maximum) radii.

    Returns
        counts : 1d array
            An array of cumulative satellite counts for each bin in `radial_bins`.
    """
    return np.array([np.sum(satellite_separations < r) for r in radial_bins])


def plot_radial_profile_by_host_mass(
    hosts,
    sats,
    radial_bins=np.arange(36, 300, 1),
    normalize=False,
    cumulative=True,
    N_boot=None,
    mass_min=9.5,
    mass_max=11.5,
    dmass=0.5,
    fname="radial_profile-by-host_mass",
):
    """Creates a radial profile plot for satellites, colored by mass bins.
    """

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
    mass_bins = np.arange(mass_min, mass_max, dmass)

    for m1, m2 in zip(mass_bins, mass_bins + dmass):
        q = Query(f"mass_GSE > {m1}", f"mass_GSE < {m2}")
        satellite_separations = q.filter(sats).sep.values

        # compute sat cumulative profile or bootstrapped profile
        if N_boot is None:
            profile = compute_radial_cdf(satellite_separations, radial_bins)

            # either normalize satellite counts or divide by number of hosts
            profile = profile / (q.count(sats) if normalize else q.count(hosts))

            if not cumulative:
                profile = np.gradient(profile, radial_bins)

            ax.plot(
                radial_bins,
                profile,
                c=mass2color((m1 + m2) / 2),
                label=f"${m1}-{m2}$",
                lw=3,
            )
        else:
            assert isinstance(N_boot, int), "Please enter an integer `N_boot`."
            profile_bootstrapped = bootstrap(
                satellite_separations,
                bootfunc=partial(compute_radial_cdf, radial_bins=radial_bins),
                bootnum=N_boot,
            )

            # TODO: should I also boostrap the denominator?
            profile_bootstrapped = profile_bootstrapped / (
                q.count(sats) if normalize else q.count(hosts)
            )

            if not cumulative:
                profile_bootstrapped = np.gradient(
                    profile_bootstrapped, radial_bins, axis=1
                )

            ax.fill_between(
                radial_bins,
                *np.quantile(profile_bootstrapped, [0.16, 0.84], axis=0),
                color=mass2color((m1 + m2) / 2),
                label=f"${m1}-{m2}$",
                lw=0,
                alpha=0.7,
            )

    xlabel = "Distance [pkpc]"
    ylabel = r"$N_{\rm sat}$" + ("(<r)" if cumulative else "(r)")
    ylabel = ("Normalized " if normalize else "") + ylabel

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(alpha=0.15)
    ax.legend(loc="upper left", fontsize=12, title="Host mass", title_fontsize=14)
    fig.tight_layout()
    fig.savefig(results_dir / f"plots/profiles/{fname}.png")


def plot_radial_profile_by_host_morphology(
    hosts,
    sats,
    radial_bins=np.arange(36, 300, 1),
    normalize=False,
    cumulative=True,
    N_boot=None,
    sersic_n_low=(0, 3),
    sersic_n_high=(3, 6),
    mass_min=9.5,
    mass_max=11.5,
    dmass=0.5,
    fname="radial_profile-by-host_morphology",
):
    """Creates two satellite radial profile plots separated by Sersic index.
    """

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 6), dpi=300, sharey=True)
    mass_bins = np.arange(mass_min, mass_max, dmass)

    for sersic_n_range, ax in zip([sersic_n_low, sersic_n_high], [ax1, ax2]):
        n1, n2 = sersic_n_range
        q_sersic_n = Query(f"SERSIC_N_NSA > {n1}", f"SERSIC_N_NSA < {n2}")

        for m1, m2 in zip(mass_bins, mass_bins + dmass):
            q_mass = Query(f"mass_GSE > {m1}", f"mass_GSE < {m2}")

            # combine morphology and mass queries
            q = q_sersic_n & q_mass
            satellite_separations = q.filter(sats).sep.values

            # compute sat cumulative profile or bootstrapped profile
            if N_boot is None:
                profile = compute_radial_cdf(satellite_separations, radial_bins)

                # either normalize satellite counts or divide by number of hosts
                profile = profile / (q.count(sats) if normalize else q.count(hosts))

                if not cumulative:
                    profile = np.gradient(profile, radial_bins)

                ax.plot(
                    radial_bins,
                    profile,
                    c=mass2color((m1 + m2) / 2),
                    label=f"${m1}-{m2}$",
                    lw=3,
                )
            else:
                assert isinstance(N_boot, int), "Please enter an integer `N_boot`."
                profile_bootstrapped = bootstrap(
                    satellite_separations,
                    bootfunc=partial(compute_radial_cdf, radial_bins=radial_bins),
                    bootnum=N_boot,
                )

                # TODO: should I also boostrap the denominator?
                profile_bootstrapped = profile_bootstrapped / (
                    q.count(sats) if normalize else q.count(hosts)
                )

                if not cumulative:
                    profile_bootstrapped = np.gradient(
                        profile_bootstrapped, radial_bins, axis=1
                    )

                ax.fill_between(
                    radial_bins,
                    *np.quantile(profile_bootstrapped, [0.16, 0.84], axis=0),
                    color=mass2color((m1 + m2) / 2),
                    label=f"${m1}-{m2}$",
                    lw=0,
                    alpha=0.7,
                )

        ax.grid(alpha=0.15)
        ax.text(
            0.5,
            0.92,
            r"${} < N_{{\rm Sersic}} < {}$".format(n1, n2),
            transform=ax.transAxes,
            ha="center",
            fontsize=16,
        )

    xlabel = "Distance [pkpc]"
    ylabel = r"$N_{\rm sat}$" + ("(<r)" if cumulative else "(r)")
    ylabel = ("Normalized " if normalize else "") + ylabel

    ax1.set_xlabel(xlabel, fontsize=12)
    ax2.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.legend(loc="upper left", fontsize=12, title="Host mass", title_fontsize=14)
    fig.tight_layout()
    fig.savefig(results_dir / f"plots/profiles/{fname}.png")


if __name__ == "__main__":

    hosts, sats = load_hosts_and_sats()
    hosts_rand, sats_rand = load_random_hosts_and_sats()

    # impose cuts
    hosts = HOST_QUERY.filter(hosts)
    hosts_rand = HOST_QUERY.filter(hosts_rand)
    sats = (SAT_QUERY & HOST_QUERY).filter(sats)
    sats_rand = (SAT_QUERY & HOST_QUERY).filter(sats_rand)

    # plot_radial_profile_by_host_mass(
    #     hosts,
    #     sats,
    #     dmass=0.25,
    #     normalize=False,
    #     N_boot=100,
    #     fname="radial_profile-by-host_mass",
    # )

    # plot_radial_profile_by_host_mass(
    #     hosts,
    #     sats,
    #     dmass=0.25,
    #     normalize=True,
    #     N_boot=100,
    #     fname="normalized_radial_profile-by-host_mass",
    # )

    # plot_radial_profile_by_host_mass(
    #     hosts,
    #     sats,
    #     radial_bins=np.arange(36, 300, 10),
    #     dmass=0.5,
    #     normalize=False,
    #     cumulative=False,
    #     N_boot=100,
    #     fname="radial_pdf-by-host_mass",
    # )

    # plot_radial_profile_by_host_mass(
    #     hosts,
    #     sats,
    #     radial_bins=np.arange(36, 300, 10),
    #     dmass=0.5,
    #     normalize=True,
    #     cumulative=False,
    #     N_boot=100,
    #     fname="normalized_radial_pdf-by-host_mass",
    # )

    plot_radial_profile_by_host_morphology(
        hosts,
        sats,
        radial_bins=np.arange(36, 300, 1),
        mass_min=9.5,
        mass_max=11.5,
        dmass=0.5,
        normalize=False,
        cumulative=True,
        N_boot=100,
        fname="radial_profile-by-host_morphology",
    )

    plot_radial_profile_by_host_morphology(
        hosts,
        sats,
        radial_bins=np.arange(36, 300, 10),
        mass_min=9.5,
        mass_max=11.5,
        dmass=0.5,
        normalize=False,
        cumulative=False,
        N_boot=100,
        fname="radial_pdf-by-host_morphology",
    )
