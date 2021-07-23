"""
John F. Wu
2021-07-22

Scripts for investigating the radial distribution of satellties around hosts. Output
plots are saved to `{ROOT}/results/xSAGA/plots/profiles-overlapping_hosts/`

Note: in `{ROOT}/results/xSAGA/plots/profiles-not_isolated_hosts/`, you can also find
the script generated before we ensured that hosts are isolated/not overlapping.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
from astropy.stats import bootstrap
from astropy import units as u
from easyquery import Query
from functools import partial
from pathlib import Path

from utils import mass2color

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

ROOT = Path(__file__).resolve().parent.parent.parent
results_dir = ROOT / "results/xSAGA"

# cuts on satellite systems
SAT_QUERY = Query("M_r < -15.0")
HOST_QUERY = Query("mass_GSE > 9.5", "z_NSA >= 0.02", "z_NSA <= 0.03")


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


def isolate_hosts(
    hosts, delta_mass=0.5, delta_z=0.003, delta_d=1.0, return_not_isolated=False
):
    """Remove hosts that have significantly more massive neighbors nearby.

    Parameters
        hosts : pd.DataFrame
            The DataFrame of hosts, pre-filtered
        delta_mass, delta_z, delta_d : floats
            If the difference in mass between a host and another neighbor within
            projected distance `delta_d` (Mpc) and redshift `delta_z` exceeds
            `delta_mass`, then remove it.
        return_not_isolated : bool
            If True, then return the converse (not isolated) subset of hosts

    Returns
        isolated_hosts : pd.DataFrame
            The DataFrame of hosts, now filtered according to given parameters.
            Includes extra columns, `nearest_host_NSAID`, which gives the NSAID
            of the nearest *projected* host, and `nearest_host_sep`, which gives
            the separation in projected Mpc.
    """

    hosts_coord = SkyCoord(hosts.ra_NSA, hosts.dec_NSA, unit="deg")

    idx, sep, _ = hosts_coord.match_to_catalog_sky(hosts_coord, nthneighbor=2)
    nearest_host_sep = (
        (sep * cosmo.kpc_proper_per_arcmin(hosts.iloc[idx].z_NSA)).to(u.Mpc).value
    )
    nearest_host_NSAID = hosts.iloc[idx].index.values

    hosts["nearest_host_sep"] = nearest_host_sep
    hosts["nearest_host_NSAID"] = nearest_host_NSAID

    not_isolated = (
        (nearest_host_sep < delta_d)
        & (np.abs(hosts.z_NSA.values - hosts.iloc[idx].z_NSA.values) < delta_z)
        & (hosts.iloc[idx].mass_GSE.values - hosts.mass_GSE.values > delta_mass)
    )

    # XOR ensures that we return the correct subset (isolated/not isolated) of hosts
    return hosts[(return_not_isolated) ^ (~not_isolated)]


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
    cumulative=True,
    normalize=False,
    surface_density=False,
    N_boot=None,
    mass_min=9.5,
    mass_max=11.5,
    dmass=0.5,
    fname="radial_profile-by-host_mass",
):
    """Creates a radial profile plot for satellites colored by host mass.

    Parameters
        hosts : pd.DataFrame
            DataFrame of hosts, which is assumed to be pre-filtered using
            cuts at top of this script.
        sats : pd.DataFrame
            DataFrame of satellites, also assumed to be filtered.
        radial_bins : 1d array-like
            A list or array of (maximum) projected radii.
        cumulative : bool
            Toggles whether to plot the satellite number within a given projected
            radius versus at a given radius.
        normalize : bool
            Toggles whether to divide the curves by the total number of satellites
        surface_density : bool
            Toggles whether to divide by the annular surface area at each radius.
        N_boot : int
            The number of bootstrap resamples (for estimating uncertainties).
            Can also be `None` if boostrapping is not wanted.
        mass_min : float
            The minimum host (log) mass.
        mass_max : float
            The maximum host (log) mass.
        dmass : float
            The interval per mass bin.
        fname : str
            The name of the output figure (JPG format), to be saved in the directory
            `./results/xSAGA/plots/profiles/`.
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

            if surface_density:
                # kpc^-2 --> Mpc^-2
                surface_area = np.pi * np.gradient(radial_bins ** 2) * 1e-6
                profile = profile / surface_area

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

            # TODO: should I also bootstrap the denominator?
            profile_bootstrapped = profile_bootstrapped / (
                q.count(sats) if normalize else q.count(hosts)
            )

            if surface_density:
                surface_area = np.pi * np.gradient(radial_bins ** 2) * 1e-6
                profile_bootstrapped = profile_bootstrapped / surface_area

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
    ylabel = (
        ("Normalized " if normalize else "")
        + (r"$\Sigma_{\rm sat}$" if surface_density else r"$N_{\rm sat}$")
        + ("(<r)" if cumulative else "(r)")
        + (" [Mpc$^{-2}$]" if surface_density else "")
    )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(alpha=0.15)

    legend_location = (
        "upper right" if (surface_density and (not cumulative)) else "upper left"
    )
    ax.legend(loc=legend_location, fontsize=12, title="Host mass", title_fontsize=14)
    fig.tight_layout()
    fig.savefig(results_dir / f"plots/profiles/{fname}.png")


def plot_corrected_radial_profile_by_host_mass(
    hosts,
    hosts_rand,
    sats,
    sats_rand,
    radial_bins=np.arange(36, 300, 1),
    cumulative=True,
    normalize=False,
    surface_density=False,
    N_boot=None,
    mass_min=9.5,
    mass_max=11.5,
    dmass=0.5,
    fname="radial_profile-by-host_mass",
):
    """Creates a satellite radial profile, corrected for randoms, as a function of
    host stellar mass.

    Parameters
        hosts : pd.DataFrame
            DataFrame of real hosts.
        hosts_rand : pd.DataFrame
            Dataframe of random hosts.
        sats : pd.DataFrame
            DataFrame of satellites around real hosts.
        sats_rand : pd.DataFrame
            DataFrame of satellites around random hosts.

    All other parameters are the same as `plot_radial_profile_by_host_mass()`.
    """

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
    mass_bins = np.arange(mass_min, mass_max, dmass)

    for m1, m2 in zip(mass_bins, mass_bins + dmass):
        q = Query(f"mass_GSE > {m1}", f"mass_GSE < {m2}")
        satellite_separations = q.filter(sats).sep.values
        random_separations = q.filter(sats_rand).sep.values

        if N_boot is None:
            profile = compute_radial_cdf(satellite_separations, radial_bins)
            profile_rand = compute_radial_cdf(random_separations, radial_bins)

            profile = profile / (q.count(sats) if normalize else q.count(hosts))
            profile_rand = profile_rand / (
                q.count(sats_rand) if normalize else q.count(hosts_rand)
            )

            if surface_density:
                surface_area = np.pi * np.gradient(radial_bins ** 2) * 1e-6
                profile = profile / surface_area
                profile_rand = profile_rand / surface_area

            if not cumulative:
                profile = np.gradient(profile, radial_bins)
                profile_rand = np.gradient(profile_rand, radial_bins)

            ax.plot(
                radial_bins,
                profile - profile_rand,
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
            profile_rand_bootstrapped = bootstrap(
                random_separations,
                bootfunc=partial(compute_radial_cdf, radial_bins=radial_bins),
                bootnum=N_boot,
            )

            profile_bootstrapped = profile_bootstrapped / (
                q.count(sats) if normalize else q.count(hosts)
            )
            profile_rand_bootstrapped = profile_rand_bootstrapped / (
                q.count(sats_rand) if normalize else q.count(hosts_rand)
            )

            if surface_density:
                surface_area = np.pi * np.gradient(radial_bins ** 2) * 1e-6
                profile_bootstrapped = profile_bootstrapped / surface_area
                profile_rand_bootstrapped = profile_rand_bootstrapped / surface_area

            if not cumulative:
                profile_bootstrapped = np.gradient(
                    profile_bootstrapped, radial_bins, axis=1
                )
                profile_rand_bootstrapped = np.gradient(
                    profile_rand_bootstrapped, radial_bins, axis=1
                )

            ax.fill_between(
                radial_bins,
                *np.quantile(
                    profile_bootstrapped - profile_rand_bootstrapped,
                    [0.16, 0.84],
                    axis=0,
                ),
                color=mass2color((m1 + m2) / 2),
                label=f"${m1}-{m2}$",
                lw=0,
                alpha=0.7,
            )

    xlabel = "Distance [pkpc]"
    ylabel = (
        ("Normalized " if normalize else "")
        + (
            r"$\big \[ \Sigma_{\rm sat} - \Sigma_{\rm rand} \big \]$"
            if surface_density
            else r"$\big \[ N_{\rm sat} - N_{\rm rand} \big \]$"
        )
        + ("(<r)" if cumulative else "(r)")
        + (" [Mpc$^{-2}$]" if surface_density else "")
    )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(alpha=0.15)

    legend_location = (
        "upper right" if (surface_density and (not cumulative)) else "upper left"
    )
    ax.legend(loc=legend_location, fontsize=12, title="Host mass", title_fontsize=14)
    fig.tight_layout()
    fig.savefig(results_dir / f"plots/profiles/{fname}.png")


def plot_radial_profile_by_host_morphology(
    hosts,
    sats,
    radial_bins=np.arange(36, 300, 1),
    cumulative=True,
    normalize=False,
    surface_density=False,
    N_boot=None,
    sersic_n_low=(0, 3),
    sersic_n_high=(3, 6),
    mass_min=9.5,
    mass_max=11.5,
    dmass=0.5,
    fname="radial_profile-by-host_morphology",
):
    """Creates two satellite radial profile plots based on a split in host morphology.

    Parameters
        hosts : pd.DataFrame
            DataFrame of hosts, which is assumed to be pre-filtered using
            cuts at top of this script.
        sats : pd.DataFrame
            DataFrame of satellites, also assumed to be filtered.
        radial_bins : 1d array-like
            A list or array of (maximum) projected radii.
        cumulative : bool
            Toggles whether to plot the satellite number within a given projected
            radius versus at a given radius.
        normalize : bool
            Toggles whether to divide the curves by the total number of satellites
        surface_density : bool
            Toggles whether to divide by the annular surface area at each radius.
        N_boot : int
            The number of bootstrap resamples (for estimating uncertainties).
            Can also be `None` if boostrapping is not wanted.
        sersic_n_low : tuple(float, float)
            A pair of min and max Sersic indices to define the disk morphology.
        sersic_n_high : tuple(float, float)
            A pair of min and max Sersic indices to define the elliptical morphology.
        mass_min : float
            The minimum host (log) mass.
        mass_max : float
            The maximum host (log) mass.
        dmass : float
            The interval per mass bin.
        fname : str
            The name of the output figure (JPG format), to be saved in the directory
            `./results/xSAGA/plots/profiles/`.
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

                if surface_density:
                    surface_area = np.pi * np.gradient(radial_bins ** 2) * 1e-6
                    profile = profile / surface_area

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

                if surface_density:
                    surface_area = np.pi * np.gradient(radial_bins ** 2) * 1e-6
                    profile_bootstrapped = profile_bootstrapped / surface_area

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
    ylabel = (
        ("Normalized " if normalize else "")
        + (r"$\Sigma_{\rm sat}$" if surface_density else r"$N_{\rm sat}$")
        + ("(<r)" if cumulative else "(r)")
        + (" [Mpc$^{-2}$]" if surface_density else "")
    )

    ax1.set_xlabel(xlabel, fontsize=12)
    ax2.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)

    legend_location = (
        "upper right" if (surface_density and (not cumulative)) else "upper left"
    )
    ax1.legend(loc=legend_location, fontsize=12, title="Host mass", title_fontsize=14)

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

    # isolate hosts and sats
    hosts = isolate_hosts(hosts, delta_mass=0.5, delta_z=0.003, delta_d=1.0)
    sats = sats[sats.NSAID.isin(hosts.index)].copy()

    # use bootstraps
    N_boot = 100

    # host mass
    # =========

    plot_radial_profile_by_host_mass(
        hosts, sats, dmass=0.25, N_boot=N_boot, fname="radial_profile-by-host_mass"
    )

    plot_radial_profile_by_host_mass(
        hosts,
        sats,
        dmass=0.25,
        normalize=True,
        N_boot=N_boot,
        fname="normalized_radial_profile-by-host_mass",
    )

    plot_radial_profile_by_host_mass(
        hosts,
        sats,
        dmass=0.25,
        surface_density=True,
        N_boot=N_boot,
        fname="surface_density_profile-by-host_mass",
    )

    plot_radial_profile_by_host_mass(
        hosts,
        sats,
        radial_bins=np.arange(36, 300, 10),
        dmass=0.5,
        cumulative=False,
        N_boot=N_boot,
        fname="radial_pdf-by-host_mass",
    )

    plot_radial_profile_by_host_mass(
        hosts,
        sats,
        radial_bins=np.arange(36, 300, 10),
        cumulative=False,
        surface_density=True,
        N_boot=N_boot,
        fname="surface_density_pdf-by-host_mass",
    )

    plot_radial_profile_by_host_mass(
        hosts,
        sats,
        radial_bins=np.arange(36, 300, 10),
        cumulative=False,
        normalize=True,
        N_boot=N_boot,
        fname="normalized_radial_pdf-by-host_mass",
    )

    # morphology
    # ==========

    plot_radial_profile_by_host_morphology(
        hosts,
        sats,
        radial_bins=np.arange(36, 300, 1),
        cumulative=True,
        N_boot=N_boot,
        fname="radial_profile-by-host_morphology",
    )

    plot_radial_profile_by_host_morphology(
        hosts,
        sats,
        radial_bins=np.arange(36, 300, 10),
        cumulative=False,
        N_boot=N_boot,
        fname="radial_pdf-by-host_morphology",
    )

    plot_radial_profile_by_host_morphology(
        hosts,
        sats,
        radial_bins=np.arange(36, 300, 10),
        cumulative=False,
        surface_density=True,
        N_boot=N_boot,
        fname="surface_density_pdf-by-host_morphology",
    )

    # more plots corrected for randoms
    # ================================

    plot_radial_profile_by_host_mass(
        hosts,
        sats,
        radial_bins=np.arange(36, 300, 1),
        cumulative=True,
        N_boot=N_boot,
        fname="corrected_radial_profile-by-host_mass",
    )

    plot_radial_profile_by_host_morphology(
        hosts,
        sats,
        radial_bins=np.arange(36, 300, 1),
        cumulative=True,
        N_boot=N_boot,
        fname="corrected_radial_profile-by-host_morphology",
    )

    plot_radial_profile_by_host_morphology(
        hosts,
        sats,
        radial_bins=np.arange(36, 300, 1),
        cumulative=True,
        normalize=True,
        N_boot=N_boot,
        fname="corrected_normalized_radial_profile-by-host_morphology",
    )
