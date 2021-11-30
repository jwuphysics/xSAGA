"""
John F. Wu (2021)

Scripts for investigating the radial distribution of satellties around hosts. Output
plots are saved to `{ROOT}/results/xSAGA/plots/profiles/`
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord
from astropy.stats import bootstrap
from astropy import units as u
from astropy.utils.misc import NumpyRNGContext
import cmasher as cmr
from easyquery import Query
from functools import partial
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

from utils import mass2color, gap2color
from satellites import compute_magnitude_gap

rng = np.random.default_rng(42)
astropy_rng = NumpyRNGContext(42)

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

ROOT = Path(__file__).resolve().parent.parent
results_dir = ROOT / "results/xSAGA"

# cuts on satellite systems
M_r_lim = -15.0
HOST_QUERY = Query("mass_GSE > 9.5", "mass_GSE < 11", "z_NSA >= 0.01", "z_NSA <= 0.03")
SAT_QUERY = Query(f"M_r < {M_r_lim}")


def load_hosts_and_sats():
    """Loads the hosts and satellite catalogs, the latter with a `M_r` column.
    """
    hosts = pd.read_parquet(results_dir / "hosts-nsa.parquet")
    sats = pd.read_parquet(results_dir / "sats-nsa_p0_5.parquet")
    sats["M_r"] = sats.r0 - cosmo.distmod(sats.z_NSA).value

    return hosts, sats


def isolate_hosts(
    hosts, delta_mass=0.5, delta_z=0.005, delta_d=1.0, return_not_isolated=False
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


def compute_radial_profile(satellite_separations, radial_bins):
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


def compute_interloper_profile(redshifts, radial_bins, interloper_surface_density=3.04):
    """Helper function to compute the cumulative number of interlopers for hosts given
    by some list of redshifts.
    """
    deg_per_kpc = cosmo.arcsec_per_kpc_proper(redshifts).to(u.deg / u.kpc).value
    angular_surface_area = np.array(
        [
            np.pi * (radial_bins * deg2kpc) ** 2 - (radial_bins.min() * deg2kpc) ** 2
            for deg2kpc in deg_per_kpc
        ]
    )

    return interloper_surface_density * angular_surface_area


def compute_nonsatellite_profile(N_unrelated_lowz, radial_bins):
    """Returns the radial profile of unrelated low-z galaxies, which depends
    on `N_unrelated_lowz`, which can be computed using `compute_nonsatellite_numbers()`
    from the `satellites.py` script.
    """

    surface_area = np.pi * (radial_bins ** 2 - radial_bins.min() ** 2)

    return np.array(
        [
            num
            * surface_area
            / (np.pi * (radial_bins.max() ** 2 - radial_bins.min() ** 2))
            for num in N_unrelated_lowz
        ]
    )


def bootstrap_saga_profile(radial_bins, M_r_lim=M_r_lim, N_boot=100, rng=rng):
    """Download and open SAGA sats catalog and return the bootstrapped radial profile.
    """
    saga = pd.read_csv(ROOT / "data/saga_stage2_sats.csv")
    saga = Query(f"R_ABS <= {M_r_lim}").filter(saga)

    N_SAGA = 36

    with astropy_rng:
        boot_profile_saga = (
            bootstrap(
                saga.D_PROJ.values,
                bootfunc=partial(compute_radial_profile, radial_bins=radial_bins),
                bootnum=N_boot,
            )
            / N_SAGA
        )

    SAGA_AT_36KPC = boot_profile_saga.min(1, keepdims=True)

    return boot_profile_saga - SAGA_AT_36KPC


def smooth(a, sigma=3, axis=-1, mode="nearest"):
    """Smooth a 1d signal by convolution with a Gaussian kernel
    """
    return gaussian_filter1d(a, sigma, axis=axis, mode=mode)


def plot_radial_profile_by_host_mass(
    hosts,
    sats,
    corrected=True,
    include_saga=True,
    radial_bins=np.arange(36, 300, 1),
    cumulative=True,
    normalize=False,
    areal_density=False,
    N_boot=None,
    sigma_smooth=None,
    mass_min=9.5,
    mass_max=11.0,
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
        corrected : bool
            If true, correct by dividing by the completeness (0.600) and subtracting
            the number of interlopers based on a constant false positive surface
            density (3.04 per deg^2).
        radial_bins : 1d array-like
            A list or array of (maximum) projected radii.
        cumulative : bool
            Toggles whether to plot the satellite number within a given projected
            radius versus at a given radius.
        normalize : bool
            Toggles whether to divide the curves by the total number of satellites
        areal_density : bool
            Toggles whether to divide by the annular area at each radius.
        N_boot : int or None
            The number of bootstrap resamples (for estimating uncertainties).
            Can also be `None` if boostrapping is not wanted.
        sigma_smooth : float or None
            The sigma for 1d Gaussian convolution. Can be None.
        include_saga : bool
            If true, then include the 16-84th percentile range for SAGA mean profile.
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

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    mass_bins = np.arange(mass_min, mass_max, dmass)

    dr = (radial_bins.max() - radial_bins.min()) / len(radial_bins)

    for m1, m2 in zip(mass_bins, mass_bins + dmass):
        q = Query(f"mass_GSE > {m1}", f"mass_GSE < {m2}")
        satellite_separations = q.filter(sats).sep.values

        try:
            # compute sat cumulative profile or bootstrapped profile
            if N_boot is None:
                profile = compute_radial_profile(satellite_separations, radial_bins)
                profile = profile / q.count(hosts)

                if sigma_smooth is not None:
                    profile = smooth(profile, sigma_smooth)

                # correct using completeness and interlopers
                if corrected:
                    interloper_profile = compute_interloper_profile(
                        q.filter(sats).z_NSA,
                        radial_bins,
                        interloper_surface_density=3.04,
                    )

                    nonsatellite_profile = compute_nonsatellite_profile(
                        hosts.loc[q.filter(sats).NSAID].N_unrelated_lowz, radial_bins
                    )

                    profile = (
                        profile / 0.600 - interloper_profile - nonsatellite_profile
                    )

                if areal_density:
                    # kpc^-2 --> Mpc^-2
                    surface_area = (
                        np.pi
                        * (
                            (radial_bins + dr / 2) ** 2
                            - (radial_bins.min() - dr / 2) ** 2
                        )
                        / 1e6
                    )

                    profile = profile / surface_area

                if not cumulative:
                    profile = np.gradient(profile, radial_bins, edge_order=2)

                if normalize:
                    profile /= profile.max()

                ax.plot(
                    radial_bins,
                    profile.mean(0),
                    c=mass2color((m1 + m2) / 2),
                    label=f"${m1}-{m2}$",
                    lw=3,
                )
            else:
                assert isinstance(N_boot, int), "Please enter an integer `N_boot`."
                profile_bootstrapped = bootstrap(
                    satellite_separations,
                    bootfunc=partial(compute_radial_profile, radial_bins=radial_bins),
                    bootnum=N_boot,
                )

                profile_bootstrapped = profile_bootstrapped / q.count(hosts)

                if sigma_smooth is not None:
                    profile_bootstrapped = smooth(profile_bootstrapped, sigma_smooth)

                if corrected:
                    interloper_profile = compute_interloper_profile(
                        q.filter(sats).z_NSA,
                        radial_bins,
                        interloper_surface_density=3.04,
                    )

                    interloper_profile_bootstrapped = rng.choice(
                        interloper_profile, size=N_boot, replace=True
                    )

                    nonsatellite_profile = compute_nonsatellite_profile(
                        hosts.loc[q.filter(sats).NSAID].N_unrelated_lowz, radial_bins
                    )

                    nonsatellite_profile_bootstrapped = rng.choice(
                        nonsatellite_profile, size=N_boot, replace=True
                    )

                    profile_bootstrapped = (
                        profile_bootstrapped / 0.600
                        - interloper_profile_bootstrapped
                        - nonsatellite_profile_bootstrapped
                    )

                if not cumulative:
                    profile_bootstrapped = np.gradient(
                        profile_bootstrapped, radial_bins, axis=1, edge_order=2
                    )

                if areal_density:
                    surface_area = (
                        np.pi
                        * (
                            (radial_bins + dr / 2) ** 2
                            - (radial_bins.min() - dr / 2) ** 2
                        )
                        / 1e6
                    )
                    profile_bootstrapped = profile_bootstrapped / surface_area

                if normalize:
                    profile_bootstrapped /= profile_bootstrapped.max(1, keepdims=True)

                ax.fill_between(
                    radial_bins,
                    *np.quantile(profile_bootstrapped, [0.16, 0.84], axis=0),
                    color=mass2color((m1 + m2) / 2),
                    label=f"${m1}-{m2}$",
                    lw=0,
                    alpha=0.7,
                )

        # do something useful here
        except ValueError as e:
            raise e

    if include_saga and cumulative and (not areal_density):
        boot_saga_profile = bootstrap_saga_profile(radial_bins=radial_bins, N_boot=100)

        if normalize:
            boot_saga_profile /= boot_saga_profile.max(1, keepdims=True)

        ax.fill_between(
            radial_bins,
            *np.quantile(boot_saga_profile, [0.16, 0.84], axis=0),
            color="0.5",
            lw=0,
            alpha=0.5,
            label="SAGA II",
        )

    xlabel = "$r$ [projected kpc]"
    ylabel = (
        ("Normalized " if normalize else "")
        + (r"$\Sigma_{\rm sat}$" if areal_density else r"$N_{\rm sat}$")
        + ("(r)" if not cumulative else "(<r)")
        + (" [Mpc$^{-2}$]" if areal_density else "")
    )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(alpha=0.15)

    legend_location = (
        "upper right" if (areal_density and (not cumulative)) else "upper left"
    )
    ax.legend(
        loc=legend_location, fontsize=12, title=r"log($M_★/M_\odot$)", title_fontsize=14
    )
    fig.tight_layout()
    fig.savefig(results_dir / f"plots/profiles/{fname}.png")


def plot_radial_profile_by_host_morphology(
    hosts,
    sats,
    radial_bins=np.arange(36, 300, 1),
    corrected=True,
    cumulative=True,
    normalize=False,
    areal_density=False,
    N_boot=None,
    sigma_smooth=None,
    sersic_n_low=(0, 2.5),
    sersic_n_high=(3, 6),
    mass_min=9.5,
    mass_max=11.0,
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
        corrected : bool
            If true, correct by dividing by the completeness (0.600) and subtracting
            the number of interlopers based on a constant false positive surface
            density (3.04 per deg^2).
        cumulative : bool
            Toggles whether to plot the satellite number within a given projected
            radius versus at a given radius.
        normalize : bool
            Toggles whether to divide the curves by the total number of satellites
        areal_density : bool
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

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5), dpi=300, sharey=True)
    mass_bins = np.arange(mass_min, mass_max, dmass)
    dr = (radial_bins.max() - radial_bins.min()) / len(radial_bins)

    for sersic_n_range, ax in zip([sersic_n_low, sersic_n_high], [ax1, ax2]):
        n1, n2 = sersic_n_range
        q_sersic_n = Query(f"SERSIC_N_NSA > {n1}", f"SERSIC_N_NSA < {n2}")

        for m1, m2 in zip(mass_bins, mass_bins + dmass):
            q_mass = Query(f"mass_GSE > {m1}", f"mass_GSE < {m2}")

            # combine morphology and mass queries
            q = q_sersic_n & q_mass
            satellite_separations = q.filter(sats).sep.values

            try:
                # compute sat cumulative profile or bootstrapped profile
                if N_boot is None:
                    profile = compute_radial_profile(satellite_separations, radial_bins)
                    profile = profile / q.count(hosts)

                    if sigma_smooth is not None:
                        profile = smooth(profile, sigma_smooth)

                    # correct using completeness and interlopers
                    if corrected:
                        interloper_profile = compute_interloper_profile(
                            q.filter(sats).z_NSA,
                            radial_bins,
                            interloper_surface_density=3.04,
                        )

                        nonsatellite_profile = compute_nonsatellite_profile(
                            hosts.loc[q.filter(sats).NSAID].N_unrelated_lowz,
                            radial_bins,
                        )

                        profile = (
                            profile / 0.600 - interloper_profile - nonsatellite_profile
                        )

                    if not cumulative:
                        profile = np.gradient(profile, radial_bins, edge_order=2)

                    if areal_density:
                        surface_area = (
                            np.pi
                            * (
                                (radial_bins + dr / 2) ** 2
                                - (radial_bins.min() - dr / 2) ** 2
                            )
                            / 1e6
                        )

                        profile = profile / surface_area

                    if normalize:
                        profile /= profile.max()

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
                        bootfunc=partial(
                            compute_radial_profile, radial_bins=radial_bins
                        ),
                        bootnum=N_boot,
                    )

                    profile_bootstrapped = profile_bootstrapped / q.count(hosts)

                    if sigma_smooth is not None:
                        profile_bootstrapped = smooth(
                            profile_bootstrapped, sigma_smooth
                        )

                    if corrected:
                        interloper_profile = compute_interloper_profile(
                            q.filter(sats).z_NSA,
                            radial_bins,
                            interloper_surface_density=3.04,
                        )

                        interloper_profile_bootstrapped = rng.choice(
                            interloper_profile, size=N_boot, replace=True
                        )

                        nonsatellite_profile = compute_nonsatellite_profile(
                            hosts.loc[q.filter(sats).NSAID].N_unrelated_lowz,
                            radial_bins,
                        )

                        nonsatellite_profile_bootstrapped = rng.choice(
                            nonsatellite_profile, size=N_boot, replace=True
                        )

                        profile_bootstrapped = (
                            profile_bootstrapped / 0.600
                            - interloper_profile_bootstrapped
                            - nonsatellite_profile_bootstrapped
                        )

                    if areal_density:
                        surface_area = (
                            np.pi
                            * (
                                (radial_bins + dr / 2) ** 2
                                - (radial_bins.min() - dr / 2) ** 2
                            )
                            / 1e6
                        )

                        profile_bootstrapped = profile_bootstrapped / surface_area

                    if not cumulative:
                        profile_bootstrapped = np.gradient(
                            profile_bootstrapped, radial_bins, axis=1, edge_order=2
                        )

                    if normalize:
                        profile_bootstrapped /= profile_bootstrapped.max(
                            1, keepdims=True
                        )

                    ax.fill_between(
                        radial_bins,
                        *np.quantile(profile_bootstrapped, [0.16, 0.84], axis=0),
                        color=mass2color((m1 + m2) / 2),
                        label=f"${m1}-{m2}$",
                        lw=0,
                        alpha=0.7,
                    )

            except ValueError:
                continue

        ax.grid(alpha=0.15)
        ax.text(
            0.5,
            0.92,
            r"${} < N_{{\rm Sersic}} < {}$".format(n1, n2),
            transform=ax.transAxes,
            ha="center",
            fontsize=16,
        )

    xlabel = "$r$ [projected kpc]"
    ylabel = (
        ("Normalized " if normalize else "")
        + (r"$\Sigma_{\rm sat}$" if areal_density else r"$N_{\rm sat}$")
        + ("(r)" if not cumulative else "(<r)")
        + (" [Mpc$^{-2}$]" if areal_density else "")
    )

    ax1.set_xlabel(xlabel, fontsize=12)
    ax2.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)

    legend_location = (
        "center right" if (areal_density and (not cumulative)) else "center left"
    )
    ax1.legend(
        loc=legend_location, fontsize=12, title=r"log($M_★/M_\odot$)", title_fontsize=14
    )

    fig.tight_layout()
    fig.savefig(results_dir / f"plots/profiles/{fname}.png")


def plot_radial_profile_by_magnitude_gap(
    hosts,
    sats,
    radial_bins=np.arange(36, 300, 1),
    corrected=True,
    cumulative=True,
    normalize=False,
    areal_density=False,
    N_boot=None,
    sigma_smooth=None,
    magnitude_gap_min=0,
    magnitude_gap_max=7,
    dgap=1,
    min_sats=50,
    mass_bins=[(9.5, 10), (10, 10.5), (10.5, 11)],
    fname="radial_profile-by-magnitude_gap",
):
    """Creates radial profile plots colored by magnitude gap and split by host mass.

    Parameters
        hosts : pd.DataFrame
            DataFrame of hosts, which is assumed to be pre-filtered using
            cuts at top of this script.
        sats : pd.DataFrame
            DataFrame of satellites, also assumed to be filtered.
        radial_bins : 1d array-like
            A list or array of (maximum) projected radii.
        corrected : bool
            If true, correct by dividing by the completeness (0.600) and subtracting
            the number of interlopers based on a constant false positive surface
            density (3.04 per deg^2).
        cumulative : bool
            Toggles whether to plot the satellite number within a given projected
            radius versus at a given radius.
        normalize : bool
            Toggles whether to divide the curves by the total number of satellites
        areal_density : bool
            Toggles whether to divide by the annular surface area at each radius.
        N_boot : int
            The number of bootstrap resamples (for estimating uncertainties).
            Can also be `None` if boostrapping is not wanted.
        magnitude_gap_min : float
            The minimum magnitude gap between host and brightest satellite.
        magnitude_gap_max : float
            The maximum magnitude gap between host and brightest satellite.
        dgap : float
            The size of each bin for magnitude gaps.
        min_sats : int
            The minimum number of satellites needed for plotting a radial profile.
        mass_bins : 1d list of tuples, or 2d array of shape (N, 2)
            N pairs of (mass_min, mass_max) for each panel in the plot.
        fname : str
            The name of the output figure (JPG format), to be saved in the directory
            `./results/xSAGA/plots/profiles/`.
    """

    N = len(mass_bins)

    if "magnitude_gap" not in sats.columns:
        sats = compute_magnitude_gap(sats)

    fig, axes = plt.subplots(1, N, figsize=(5 * N, 5), dpi=300, sharey=True)

    magnitude_gap_bins = np.arange(magnitude_gap_min, magnitude_gap_max, dgap)
    dr = (radial_bins.max() - radial_bins.min()) / len(radial_bins)

    for i, ([m1, m2], ax) in enumerate(zip(mass_bins, axes.flat)):
        q_mass = Query(f"mass_GSE > {m1}", f"mass_GSE < {m2}")

        for mag1, mag2 in zip(magnitude_gap_bins, magnitude_gap_bins + dgap):
            q_gap = Query(f"magnitude_gap > {mag1}", f"magnitude_gap < {mag2}")

            # combine morphology and mass queries
            q = q_mass & q_gap
            satellite_separations = q.filter(sats).sep.values

            # skip if too few satellites to form a profile
            if (len(satellite_separations) < min_sats) or (mag2 > 2 * (i + 1)):
                continue

            try:
                # compute sat cumulative profile or bootstrapped profile
                if N_boot is None:
                    profile = compute_radial_profile(satellite_separations, radial_bins)
                    profile = profile / len(q.filter(sats).NSAID.unique())

                    if sigma_smooth is not None:
                        profile = smooth(profile, sigma_smooth)

                    # correct using completeness and interlopers
                    if corrected:
                        interloper_profile = compute_interloper_profile(
                            q.filter(sats).z_NSA,
                            radial_bins,
                            interloper_surface_density=3.04,
                        )

                        nonsatellite_profile = compute_nonsatellite_profile(
                            hosts.loc[q.filter(sats).NSAID].N_unrelated_lowz,
                            radial_bins,
                        )

                        profile = (
                            profile / 0.600 - interloper_profile - nonsatellite_profile
                        )

                    if areal_density:
                        surface_area = (
                            np.pi
                            * (
                                (radial_bins + dr / 2) ** 2
                                - (radial_bins.min() - dr / 2) ** 2
                            )
                            / 1e6
                        )

                        profile = profile / surface_area

                    if not cumulative:
                        profile = np.gradient(profile, radial_bins, edge_order=2)

                    if normalize:
                        profile /= profile.max()

                    ax.plot(
                        radial_bins,
                        profile,
                        c=gap2color((mag1 + mag2) / 2),
                        label=f"${m1}-{m2}$",
                        lw=3,
                    )
                else:
                    assert isinstance(N_boot, int), "Please enter an integer `N_boot`."
                    profile_bootstrapped = bootstrap(
                        satellite_separations,
                        bootfunc=partial(
                            compute_radial_profile, radial_bins=radial_bins
                        ),
                        bootnum=N_boot,
                    )

                    profile_bootstrapped = profile_bootstrapped / len(
                        q.filter(sats).NSAID.unique()
                    )

                    if sigma_smooth is not None:
                        profile_bootstrapped = smooth(
                            profile_bootstrapped, sigma_smooth
                        )

                    if corrected:
                        interloper_profile = compute_interloper_profile(
                            q.filter(sats).z_NSA,
                            radial_bins,
                            interloper_surface_density=3.04,
                        )

                        interloper_profile_bootstrapped = rng.choice(
                            interloper_profile, size=N_boot, replace=True
                        )

                        nonsatellite_profile = compute_nonsatellite_profile(
                            hosts.loc[q.filter(sats).NSAID].N_unrelated_lowz,
                            radial_bins,
                        )

                        nonsatellite_profile_bootstrapped = rng.choice(
                            nonsatellite_profile, size=N_boot, replace=True
                        )

                        profile_bootstrapped = (
                            profile_bootstrapped / 0.600
                            - interloper_profile_bootstrapped
                            - nonsatellite_profile_bootstrapped
                        )

                    if areal_density:
                        surface_area = (
                            np.pi
                            * (
                                (radial_bins + dr / 2) ** 2
                                - (radial_bins.min() - dr / 2) ** 2
                            )
                            / 1e6
                        )

                        profile_bootstrapped = profile_bootstrapped / surface_area

                    if not cumulative:
                        profile_bootstrapped = np.gradient(
                            profile_bootstrapped, radial_bins, axis=1, edge_order=2
                        )

                    if normalize:
                        profile_bootstrapped /= profile_bootstrapped.max(
                            1, keepdims=True
                        )

                    ax.fill_between(
                        radial_bins,
                        *np.quantile(profile_bootstrapped, [0.16, 0.84], axis=0),
                        color=gap2color((mag1 + mag2) / 2),
                        label=f"${mag1}-{mag2}$",
                        lw=0,
                        alpha=0.7,
                    )

            except ValueError:
                continue

        ax.grid(alpha=0.15)
        ax.text(
            0.5,
            0.92,
            r"${0:g} < \log(M_★/M_\odot) < {1:g}$".format(m1, m2),
            transform=ax.transAxes,
            ha="center",
            fontsize=16,
        )
        ax.set_xlabel("$r$ [projected kpc]", fontsize=12)

    ylabel = (
        ("Normalized " if normalize else "")
        + (r"$\Sigma_{\rm sat}$" if areal_density else r"$N_{\rm sat}$")
        + ("(r)" if not cumulative else "(<r)")
        + (" [Mpc$^{-2}$]" if areal_density else "")
    )

    axes.flat[0].set_ylabel(ylabel, fontsize=12)
    fig.legend(
        *axes.flat[-1].get_legend_handles_labels(),
        loc="upper left",
        fontsize=14,
        title=r"$\Delta m_{r,*}$",
        title_fontsize=16,
        framealpha=0,
        borderpad=2,
        borderaxespad=2,
    )

    fig.tight_layout()
    fig.savefig(results_dir / f"plots/profiles/{fname}.png")


if __name__ == "__main__":

    hosts, sats = load_hosts_and_sats()

    # impose cuts
    hosts = HOST_QUERY.filter(hosts)
    sats = (HOST_QUERY & SAT_QUERY).filter(sats)

    # isolate hosts and sats
    hosts = isolate_hosts(hosts, delta_mass=0.0, delta_z=0.005, delta_d=1.0)
    sats = sats[sats.NSAID.isin(hosts.index)].copy()

    print(len(hosts), len(sats), hosts.n_corr_sats_in_300kpc.sum())

    # use bootstraps
    N_boot = 100

    # host mass
    # =========

    plot_radial_profile_by_host_mass(
        hosts,
        sats,
        corrected=True,
        dmass=0.25,
        N_boot=N_boot,
        fname="radial_profile-by-host_mass",
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
        areal_density=True,
        N_boot=N_boot,
        fname="areal_density_profile-by-host_mass",
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
        areal_density=True,
        N_boot=N_boot,
        fname="areal_density_pdf-by-host_mass",
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
        radial_bins=np.arange(36, 300, 20),
        cumulative=False,
        areal_density=True,
        N_boot=N_boot,
        fname="areal_density_pdf-by-host_morphology",
    )

    # magnitude gap
    # =============
    plot_radial_profile_by_magnitude_gap(
        hosts,
        sats,
        radial_bins=np.arange(36, 300, 1),
        N_boot=N_boot,
        fname="radial_profile-by-magnitude_gap",
    )

    # exploring concentrations
    # ========================

    plot_radial_profile_by_host_morphology(
        hosts,
        sats,
        radial_bins=np.arange(36, 300, 10),
        normalize=True,
        cumulative=True,
        N_boot=N_boot,
        fname="normalized_radial_profile-by-host_morphology",
    )

    plot_radial_profile_by_magnitude_gap(
        hosts,
        sats,
        radial_bins=np.arange(36, 300, 10),
        normalize=True,
        cumulative=True,
        N_boot=N_boot,
        fname="normalized_radial_profile-by-magnitude_gap",
    )

    # radial profiles by brightest satellites
    # =======================================

    sats["r_abs"] = sats.r0 - cosmo.distmod(sats.z_NSA)
    sats["magnitude_gap"] = sats.r_abs - sats.M_r_NSA

    sats_subset = sats.join(
        sats[sats.magnitude_gap > 0].groupby("NSAID").M_r.min().rename("M_r_sat"),
        on="NSAID",
    )

    hosts_subset = hosts.join(
        sats_subset.set_index("NSAID", drop=True).M_r_sat, on="NSAID", how="left"
    ).drop_duplicates()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300, sharey=True)

    N_boot = 30
    radial_bins = np.arange(36, 300, 1)
    mass_bins = np.arange(9.5, 11, 0.5)
    M_r_sat_bins = np.arange(-20, -15, 1)

    for ax, m1, m2 in zip(axes.flat, mass_bins, mass_bins + 0.5):
        q_mass = Query(f"mass_GSE > {m1}", f"mass_GSE < {m2}")
        for r1, r2 in zip(M_r_sat_bins, M_r_sat_bins + 1):
            if r1 < -2 * m1 + 1:
                continue

            q = Query(f"M_r_sat > {r1}", f"M_r_sat < {r2}") & q_mass
            satellite_separations = q.filter(sats_subset).sep.values

            profile_bootstrapped = bootstrap(
                satellite_separations,
                bootfunc=partial(compute_radial_profile, radial_bins=radial_bins),
                bootnum=N_boot,
            )

            profile_bootstrapped = profile_bootstrapped / q.count(hosts_subset)

            interloper_profile = compute_interloper_profile(
                q.filter(sats_subset).z_NSA,
                radial_bins,
                interloper_surface_density=3.04,
            )

            interloper_profile_bootstrapped = rng.choice(
                interloper_profile, size=N_boot, replace=True
            )

            nonsatellite_profile = compute_nonsatellite_profile(
                hosts.loc[q.filter(sats_subset).NSAID].N_unrelated_lowz, radial_bins
            )

            nonsatellite_profile_bootstrapped = rng.choice(
                nonsatellite_profile, size=N_boot, replace=True
            )

            profile_bootstrapped = (
                profile_bootstrapped / 0.600
                - interloper_profile_bootstrapped
                - nonsatellite_profile_bootstrapped
            )

            # xSAGA
            ax.fill_between(
                radial_bins,
                *np.quantile(profile_bootstrapped, [0.16, 0.84], axis=0),
                color=mass2color(
                    (r1 + r2) / 2, cmap=cmr.rainforest_r, mass_min=-21, mass_max=-15
                ),
                label=f"${r1:g}$ to ${r2:g}$",
                lw=0,
                alpha=0.7,
                zorder=3,
            )
        ax.grid(alpha=0.15)
        ax.set_xlabel("$r$ [projected kpc]", fontsize=12)
        ax.text(
            0.5,
            0.92,
            r"${0:g} < \log(M_★/M_\odot) < {1:g}$".format(m1, m2),
            transform=ax.transAxes,
            ha="center",
            fontsize=16,
        )
    axes.flat[0].set_ylabel(r"$N_{\rm sat}(<r)$", fontsize=12)
    fig.legend(
        *axes.flat[-1].get_legend_handles_labels(),
        loc="upper left",
        fontsize=14,
        title="$M_{r,*}$",
        title_fontsize=16,
        framealpha=0,
        borderpad=2,
        borderaxespad=2,
    )
    fig.tight_layout()
    fig.savefig(results_dir / "plots/profiles/radial_profile-by-brightest_sat.png")
