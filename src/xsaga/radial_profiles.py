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
import cmasher as cmr
from easyquery import Query
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


def compute_radial_cdf(sats, radial_bins):
    """Counts the cumulative number of satellites within a list of radial bins.

    Parameters
        sats : pd.DataFrame
            A DataFrame of satellites, presumably already filtered by host property,
            and with a column `sep` that is the distance in pkpc from matched host.
        radial_bins : 1d array-like
            A list or array of (maximum) radii.

    Returns
        counts : 1d array
            An array of cumulative satellite counts for each bin in `radial_bins`.
    """
    return np.array([Query(f"sep < {r}").count(sats) for r in radial_bins])


def plot_radial_profile_by_host_mass(
    hosts,
    sats,
    radial_bins=np.arange(36, 300, 1),
    normalize=False,
    cumulative=True,
    mass_min=9.5,
    mass_max=11.5,
    dmass=0.5,
    fname="radial_cdf-by-host_mass",
):
    """Creates a radial profile plot for satellites.
    """

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
    mass_bins = np.arange(mass_min, mass_max, dmass)

    for m1, m2 in zip(mass_bins, mass_bins + dmass):
        q = Query(f"mass_GSE > {m1}", f"mass_GSE < {m2}")

        # either normalize satellite counts or divide by number of hosts
        cdf = compute_radial_cdf(q.filter(sats), radial_bins)
        profile = cdf / (q.count(sats) if normalize else q.count(hosts))

        if not cumulative:
            profile = np.gradient(profile, radial_bins)

        ax.plot(
            radial_bins,
            profile,
            c=mass2color((m1 + m2) / 2),
            label=f"${m1}-{m2}$",
            lw=3,
        )

    xlabel = "Distance [pkpc]"
    ylabel = r"$N_{\rm sat}(<r)$" if cumulative else r"$\frac{dN_{\rm sat}}{dr}(<r)$"
    ylabel = ("Normalized " if normalize else "") + ylabel

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(alpha=0.15)
    ax.legend(fontsize=12)
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

    plot_radial_profile_by_host_mass(
        hosts,
        sats,
        dmass=0.25,
        normalize=False,
        fname="radial_profile-by-host_mass",
    )
    plot_radial_profile_by_host_mass(
        hosts,
        sats,
        dmass=0.25,
        normalize=True,
        fname="normalized_radial_profile-by-host_mass",
    )

    plot_radial_profile_by_host_mass(
        hosts,
        sats,
        dmass=0.5,
        normalize=False,
        cumulative=False,
        fname="dNdr-by-host_mass",
    )
    plot_radial_profile_by_host_mass(
        hosts,
        sats,
        dmass=0.5,
        normalize=True,
        cumulative=False,
        fname="normalized_dNdr-by-host_mass",
    )
