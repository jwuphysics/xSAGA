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


def load_hosts(impose_cut=True):
    """Loads the host catalog with option to host impose mass and redshift cuts.
    """
    hosts = pd.read_parquet(results_dir / "hosts-nsa.parquet")

    if impose_cut:
        return HOST_QUERY.filter(hosts)
    return hosts


def load_sats(impose_cut=True):
    """Loads the satellite catalog, with generated `M_r` column, and with option to
    impose cuts using cuts defined above.
    """
    sats = pd.read_parquet(results_dir / "sats-nsa_p0_5.parquet")
    sats["M_r"] = sats.r0 - cosmo.distmod(sats.z_NSA).value

    if impose_cut:
        return (SAT_QUERY & HOST_QUERY).filter(sats)
    return sats


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


def plot_radial_cdf_by_host_mass(
    hosts,
    sats,
    radial_bins=np.arange(36, 300, 1),
    normalize=False,
    mass_min=9.5,
    mass_max=11.0,
    dmass=0.5,
    fname="radial_profiles-by-host_mass",
):
    """Creates a radial profile plot for satellites.
    """

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
    mass_bins = np.arange(mass_min, mass_max, dmass)

    for m1, m2 in zip(mass_bins, mass_bins + dmass):
        q = Query(f"mass_GSE > {m1}", f"mass_GSE < {m2}")

        # total satellite counts
        cdf = compute_radial_cdf(q.filter(sats), radial_bins)

        # either normalize to unity at max radius, or divide by number of hosts
        if normalize:
            cdf = cdf / q.count(sats)
        else:
            cdf = cdf / q.count(hosts)

        ax.plot(
            radial_bins, cdf, c=mass2color((m1 + m2) / 2), label=f"${m1}-{m2}$", lw=3
        )

    ax.set_xlabel("Distance [pkpc]", fontsize=12)
    ax.set_ylabel(r"$N_{\rm sat}(<r)$", fontsize=12)
    ax.grid(alpha=0.15)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(results_dir / f"plots/profiles/{fname}.png")


if __name__ == "__main__":

    hosts = load_hosts(impose_cut=False)
    sats = load_sats(impose_cut=False)

    plot_radial_cdf_by_host_mass(
        hosts, sats, normalize=False, fname="radial_profiles-by-host_mass"
    )
    plot_radial_cdf_by_host_mass(
        hosts, sats, normalize=False, fname="normalized-radial-profiles-by-host_mass"
    )
