"""John F Wu
2021-08-03

Scripts for quantifying the uncertainties, biases, and errors in satellite catalogs.
"""

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from easyquery import Query
from pathlib import Path

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

ROOT = Path(__file__).resolve().parent.parent
results_dir = ROOT / "results/xSAGA"


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


if __name__ == "__main__":
    hosts = pd.read_parquet(results_dir / "hosts-nsa.parquet")
    hosts_rand = pd.read_parquet(results_dir / "hosts_rand-nsa.parquet")

    p_host_chance = probability_host_alignments(hosts, different_redshifts=True)
    p_host_overlap = probability_host_alignments(hosts, different_redshifts=False)
    print(
        f"The fraction of chance overlaps in real hosts is {p_host_chance:.4f} and\n"
        f"the fraction same-redshift overlaps is {p_host_overlap:.4f}.\n"
    )

    p_host_rand_chance = probability_host_alignments(
        hosts_rand, different_redshifts=True
    )
    p_host_rand_overlap = probability_host_alignments(
        hosts_rand, different_redshifts=False
    )
    print(
        f"The fraction of chance overlaps in random hosts is {p_host_rand_chance:.4f}\n"
        f"and the fraction at similar redshifts is {p_host_rand_overlap:.4f}.\n"
    )
