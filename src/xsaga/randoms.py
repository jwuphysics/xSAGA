"""
John F. Wu
2021-07-19

A collection of methods for creating catalogs of satellites around random
hosts in order to estimate xSAGA backgrounds.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from easyquery import Query
from pathlib import Path

from satellites import assign_satellites_to_hosts, count_satellites_per_host

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

MAKE_PLOTS = False
EXT = "png"

ROOT = Path(__file__).resolve().parent.parent.parent
results_dir = ROOT / "results/xSAGA"


def generate_random_hosts(hosts, ra_range=[120, 240], dec_range=[0, 50]):
    """Generate a version of the hosts catalog except with random coordinates.
    """

    ra_min, ra_max = ra_range
    dec_min, dec_max = dec_range

    N_rand = Query(
        f"ra_NSA > {ra_min}",
        f"ra_NSA < {ra_max}",
        f"dec_NSA > {dec_min}",
        f"dec_NSA < {dec_max}",
    ).count(hosts)

    hosts_rand = hosts.sample(N_rand).copy()

    hosts_rand["ra_NSA"] = (np.random.random(size=N_rand)) * (ra_max - ra_min) + ra_min
    hosts_rand["dec_NSA"] = np.rad2deg(
        np.pi / 2
        - np.arccos(
            np.random.random(size=N_rand) * np.deg2rad(dec_max - dec_min) + dec_min
        )
    )

    # remove columns that don't apply, since host-sat assignment hasn't been
    # done for randoms yet
    excess_cols = [col for col in hosts_rand.columns if col.startswith("n_sats_in")]
    hosts_rand.drop(excess_cols, axis=1, inplace=True)

    return hosts_rand


if __name__ == "__main__":

    # load low-z, massive hosts
    # ================
    hosts_file = results_dir / "hosts-nsa.parquet"
    hosts = pd.read_parquet(hosts_file)

    # create random host catalog
    # ==========================
    hosts_rand_file = results_dir / "hosts_rand-nsa.parquet"
    try:
        hosts_rand = pd.read_parquet(hosts_rand_file)
    except (FileNotFoundError, OSError):
        hosts_rand = generate_random_hosts(
            hosts, ra_range=[120, 240], dec_range=[0, 50]
        )
        hosts_rand.to_parquet(hosts_rand_file)

    # prevent any extra errors
    del hosts

    # load lowz with p_cnn_thresh=0.5
    # ===============================
    lowz_file = results_dir / "lowz-p0_5.parquet"
    try:
        lowz = pd.read_parquet(lowz_file)
    except (FileNotFoundError, OSError) as e:
        print(f"Low-z catalog {lowz_file} does not exist!")
        raise e

    # assign satellites to host_rand
    # ==============================
    sats_rand_file = results_dir / "sats_rand-nsa_p0_5.parquet"
    try:
        sats_rand = pd.read_parquet(sats_rand_file)
    except (FileNotFoundError, OSError):
        sats_rand = assign_satellites_to_hosts(
            hosts_rand, lowz, rank_by="mass_GSE", z_min=0.005, savefig=MAKE_PLOTS
        )
        sats_rand.to_parquet(sats_rand_file)

    # count number of sats in each random host
    # ========================================
    hosts_rand_file = results_dir / "hosts_rand-nsa.parquet"
    try:
        hosts_rand = pd.read_parquet(hosts_rand_file)
        assert "n_sats_in_300kpc" in hosts_rand.columns
    except AssertionError:
        hosts_rand = count_satellites_per_host(
            hosts_rand, sats_rand, savefig=MAKE_PLOTS
        )
        hosts_rand.to_parquet(hosts_rand_file)
