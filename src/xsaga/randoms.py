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

from satellites import assign_satellites_to_hosts

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

MAKE_PLOTS = False
EXT = "png"

ROOT = Path(__file__).resolve().parent.parent.parent
results_dir = ROOT / "results/xSAGA"


def generate_random_hosts(hosts, ra_range=[120, 240], dec_range=[0, 50]):
    """Generate a version of the hosts catalog except with random coordinates.
    """

    hosts_rand = hosts.sample(N_rand).copy()

    ra_min, ra_max = ra_range
    dec_min, dec_max = dec_range

    hosts_rand["ra_NSA"] = (np.random.random(size=N_rand)) * (ra_max - ra_min) + ra_min
    hosts_rand["dec_NSA"] = np.rad2deg(
        np.pi / 2
        - np.arccos(
            np.random.random(size=N_rand) * np.deg2rad(dec_max - dec_min) + dec_min
        )
    )

    return hosts_rand


if __name__ == "__main__":

    hosts_file = results_dir / "hosts-nsa.parquet"
    hosts = pd.read_parquet(hosts_file)

    hosts_rand = generate_random_hosts(hosts, ra_range=[120, 240], dec_range=[0, 50])
