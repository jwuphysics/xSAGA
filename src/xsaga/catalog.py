"""
John F. Wu
2021-07-20

Scripts for creating catalogs and figures to be in the first xSAGA paper.
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


def two_point_statistics():
    pass


def redshift_clustering():
    pass


if __name__ == "__main__":
    pass
