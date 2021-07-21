"""
John F. Wu
2021-07-21

Scripts for evaluating CNN performance.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import purity, completeness, accuracy


ROOT = Path(__file__).resolve().parent.parent.parent
results_dir = ROOT / "results/xSAGA"


def compare_north_and_south():
    """Compare p_CNN thresholds in the Legacy Survey North and South, which have
    different imaging systematics.

    """
    pass


def compare_magnitude(predictions, labels, r):
    """Compare metrics as a function of apparent magnitude.
    """
    pass


def compare_surface_brightness(predictions, labels, mu_eff):
    pass


def compare_color(predictions, labels, gmr):
    pass
