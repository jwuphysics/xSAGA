"""
John F. Wu (2021)

Utility functions for xSAGA analysis and plot-smithing.
"""
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import cmasher as cmr
from functools import partial

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def _deg2kpc(theta, z):
    """Convenience function for converting angular distances [deg] to physical
    distances [kpc] at a given redshift. Used in `ax.secondary_xaxis` for plotting.
    """
    return (theta * u.deg * cosmo.kpc_proper_per_arcmin(z=z)).to("kpc").value


def _kpc2deg(dist, z):
    """Convenience function for converting physical distances [kpc] to angular
    distances [deg] at a given redshift. Used in `ax.secondary_xaxis` for plotting.
    """
    return (dist * u.kpc / cosmo.kpc_proper_per_arcmin(z=z)).to("deg").value


deg2kpc = partial(_deg2kpc, z=0.03)
kpc2deg = partial(_kpc2deg, z=0.03)


def mass2color(mass, cmap=cmr.ember, mass_min=9.5, mass_max=11.0):
    """Convenience function for mapping a stellar mass, normalized to some mass range,
    to a color determined by colormap `cmap`.
    """
    return cmap((mass - mass_min) / (mass_max - mass_min))


def gap2color(gap, cmap=cmr.savanna_r, gap_min=-1, gap_max=6):
    """Convenience function for mapping a magnitude gap, normalized to some gap range,
    to a color determined by colormap `cmap`.
    """
    return cmap((gap - gap_min) / (gap_max - gap_min))
