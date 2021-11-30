"""
John F. Wu (2021)

A collection of methods to catalog xSAGA satellites and assign them to hosts.
This file can produce the following catalogs: `hosts-nsa`, `lowz`, and `sats`,
along with several diagnostic or results figures.

We don't yet have a strict naming convention, but it would be helpful to
stick to something like this:

    Name                    Description
    ----                    -----------
    hosts-nsa.parquet       hosts based on NASA-Sloan Atlas
    hosts-sga.parquet       hosts based on Siena Galaxy Atlas
    hosts-nsga.parquet      hosts based on NSA crossmatched with SGA
    lowz-p0_5.parquet       low-z catalog for p_CNN > 0.5
    lowz-p0_35.parquet      low-z catalog for p_CNN > 0.5
    sats-nsa-p0_5.parquet   satellites with p_CNN > 0.5 assigned to NSA hosts
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

MAKE_PLOTS = True
EXT = "png"

ROOT = Path(__file__).resolve().parent.parent
results_dir = ROOT / "results/xSAGA"


def compute_surface_brightness(df):
    """Returns a column of surface brightnesses in the r band.

    For use with the satellite DataFrame.
    """

    return df.r0 + 2.5 * np.log10(2 * np.pi * df.R_eff ** 2)


def compute_gmr_color(df):
    """Returns a column of g-r compute_gmr_color

    Fro use with the satellite DataFrame.
    """
    return df.g0 - df.r0


def load_NSA():
    """Returns dataframe with select NASA-Sloan Atlas columns.

    Note: loading NSA and the GSE catalog requires ~5GB of RAM, and
    will likely crash on a laptop with 8GB fo memory.
    """

    # convenience function for FITS files to DataFrames
    def byteorder(row):
        return row.byteswap().newbyteorder()

    nsa = fits.getdata(ROOT / "data/nsa_v1_0_1.fits")
    nsa = pd.DataFrame(
        {
            "NSAID": byteorder(nsa.NSAID),
            "z_NSA": byteorder(nsa.Z),
            "ra_NSA": byteorder(nsa.RA),
            "dec_NSA": byteorder(nsa.DEC),
            "M_r_NSA": byteorder(nsa.ELPETRO_ABSMAG[:, 4]),
            "M_g_NSA": byteorder(nsa.ELPETRO_ABSMAG[:, 3]),
            "mass_NSA": byteorder(np.log10(nsa.ELPETRO_MASS)),
            "SERSIC_N_NSA": byteorder(nsa.SERSIC_N),
            "SERSIC_BA_NSA": byteorder(nsa.SERSIC_BA),
            "SERSIC_PHI_NSA": byteorder(nsa.SERSIC_PHI),
            "ELPETRO_BA_NSA": byteorder(nsa.ELPETRO_BA),
            "PLATE": byteorder(nsa.PLATE),
            "MJD": byteorder(nsa.MJD),
            "FIBERID": byteorder(nsa.FIBERID),
        }
    )

    # load value-added catalogs for stellar masses
    gse = fits.getdata(ROOT / "data/galSpecExtra-dr8.fits")
    gse = pd.DataFrame(
        {
            "PLATE": byteorder(gse.PLATEID),
            "MJD": byteorder(gse.MJD),
            "FIBERID": byteorder(gse.FIBERID),
            "mass_GSE": byteorder(gse.LGM_TOT_P50),
            "mass_err_GSE": 0.5
            * (byteorder(gse.LGM_TOT_P84) - byteorder(gse.LGM_TOT_P16)),
        }
    )

    join_columns = ["PLATE", "MJD", "FIBERID"]
    nsa = nsa.join(gse.set_index(join_columns), on=join_columns, how="left")

    return nsa.set_index("NSAID")


def load_lowz(p_cnn_thresh=0.5):
    """Returns dataframe of lowz candidates in xSAGA.
    """

    df = pd.read_csv(ROOT / "results/predictions-dr9.csv")
    df = df[df.p_CNN > p_cnn_thresh].copy()

    # remove NaN coordinates
    df = Query("ra == ra", "dec == dec").filter(df)

    df["mu_eff"] = compute_surface_brightness(df)
    df["gmr"] = compute_gmr_color(df)

    return df


def plot_satellite_positions(
    sats,
    colored_by=None,
    sized_by=None,
    colormap="viridis",
    figname=None,
    vmin=None,
    vmax=None,
    colorbar_label=None,
    xlim=None,
    ylim=None,
):
    """Make a scatter plot of satellites by position.
    """

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=300)

    sc = ax.scatter(
        sats.ra,
        sats.dec,
        edgecolor="none",
        s=1,
        c=sats[colored_by],
        vmin=vmin,
        vmax=vmax,
        cmap=colormap,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")

    fig.colorbar(sc, label=colorbar_label)
    fig.tight_layout()

    if figname is not None:
        fig.savefig(results_dir / f"plots/{figname}.{EXT}", format=EXT)
    else:
        fig.savefig(
            results_dir / f"plots/satellite_positions-by-{colored_by}.{EXT}", format=EXT
        )


def remove_hosts_from_lowz(hosts, lowz, match_sep=1 * u.arcsec, savefig=False):
    """Filters the `lowz` catalog by crossmatching with `hosts`, and returns
    `lowz` as well as `hosts` crossmatched with `lowz` matches.
    """
    host_coords = SkyCoord(hosts.ra_NSA, hosts.dec_NSA, unit="deg")
    lowz_coords = SkyCoord(lowz.ra, lowz.dec, unit="deg")

    host_idx, sep, _ = lowz_coords.match_to_catalog_sky(host_coords)

    # plot matches
    if savefig:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=300)
        ax.hist(sep.to("arcsec").value, bins=np.logspace(-2, 5, num=100))
        ax.set_xscale("log")
        ax.set_xlabel("Separation [arcsec]")
        ax.set_title("Separation between low-z candidates and NSA galaxies")
        ax.grid(alpha=0.2)
        fig.tight_layout()

        fig.savefig(results_dir / f"plots/host-lowz-separations.{EXT}", format=EXT)

    lowz_match = sep < match_sep
    host_idx_match = host_idx[lowz_match]

    hosts_x_lowz = pd.concat(
        (hosts.iloc[host_idx_match].reset_index(), lowz[lowz_match].reset_index()),
        axis=1,
    ).set_index("NSAID", drop=True)

    lowz_not_hosts = lowz[~lowz_match]

    return lowz_not_hosts, hosts_x_lowz


def assign_satellites_to_hosts(
    hosts,
    lowz,
    z_min=0.005,
    rank_by="mass_GSE",
    descending_order=True,
    sep_min=36.0,
    sep_max=300.0,
    savefig=False,
):
    """Match lowz galaxies to hosts in order to determine satellite systems.
    Each host is matched to all surrounding satellites, and satellites
    at distances greater than `sep_max` are removed. Finally, the satellites
    are sorted by `rank_by` (e.g., descending host mass), such that the
    most massive hosts are listed first, and duplicate matches are removed
    such that the first instance is kept.

    Parameters:
        hosts : DataFrame
            Catalog of hosts generated from NSA or SGA catalog
        lowz : DataFrame
            Catalog of lowz galaxies used to preselecting satellites
        z_min : float
            Minimum redshift considered
        rank_by : str [column name of hosts]
            The column to rank hosts by (in descending_order)
        descending_order : bool
            Whether to rank column in descending order
        sep_min : float [kpc]
            The minimum satellite separation for a host, designed to minimize
            shredded host contaminants
        sep_max : float [kpc]
            The maximum satellite separation (virial radius) for a host
        savefig : bool
            Save a figure of host-low-z and host-satellite separations

    Returns:
        sats :  DataFrame
            The subset of lowz that can be assigned to a host.
    """

    hosts = Query(f"z_NSA >= {z_min}").filter(hosts)

    lowz_coords = SkyCoord(lowz.ra, lowz.dec, unit="deg")
    host_coords = SkyCoord(hosts.ra_NSA, hosts.dec_NSA, unit="deg")

    max_seplimit = (sep_max * u.kpc / cosmo.kpc_proper_per_arcmin(z_min)).to(u.arcmin)
    lowz_idx, host_idx, angsep, _ = host_coords.search_around_sky(
        lowz_coords, seplimit=max_seplimit
    )

    sep = (angsep * cosmo.kpc_proper_per_arcmin(hosts.iloc[host_idx].z_NSA)).to(u.kpc)

    # first add satellite-host separation as a column, and then combine with hosts
    sats = pd.concat(
        [
            pd.concat(
                [
                    lowz.iloc[lowz_idx].reset_index(drop=True),
                    pd.Series(sep, name="sep"),
                ],
                axis=1,
            ),
            hosts.iloc[host_idx].reset_index(),
        ],
        axis=1,
    )

    sats = sats.sort_values("mass_GSE", ascending=False).set_index("objID")
    sats = Query(f"sep > {sep_min}", f"sep <= {sep_max}").filter(sats)
    sats = sats[~sats.index.duplicated(keep="first")]

    if savefig:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=300)
        ax.hist(sep, bins=100, range=[0, sep_max], label="All low-$z$")
        ax.hist(sats.sep, bins=100, range=[0, sep_max], label="Satellites")
        ax.set_xlabel("Distance [pkpc]")
        ax.set_ylabel("Number")
        ax.grid(alpha=0.2)
        ax.legend(framealpha=0)
        fig.tight_layout()
        fig.savefig(results_dir / f"plots/host-sat-separations.{EXT}", format=EXT)

    return sats


def count_satellites_per_host(
    hosts,
    sats,
    column_name="n_sats_in_300kpc",
    fname="satellite_counts_per_host",
    savefig=False,
):
    """Plot a histogram of satellites per host, including hosts with no
    matched satellites.
    """

    counts = (
        sats.value_counts("NSAID")
        .append(
            pd.Series(
                {nsaid: 0 for nsaid in hosts.index[~hosts.index.isin(sats.NSAID)]}
            )
        )
        .rename(column_name)
    )

    if savefig:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)

        ax.hist(counts, bins=50, range=[0, 50], log=True)
        ax.set_xlabel("Satellites per host")
        ax.set_ylabel("Number of hosts")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(results_dir / f"plots/{fname}.{EXT}", format=EXT)

    hosts = hosts.reset_index().join(counts, on="NSAID").set_index("NSAID")

    return hosts


def count_corrected_satellites_per_host(
    hosts,
    sats,
    column_name="n_corr_sats_in_300kpc",
    completeness=0.600,
    interloper_surface_density=3.04,
    nonsatellite_volume_density=0.0142,
    fname="corrected-satellite_counts_per_host",
    savefig=False,
):
    """Computes the number of corrected satellites per host by dividing by a
    completeness term and subtracting a number of interlopers and non-satellite
    contaminants that scale with the surface area of the halo.

    Note: nonsatellite_surface_density = 9.32 is an alternative method for estimating
    non-satellite contaminants.
    """

    counts_col = "n_sats_in_300kpc"
    counts = (
        hosts[counts_col]
        if counts_col in hosts.columns
        else count_satellites_per_host(hosts, sats)[counts_col]
    )

    sq_deg_per_sq_kpc = (
        cosmo.arcsec_per_kpc_proper(hosts.z_NSA).to(u.deg / u.kpc).value
    ) ** 2
    surface_area = np.pi * (300 ** 2 - 36 ** 2) * sq_deg_per_sq_kpc

    corrected_counts = counts / completeness - interloper_surface_density * surface_area

    if nonsatellite_volume_density is not None:
        N_unrelated_lowz = compute_nonsatellite_numbers(
            hosts.z_NSA, volume_density=nonsatellite_volume_density
        )
        # hosts = hosts.join(
        #     pd.Series(N_unrelated_lowz, name="N_unrelated_lowz", index=hosts.index)
        # )

        corrected_counts = corrected_counts - N_unrelated_lowz

    hosts = hosts.join(corrected_counts.rename(column_name, axis=1), on="NSAID")

    return hosts


def compute_nonsatellite_numbers(redshifts, delta_z=0.005, volume_density=0.0142):
    """Compute the number of low-z galaxies that *aren't* satellites, for hosts
    at given redshifts. `volume_density` should be in Mpc^-3.
    """
    z_upper = redshifts + delta_z
    z_lower = redshifts - delta_z
    z_upper = np.where(z_upper > 0.03, 0.03, z_upper)
    z_lower = np.where(z_lower < 0, 0, z_lower)

    total_volumes = (
        cosmo.comoving_volume(0.03) - cosmo.comoving_volume(z_upper)
    ) / 1.03 ** 3 + (cosmo.comoving_volume(z_lower)) / (1 + z_lower) ** 3
    volumes = (
        ((cosmo.kpc_proper_per_arcmin(redshifts) / (300 * u.kpc)) ** -2)
        .to(u.steradian)
        .value
        / (4 * np.pi)
        * total_volumes
    )

    return volume_density * volumes


def compute_magnitude_gap(sats):
    """Computes the r-band magnitude gap of host and brightest satellite.
    """

    sats["M_r"] = sats.r0 - cosmo.distmod(sats.z_NSA).value
    sats["magnitude_gap"] = sats.M_r - sats.M_r_NSA

    # only keep minimum value
    sats["magnitude_gap"] = sats.groupby("NSAID").magnitude_gap.transform(np.min)

    return sats


if __name__ == "__main__":

    # load hosts
    # ==========
    print("Loading NSA hosts")
    hosts_file = results_dir / "hosts-nsa.parquet"
    try:
        nsa = pd.read_parquet(hosts_file)
    except (FileNotFoundError, OSError):
        nsa = load_NSA()
        nsa.to_parquet(results_dir / "hosts-nsa.parquet")

    # load lowz candidates
    # ====================
    print("Loading low-z candidates")
    lowz_file = results_dir / "lowz-p0_5.parquet"
    try:
        lowz = pd.read_parquet(lowz_file)
    except (FileNotFoundError, OSError):
        lowz = load_lowz(p_cnn_thresh=0.5)
        lowz.to_parquet(lowz_file)

    # make lowz scatterplots
    # ======================
    if MAKE_PLOTS:
        print("Making low-z scatter plots")
        plot_satellite_positions(
            lowz,
            colored_by="gmr",
            colorbar_label="$(g-r)_0$",
            figname="lowz_positions-by-color",
            vmin=0,
            vmax=0.8,
            colormap="RdYlBu_r",
            xlim=(240, 120),
            ylim=(0, 60),
        )

        plot_satellite_positions(
            lowz,
            colored_by="mu_eff",
            colorbar_label=r"$\mu_{r,\rm eff}$",
            figname="lowz_positions-by-surface_brightness",
            vmin=21,
            vmax=26,
            colormap="viridis",
            xlim=(240, 120),
            ylim=(0, 60),
        )

    # remove massive hosts from lowz catalog
    # ======================================
    print("Removing massive hosts and high-z hosts from low-z catalog")
    hosts = Query("mass_GSE >= 9.5").filter(nsa)

    lowz, hosts_x_lowz = remove_hosts_from_lowz(hosts, lowz, savefig=MAKE_PLOTS)

    hosts = Query("z_NSA <= 0.03").filter(hosts)
    hosts.to_parquet(hosts_file)

    # identify (or load) satellites
    # =============================
    print("Loading satellites")
    sats_file = results_dir / "sats-nsa_p0_5.parquet"
    try:
        sats = pd.read_parquet(sats_file)
    except (FileNotFoundError, OSError):
        sats = assign_satellites_to_hosts(
            hosts, lowz, rank_by="mass_GSE", z_min=0.005, savefig=MAKE_PLOTS
        )
        sats.to_parquet(sats_file)

    # count satellites per host
    # =========================
    print("Counting satellites per host")
    hosts_file = results_dir / "hosts-nsa.parquet"
    column_name = "n_sats_in_300kpc"
    try:
        hosts = pd.read_parquet(hosts_file)
        assert column_name in hosts.columns
    except AssertionError:
        hosts = count_satellites_per_host(
            hosts,
            sats,
            column_name=column_name,
            fname=f"{column_name}_per_host",
            savefig=True,
        )
        hosts.to_parquet(hosts_file)

    # count satellites within 150 kpc
    # ===============================
    print("Counting satellites within 150 kpc")
    hosts_file = results_dir / "hosts-nsa.parquet"
    column_name = "n_sats_in_150kpc"
    try:
        hosts = pd.read_parquet(hosts_file)
        assert column_name in hosts.columns
    except AssertionError:
        sats_in_150kpc = Query("sep <= 150").filter(sats)
        hosts = count_satellites_per_host(
            hosts,
            sats_in_150kpc,
            column_name=column_name,
            fname=f"{column_name}_per_host",
            savefig=True,
        )
        hosts.to_parquet(hosts_file)

    # count *bright* satellites per host (M_r < -15)
    # ==============================================
    print("Counting bright satellites")
    hosts_file = results_dir / "hosts-nsa.parquet"
    column_name = "n_bright_sats_in_300kpc"
    try:
        hosts = pd.read_parquet(hosts_file)
        assert column_name in hosts.columns
    except AssertionError:
        # compute absolute magnitude and select those within completeness limit
        sats["M_r"] = sats.r0 - cosmo.distmod(sats.z_NSA).value
        bright_sats = Query("M_r < -15.0").filter(sats)

        hosts = count_satellites_per_host(
            hosts,
            bright_sats,
            column_name=column_name,
            fname=f"{column_name}_per_host",
            savefig=True,
        )
        hosts.to_parquet(hosts_file)

    # correct number of sats
    # ======================
    print("Counting corrected number of satellites")
    hosts_file = results_dir / "hosts-nsa.parquet"
    column_name = "n_corr_sats_in_300kpc"
    try:
        hosts = pd.read_parquet(hosts_file)
        assert column_name in hosts.columns
    except AssertionError:
        hosts = count_corrected_satellites_per_host(
            hosts, sats, column_name=column_name
        )
        hosts.to_parquet(hosts_file)
