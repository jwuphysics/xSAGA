import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cmasher as cmr

from astropy.io import fits
from astropy.coordinates import SkyCoord, match_coordinates_sky
import astropy.units as u
from PIL import Image

import os
from pathlib import Path
import requests
from tqdm.notebook import tqdm

ROOT = Path(__file__).resolve().parent.parent
results_dir = ROOT/'results/xSAGA-init'

def load_satellites(p_sat_thresh=0.5):
    """Returns dataframe of satellite candidates in xSAGA-init.
    """

    df = pd.read_csv(ROOT/'results/predictions-initial.csv')
    return df[df.p_sat > p_sat_thresh].copy()


def load_NSA():
    """Returns dataframe with some NSA info.
    """
    nsa = fits.getdata(ROOT/'data/nsa_v1_0_1.fits')

    # do this column by column, *not* on entire catalog
    byteorder = lambda x: x.byteswap().newbyteorder()

    # doesn't use .byteswap().newbyteorder() because that's super slow...
    return pd.DataFrame({
        'NSAID': byteorder(nsa.NSAID),
        'z_NSA': byteorder(nsa.Z),
        'ra_NSA': byteorder(nsa.RA),
        'dec_NSA': byteorder(nsa.DEC),
        'M_r_NSA': byteorder(nsa.ELPETRO_ABSMAG[:, 4])
    })

def load_satellites_x_NSA(p_sat_thresh=0.5, z_cut=0.03):
    """Returns the satellite catalog crossmatched to NSA.

    Imposes a cut on NSA spectroscopic redshift in order to
    select low-z galaxies. This value is `z_cut` and should be
    set to 9999 if the cut is not desired.
    """
    sat = load_satellites(p_sat_thresh=p_sat_thresh)
    sat_coords = SkyCoord(sat.ra, sat.dec, unit='deg')

    nsa = load_NSA()
    nsa = nsa[nsa.z_NSA < z_cut]
    nsa_coords = SkyCoord(nsa.ra_NSA, nsa.dec_NSA, unit='deg')

    # crossmatch candidates with confirmed low-z NSA galaxies
    nsa_idx, sep, _ = sat_coords.match_to_catalog_sky(nsa_coords)
    sat['sep_NSA'] = sep.to('arcsec').value

    # should be row-aligned, but pandas needs the index to also 
    # be aligned (this is dumb) before we can concatenate them
    nsa = nsa.iloc[nsa_idx].copy()
    nsa.index = sat.index
    return pd.concat([sat, nsa], axis=1)

def get_satellites_around_hosts(sat, nsa, M_r_range=(-21, -20), z_cut=0.03):
    """Load in satellites (from `sat`) around hosts (from `nsa`). 
    
    Hosts are selected using the cuts in `M_r_range` (2-tuple or list)
    and `z_cut` (the upper redshift limit). 

    Satellites around hosts can be easily selected using a cut like
        ```
        all_sats = all_sats[all_sats.host_sep < 3600].copy()
        ```
    where 3600 [arcsec] is a typical maximum separation. Or satellites
    can be grouped among hosts like 
        ```
        grouped = all_sats.groupby('host_nsaid')
        host_objID, sats_in_group = next(iter(grouped))
        ```
    """

    # isolate those that satisfy a magnitude and redshift cut (e.g., MW-like hosts)
    cut = (nsa.M_r_NSA > min(M_r_range)) & (nsa.M_r_NSA < max(M_r_range)) & (nsa.z_NSA < z_cut)
    nsa = nsa[cut].copy()

    sat_coords = SkyCoord(sat.ra, sat.dec, unit='deg')
    nsa_coords = SkyCoord(nsa.ra_NSA, nsa.dec_NSA, unit='deg')
    sat_idx, sep, _ = nsa_coords.match_to_catalog_sky(sat_coords)

    nsa = nsa[sep < 1*u.deg].copy()

    # get all hosts
    hosts = nsa.set_index('NSAID').copy()
    host_coords = SkyCoord(hosts.ra_NSA, hosts.dec_NSA, unit='deg')

    host_idx, host_sep, _ = sat_coords.match_to_catalog_sky(host_coords)
    host_sep = host_sep.to('arcsec').value

    # get all satellite candidates matched to hosts
    all_sats = load_satellites_x_NSA()

    all_sats['host_nsaid'] = hosts.iloc[host_idx].index.values
    all_sats['host_sep'] = host_sep

    return all_sats

def remove_duplicates(sats_in_host, duplicate_sep=1*u.arcmin):
    """Given dataframe `sats_in_host`, removes all sources that are
    within `duplicate_sep` from each other, except the brightest one.
    """
    
    coords = SkyCoord(sats_in_host[['ra', 'dec']].values, unit='deg')
    idx_i, idx_j, sep, _ = coords.search_around_sky(coords, duplicate_sep)

    fainter_ids = []
    for i, j, s in zip(idx_i, idx_j, sep):
        if i != j:
            fainter_ids.append(
                np.where(
                    sats_in_host.r0.iloc[i] > sats_in_host.r0.iloc[j], 
                    sats_in_host.objID.iloc[i],
                    sats_in_host.objID.iloc[j]
                )
            )

    fainter_ids = np.unique(fainter_ids)
    return sats_in_host.set_index('objID').drop(fainter_ids).copy()

if __name__ == '__main__':
    

    # sat = load_satellites()
    nsa = load_NSA()
    hosts = nsa.set_index('NSAID').copy()

    # save the satellites x NSA (z < 0.03) crossmatched catalog 
    # sat_x_nsa = load_satellites_x_NSA()
    # sat_x_nsa.to_csv(results_dir/'sat_x_NSA.csv', index=False)

    # sat_x_nsa = pd.read_csv(results_dir/'sat_x_NSA.csv')

    # all_sats = get_satellites_around_hosts(sat, nsa, M_r_range=(-22, -19))
    # all_sats.to_csv(results_dir/'all_sats.csv', index=False)

    all_sats = pd.read_csv(results_dir/'all_sats.csv')
    grouped = all_sats.groupby('host_nsaid')


