import pandas as pd

df = pd.read_csv('~/xSAGA/data/saga_redshifts_2021-02-11.csv')

with open('/home/jupyter/xSAGA/data/saga-2021-02-11_legacy_urls_dr9.txt', 'a') as f:
    for row in df.itertuples():
        objID = row.OBJID
        ra, dec = row.RA, row.DEC

        f.write(f'{objID}.jpg http://legacysurvey.org/viewer/cutout.jpg?ra={ra}&dec={dec}&pixscale=0.262&layer=ls-dr9&size=224\n')
