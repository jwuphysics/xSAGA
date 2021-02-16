import numpy as np
import pandas as pd

from fastai2.basics import *
from fastai2.vision.all import *

from pathlib import Path

__all__ = ['load_saga', 'oversample', 'get_saga_dls', 'resample_dls', 'legacy_image_stats', 'item_tfms', 'batch_tfms']

PATH = Path('..').resolve()

legacy_image_stats = [np.array([0.14814416, 0.14217226, 0.13984123]), np.array([0.0881476 , 0.07823102, 0.07676626])]

item_tfms = [Resize(224)]
batch_tfms = (
    aug_transforms(max_zoom=1., flip_vert=True, max_lighting=0., max_warp=0.) + 
    [Normalize.from_stats(*legacy_image_stats)]
)
seed = 256

def load_saga(cleaned=True, PATH=PATH,):
    """Returns the SAGA redshift catalog as a `pd.DataFrame`.
    
    If `cleaned` is True, then it will only return rows that do not 
    have corrupted Legacy imaging.
    """
    dtype_dict = {
        'OBJID': str, # note that fastai doesn't handle int64 well
        'RA': np.float64, 
        'DEC': np.float64,
        'SPEC_Z': np.float64,
        'SPEC_FLAG': np.int32,
        'HAS_SAT_Z': np.int32,
    }
    
    if cleaned: 
        saga_filepath = f'{PATH}/data/SAGA-s2-redshifts_2020-08-26.csv'
    else:
        saga_filepath = f'{PATH}/data/SAGA-s2-redshifts_2020-08-04.csv'
        
    return pd.read_csv(saga_filepath, dtype=dtype_dict)

def oversample(df: pd.DataFrame, label_column='HAS_SAT_Z'):
    """Oversample a `pd.DataFrame` such that `label_column` classes
    are balanced. Note that the result is not shuffled.
    
    See https://gist.github.com/scart97/8c33b84db8d6375739b57afab1355900
    """
    max_size = df[label_column].value_counts().max()
    lst = [df]
    for class_index, group in df.groupby(label_column):
        lst.append(group.sample(max_size-len(group), replace=True))
    return pd.concat(lst)


def get_saga_dls(
    saga, 
    label_column='HAS_SAT_Z',
    split_column=None,
    oversample_satellites=True, 
    undersample_nonsatellites=None, 
    valid_pct=0.25, 
    bs=64, 
    PATH=PATH,
    img_dir='images-legacy_saga-2021-02-11',
    item_tfms=item_tfms, 
    batch_tfms=batch_tfms, 
    seed=256,
):
    """Returns Dataloaders `dls` based on an input catalog `saga`. 
    If `undersample_nonsatellites` is an integer, then it will sample from the 
    nonsatellites class. User can also provide fraction of sample for validation
    by supplying `valid_pct`.
    """
    
    not_satellite = (saga.SPEC_FLAG == 1) & (saga.SPEC_Z > 0.03)
    is_satellite = (saga.SPEC_FLAG == 1) & (saga[label_column] == 1) # 1 == True

    if undersample_nonsatellites is None:
        df = saga[not_satellite | is_satellite]
    elif isinstance(undersample_nonsatellites, int):
        df = pd.concat(
            (saga[not_satellite].sample(undersample_nonsatellites), saga[is_satellite]),
        )
    else:
        raise TypeError('Please enter an integer for `undersample_nonsatellites`')
    
    if split_column is None:
        dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_x=ColReader(['OBJID'], pref=f'{PATH}/{img_dir}/', suff='.jpg'),
            get_y=ColReader(label_column),
            splitter=RandomSplitter(valid_pct=valid_pct, seed=seed),
            item_tfms=item_tfms,
            batch_tfms=batch_tfms,
        )
    elif split_column in df.columns:
        dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_x=ColReader(['OBJID'], pref=f'{PATH}/{img_dir}/', suff='.jpg'),
            get_y=ColReader(label_column),
            splitter=ColSplitter(split_column),
            item_tfms=item_tfms,
            batch_tfms=batch_tfms,
        )
    else:
        raise TypeError('Please enter a valid column for splitting train/valid subsets')
    
    dls = ImageDataLoaders.from_dblock(dblock, df, path=PATH, bs=bs)
    
    if oversample_satellites:
        return resample_dls(
            dls, label_column=label_column, split_column=split_column,
            bs=bs, PATH=PATH, item_tfms=item_tfms, batch_tfms=batch_tfms, seed=seed
        )
    else:
        return dls
    

def resample_dls(
    dls, 
    label_column='HAS_SAT_Z', 
    split_column=None,
    bs=64, PATH=PATH, 
    img_dir='images-legacy_saga-2021-02-11',
    item_tfms=item_tfms, 
    batch_tfms=batch_tfms, 
    seed=seed,
):
    """Given Dataloaders `dls` -- you may need to run `get_saga_dls()` first --
    oversample the satellites (generally outnumbered 100:1) in order to
    balance the classes. The classes are split by a binary `label_column`,
    which defaults to 'HAS_SAT_Z'. 
    
    Note that this method is probably not memory efficient.
    """
    train = dls.train.items.copy()
    valid = dls.valid.items.copy()

    train_oversampled = oversample(train, label_column=label_column)

    if split_column is None:
        split_column = 'is_valid'
        train_oversampled[split_column] = False
        valid[split_column] = True
        
    df_oversampled = pd.concat((train_oversampled, valid))
    
    # create resampled datablock
    dblock_oversampled = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=ColReader(['OBJID'], pref=f'{PATH}/{img_dir}/', suff='.jpg'),
        get_y=ColReader(label_column),
        splitter=ColSplitter(split_column),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
    )

    return ImageDataLoaders.from_dblock(dblock_oversampled, df_oversampled, path=PATH, bs=bs)