"""
John F. Wu (2021)

Script for training the convolutional neural network. Uses the model files found
in `model/xresnet_deconv.py` and `model/deconv.py`.
"""

import numpy as np
import pandas as pd
import torch

from fastai2.basics import *
from fastai2.vision.all import *
from mish_cuda import MishCuda
from model.xresnet_deconv import xresnet34_hybrid
from pathlib import Path
from PIL import ImageFile
from sklearn.model_selection import KFold
from tqdm import tqdm

seed = 42
ImageFile.LOAD_TRUNCATED_IMAGES = True
MIN_JPG_SIZE = 4096

ROOT = Path(__file__).resolve().parent.parent
results_dir = ROOT / "results/xSAGA"

legacy_image_stats = [
    np.array([0.14814416, 0.14217226, 0.13984123]),
    np.array([0.0881476, 0.07823102, 0.07676626]),
]

sz = 144
item_tfms = [CropPad(sz)]
batch_tfms = aug_transforms(
    max_zoom=1.0, flip_vert=True, max_lighting=0.0, max_warp=0.0
) + [Normalize.from_stats(*legacy_image_stats)]

learner_metrics = [accuracy, F1Score(), Recall(), Precision(), RocAuc()]

dtype_dict = {
    "OBJID": str,
    "RA": np.float64,
    "DEC": np.float64,
    "SPEC_Z": np.float64,
    "SPEC_FLAG": np.int32,
    "HAS_SAT_Z": np.int32,
}

# Borrowed from https://github.com/fastai/fastai/blob/master/fastai/losses.py#L48
class FocalLossFlat(CrossEntropyLossFlat):
    """
    Same as CrossEntropyLossFlat but with focal paramter, `gamma`. Focal loss is
    introduced by Lin et al. https://arxiv.org/pdf/1708.02002.pdf. Note the class
    weighting factor in the paper, alpha, can be implemented through pytorch `weight`
    argument in nn.CrossEntropyLoss.
    """

    y_int = True

    @use_kwargs_dict(keep=True, weight=None, ignore_index=-100, reduction="mean")
    def __init__(self, *args, gamma=2, axis=-1, **kwargs):
        self.gamma = gamma
        self.reduce = kwargs.pop("reduction") if "reduction" in kwargs else "mean"
        super().__init__(*args, reduction="none", axis=axis, **kwargs)

    def __call__(self, inp, targ, **kwargs):
        ce_loss = super().__call__(inp, targ, **kwargs)
        pt = torch.exp(-ce_loss)
        fl_loss = (1 - pt) ** self.gamma * ce_loss
        return (
            fl_loss.mean()
            if self.reduce == "mean"
            else fl_loss.sum()
            if self.reduce == "sum"
            else fl_loss
        )


def train(
    df,
    K=4,
    bs=128,
    n_epochs=10,
    max_lr=1e-2,
    img_dir="images-legacy_saga-2021-02-19",
    version="2021-02-19",
):
    """Train the CNN using k-fold cross-validation.
    """

    df = df.sample(frac=1, random_state=seed).copy()
    df["low_z"] = df.SPEC_Z < 0.03

    kf = KFold(K)
    df["kfold"] = -1

    N = len(df)
    df_folds = []

    for k, [_, val_idx] in enumerate(kf.split(range(N))):

        k = k + 1
        df.loc[val_idx, "kfold"] = k
        df["kfold_split"] = df.kfold == k

        # load data
        dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_x=ColReader(["OBJID"], pref=f"{ROOT}/{img_dir}/", suff=".jpg"),
            get_y=ColReader("low_z"),
            splitter=ColSplitter("kfold_split"),
            item_tfms=item_tfms,
            batch_tfms=batch_tfms,
        )

        dls = ImageDataLoaders.from_dblock(dblock, df, path=ROOT, bs=bs)

        loss_func = FocalLossFlat(gamma=2)
        model = xresnet34_hybrid(
            n_out=2, sa=True, act_cls=MishCuda, groups=64, reduction=8
        )

        learn = Learner(dls, model, opt_func=ranger, loss_func=loss_func)

        # train
        learn.fit_one_cycle(n_epochs, max_lr)

        torch.save(
            learn.model.state_dict(),
            ROOT / f"models/saga_FL-hdxresnet34-sz{sz}_{version}_{k}-of-{K}.pth",
        )

        # get cross-validation predictions
        p_low_z, true_low_z = learn.get_preds()

        valid = learn.dls.valid.items.copy()
        valid["pred_low_z"] = p_low_z[:, 1]

        df_folds.append(valid)

    results = pd.concat(df_folds)
    results.drop("kfold_split", axis=1, inplace=True)

    return results


def predict(
    df,
    filenames,
    K=4,
    img_dir="images-legacy_saga-2021-02-19",
    bs=128,
    version="2019-02-19",
):
    """Make predictions using model trained above
    """

    model = xresnet34_hybrid(n_out=2, sa=True, act_cls=MishCuda, groups=64, reduction=8)

    all_preds = []

    # load dls in order to do the same transforms
    df["low_z"] = df.SPEC_Z < 0.03

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=ColReader(["OBJID"], pref=f"{ROOT}/{img_dir}/", suff=".jpg"),
        get_y=ColReader("low_z"),
        splitter=RandomSplitter(0),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
    )

    dls = ImageDataLoaders.from_dblock(dblock, df, path=ROOT, bs=bs)
    test_dl = dls.test_dl(filenames, num_workers=8, bs=bs)

    for k in range(K):
        k = k + 1

        model.load_state_dict(
            torch.load(ROOT / f"models/saga_FL-hdxresnet34-sz{sz}_{version}_{k}-of-4")
        )
        model.to("cuda")

        # evaluate
        print(f"Loaded model {k}")

        m = model.eval()
        outputs = []
        with torch.no_grad():
            for (xb,) in tqdm(iter(test_dl), total=len(test_dl)):
                outputs.append(m(xb).cpu())

        outs = torch.cat(outputs)
        p_CNNs = outs.softmax(1)

        # save
        objIDs = list(x.stem for x in filenames)
        objIDs = np.array(objIDs, dtype=str)

        preds = pd.DataFrame({f"p_CNN_{k}": p_CNNs[:, 1]}, index=objIDs)
        preds.to_parquet(results_dir / f"predictions-{k}.parquet")

        all_preds.append(preds)

    all_preds = pd.concat(all_preds)
    all_preds.drop("kfold_split", axis=1, inplace=True)

    return all_preds


def train_resnet(
    df,
    K=4,
    bs=128,
    n_epochs=10,
    max_lr=1e-2,
    img_dir="images-legacy_saga-2021-02-19",
    version="2021-02-19",
):
    """Train the CNN using k-fold cross-validation.
    """

    df = df.sample(frac=1, random_state=seed).copy()
    df["low_z"] = df.SPEC_Z < 0.03

    kf = KFold(K)
    df["kfold"] = -1

    N = len(df)
    df_folds = []

    for k, [_, val_idx] in enumerate(kf.split(range(N))):

        k = k + 1
        df.loc[val_idx, "kfold"] = k
        df["kfold_split"] = df.kfold == k

        # load data
        dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_x=ColReader(["OBJID"], pref=f"{ROOT}/{img_dir}/", suff=".jpg"),
            get_y=ColReader("low_z"),
            splitter=ColSplitter("kfold_split"),
            item_tfms=item_tfms,
            batch_tfms=batch_tfms,
        )

        dls = ImageDataLoaders.from_dblock(dblock, df, path=ROOT, bs=bs)

        learn = cnn_learner(dls, resnet34, pretrained=False)

        # train
        learn.fit_one_cycle(n_epochs, max_lr)

        torch.save(
            learn.model.state_dict(),
            ROOT / f"models/saga_FL-hdxresnet34-sz{sz}_{version}_{k}-of-{K}.pth",
        )

        # get cross-validation predictions
        p_low_z, true_low_z = learn.get_preds()

        valid = learn.dls.valid.items.copy()
        valid["pred_low_z"] = p_low_z[:, 1]

        df_folds.append(valid)

    results = pd.concat(df_folds)
    results.drop("kfold_split", axis=1, inplace=True)

    return results


if __name__ == "__main__":
    saga = pd.read_csv(ROOT / "data/saga_redshifts_2021-02-19.csv", dtype=dtype_dict)

    results = train(saga, n_epochs=10)
    results.to_csv(
        results_dir / "cnn-training/saga_FL-hdxresnet34-sz144_2021-02-19.csv"
    )

    results_resnet = train_resnet(saga, n_epochs=10)
    results_resnet.to_csv(
        results_dir / "cnn-training/saga_resnet34-sz144_2021-02-19.csv"
    )

    # predict
    filenames = list(
        x
        for x in (ROOT / "images-legacy-dr9").rglob("*.jpg")
        if (x.stat().st_size >= MIN_JPG_SIZE)
    )

    all_preds = predict(saga, filenames)
