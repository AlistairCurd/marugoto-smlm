# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from fastai.learner import load_learner
import h5py
from PIL import Image
from marugoto import mil
from marugoto.mil.data import get_target_enc

__all__ = ["plot_top_att_tiles_", "plot_top_att_from_df_"]


def _top_att_tiles_df(
    feature_dir: Path,
    tile_dir: Path,
    model_path: Path,
    patient_preds_csv: Path,
    slide_csv: Path,
    target_label: str,
    pos_class: str,
    n_patients=5,
    n_tiles: int = 5,
    out_dir: Path = None,
) -> pd.DataFrame:
    feature_dir = Path(feature_dir)
    tile_dir = Path(tile_dir)
    if not out_dir:
        out_dir = Path(patient_preds_csv).parent
    learn = load_learner(model_path)

    # glob does not work with lists anymore
    # if ~isinstance(feature_dir, list):
    #    feature_dir = [feature_dir]

    # Get cohort dataframe
    df = mil.data.get_cohort_df(
        clini_table=patient_preds_csv,
        slide_table=slide_csv,
        feature_dir=feature_dir,
        target_label=target_label,
        # categories=[pos_class],
        )[0]  # Returns df and categories
    # Restrict to chosen class - might not be neceaary - see line 62
    df = df[df[target_label] == pos_class]
    # Make loss column is float type
    df['loss'] = df['loss'].astype(float)

    encoder = learn.encoder.eval()
    attention = learn.attention.eval()
    head = learn.head.eval()
    # get the index of the positive class' column
    target_enc = get_target_enc(learn)
    pos_idx = target_enc.transform([[pos_class]]).argmax()

    # fig, axs = plt.subplots(n_patients, n_tiles, figsize=(10, 10), dpi=300)
    top_ts = []
    # i_patients = np.repeat([np.arange(n_tiles)],n_patients,axis=0)
    i_patients = np.arange(n_patients)
    iter_df = df[df[target_label] == pos_class].copy()

    # Use top n_patients by loss value
    for i_patient, slide_paths in zip(
        i_patients, iter_df.nsmallest(n_patients, "loss")["slide_path"].values
    ):
        # aggregate all the slides' features
        feats, coords, sizes = [], [], []
        for slide_path in slide_paths:
            with h5py.File(slide_path, "r") as f:
                feats.append(torch.from_numpy(f["feats"][:]).float())
                sizes.append(len(f["feats"]))
                coords.append(torch.from_numpy(f["coords"][:]))
        feats, coords = torch.cat(feats), torch.cat(coords)

        # get the attention, class score for each tile
        encs = encoder(feats).squeeze()
        patient_atts = torch.softmax(attention(encs).squeeze(), 0).detach()
        patient_atts *= len(patient_atts)
        patient_scores = torch.softmax(head(encs), 1)[:, pos_idx].detach().squeeze()
        n_classes = patient_scores.shape[-1] * 1.0
        # scores scaled by attention, centered around 0
        # assume "neutral" score is 1/n_classes
        patient_weighted_scores = patient_atts * (patient_scores - 1.0 / n_classes)
        # sample from upper 10% of patient's tiles
        n = len(patient_weighted_scores)

        # sample randomly in future ?!
        # Take top 20 % of tiles, then take every 0.1n/ntiles-th  tile
        # top_idxs = patient_weighted_scores.topk(n // 5).indices[:: n // 10 // n_tiles][
        #     :n_tiles
        # ]

        # Try just using all of top 4 tiles:
        top_idxs = patient_weighted_scores.topk(4).indices
        top_slides = np.repeat(slide_path, sizes)[top_idxs]
        top_scores = patient_scores[top_idxs]
        top_atts = patient_atts[top_idxs]

        # breakpoint()
        for tile_nr, (slide, (y, x), score, att) in enumerate(
            zip(top_slides, coords[top_idxs], top_scores, top_atts)
        ):
            # This is wrong for my tiling
            # f_name = tile_dir / slide.stem / f"{slide.stem}_({y},{x}).jpg"

            # So use this:
            f_name = tile_dir / slide.stem / f"tile_({y},{x}).jpg"
            top_ts.append([i_patient, tile_nr, f_name, att.item(), score.item()])

    top_t_results = pd.DataFrame(
        top_ts, columns=["PATIENT_NR", "TILES_NR", "file_name", "attention", "score"]
    )

    return top_t_results


def plot_top_att_from_df_(df: pd.DataFrame, out_file: Path):
    # n_rows = df.shape[0]
    n_tiles = np.max(df["TILES_NR"]) + 1
    # n_patients = n_rows // n_tiles
    n_patients = np.max(df["PATIENT_NR"]) + 1
    fig, axs = plt.subplots(n_patients, n_tiles, figsize=(10, 10), dpi=300)
    for ix, row in df.iterrows():
        att = row["attention"]
        score = row["score"]
        fname = row["file_name"]
        # ax = axs.reshape(-1)[ix]
        ax = axs[row["PATIENT_NR"], row["TILES_NR"]]

        # Scale tile to make data visible!
        raw_tile = Image.open(fname)
        raw_tile = np.array(raw_tile)
        norm_tile = (raw_tile * 255. / raw_tile.max()).astype(np.uint8)
        norm_tile = Image.fromarray(norm_tile, 'RGB')

        ax.imshow(norm_tile)
        ax.set_title(f"a={att:0.2f} s={score:0.2f}")
        ax.axis("off")
    print(f"Writing top tiles images to: {out_file}")
    fig.savefig(out_file)


def plot_top_att_tiles_(
    feature_dir: Path,
    tile_dir: Path,
    model_path: Path,
    patient_preds_csv: Path,
    slide_csv: Path,
    target_label: str,
    pos_class: str,
    n_patients=5,
    n_tiles: int = 5,
    out_dir: Path = None,
) -> None:
    if not out_dir:
        out_dir = Path(patient_preds_csv).parent

    df = _top_att_tiles_df(
        feature_dir,
        tile_dir,
        model_path,
        patient_preds_csv,
        slide_csv,
        target_label,
        pos_class,
        n_patients=n_patients,
        n_tiles=n_tiles,
        out_dir=out_dir,
    )
    if out_dir.exists():
        pass
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{target_label}_{pos_class}top_att_tiles.csv"
    out_file = Path(str(out_file).replace(" ", ""))
    print(f"Writing top tiles csv to: {out_file}")
    df.to_csv(out_file)
    suffix = [".jpg", ".svg"]
    for s in suffix:
        out_img = out_file.with_suffix(s)
        plot_top_att_from_df_(df, out_img)

    # Save individual tiles
    for ix, row in df.iterrows():
        in_path = Path(row["file_name"])
        tile = Image.open(in_path)
        out_path = out_dir / 'raw_tiles' / \
            (in_path.parent.stem + '_' + in_path.stem
             + 'att{:.2f}_score{:.4f}.jpeg'
             ).format(row["attention"], row["score"])
        if out_path.parent.exists():
            pass
        else:
            out_path.parent.mkdir()
        tile.save(out_path)

        tile = np.array(tile)
        norm_tile = (tile * 255. / tile.max()).astype(np.uint8)
        norm_tile = Image.fromarray(norm_tile, 'RGB')
        out_path = out_dir / 'norm_tiles' / \
            (out_path.stem + '_norm.jpg')
        if out_path.parent.exists():
            pass
        else:
            out_path.parent.mkdir()
        norm_tile.save(out_path)


if __name__ == "__main__":
    from fire import Fire

    Fire(plot_top_att_tiles_)
