"""Compare inference-time configurations for the trained v1.1 fold ensemble.

Variants: slices per series (12 evenly / all 24) x pooling (top4-mean / top1-max
for focal findings). Reports gold-58 AUC per variant; writes the best config's
per-study predictions for sanity inspection.
"""
import argparse
import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from common import LABELS, seed_everything, SILVER_STEVEN, TRAIN_CSV, TRAIN_SERIES_CSV, VOLUMES_DIR
from dataset import build_volume_index, VolumeDataset, collate_series
from model import AttnSeriesClassifier

FOCAL = {"Fracture", "Contusion", "Medial Meniscus", "Lateral Meniscus"}


def study_auc(pred_df, gold_df, tag):
    common = gold_df.index.intersection(pred_df.index)
    aucs = []
    for l in LABELS:
        g = gold_df.loc[common, l]
        m = g.notna()
        y = (g[m] > 0.5).astype(int)
        if m.sum() == 0 or y.nunique() < 2:
            aucs.append(np.nan)
            continue
        aucs.append(roc_auc_score(y, pred_df.loc[common[m], l]))
    print(f"[{tag}] mean AUC = {np.nanmean(aucs):.4f}  "
          + " ".join(f"{l[:6]}={a:.3f}" for l, a in zip(LABELS, aucs) if a == a))
    return np.nanmean(aucs)


@torch.no_grad()
def run(models, loader, device, focal_max=False):
    model_dims = len(models)
    rows = []
    for batch in loader:
        imgs = batch["imgs"].to(device, non_blocking=True)
        with torch.autocast("cuda"):
            feats = models[0].encode(imgs).float()
            logits = models[0].forward_groups(feats, batch["counts"])
            for m in models[1:]:
                f = m.encode(imgs).float()
                logits = logits + m.forward_groups(f, batch["counts"])
        logits = (logits / model_dims).sigmoid().cpu().numpy()
        if focal_max:
            # series-level: focal labels read optimistically; focal max lives in
            # the study aggregation below, so keep series probs as-is here
            pass
        for study, sp in zip(batch["series_studies"], logits):
            rows.append((study, *sp.tolist()))
    pred = pd.DataFrame(rows, columns=["study"] + LABELS).set_index("study")
    return pred


def aggregate_study(pred, focal_max):
    if not focal_max:
        return pred.groupby(level=0).mean()
    g = pred.groupby(level=0)
    mean_p = g.mean()
    max_p = g.max()
    out = mean_p.copy()
    for l in FOCAL & set(mean_p.columns):
        # optimistic blend: 0.5*mean + 0.5*max, AUC only cares about order
        out[l] = 0.5 * mean_p[l] + 0.5 * max_p[l]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="artifacts/v11_dino_fold*.pt")
    ap.add_argument("--out", default="artifacts/ens_oof.csv")
    args = ap.parse_args()

    seed_everything(42)
    device = "cuda"
    ckpts = sorted(glob.glob(args.pattern))
    models = []
    for c in ckpts:
        ck = torch.load(c, map_location="cpu", weights_only=False)
        m = AttnSeriesClassifier(ck["backbone"], pretrained=False)
        m.load_state_dict(ck["state_dict"])
        m.to(device).eval()
        models.append(m)
    print(f"loaded {len(models)} folds")

    train = pd.read_csv(TRAIN_CSV).set_index("StudyInstanceUID")
    gold = train.dropna(subset=LABELS, how="all")
    silver = pd.read_csv(SILVER_STEVEN).set_index("StudyInstanceUID")
    series_df = build_volume_index(pd.read_csv(TRAIN_SERIES_CSV), VOLUMES_DIR)
    series_df = series_df[series_df.study.isin(silver.index)].reset_index(drop=True)

    for sps, name in [(24, "all24"), (12, "even12")]:
        ds = VolumeDataset(series_df, silver, size=224, slices_per_series=sps, train=False)
        dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=8,
                        collate_fn=collate_series, pin_memory=True)
        pred = run(models, dl, device)
        for focal in [False, True]:
            agg = aggregate_study(pred, focal)
            tag = f"{name} focal_max={focal}"
            study_auc(agg, gold, tag)
        pred.to_csv(args.out.replace(".csv", f"_{name}.csv"))
        del ds, dl


if __name__ == "__main__":
    main()
