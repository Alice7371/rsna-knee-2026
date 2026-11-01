"""Train a slice-level classifier on LLM-derived silver labels.

5-fold over studies; targets are soft silver labels; validation reports
study-level AUC against silver (in-domain) and against the 58 gold studies.
"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from common import (LABELS, seed_everything, SILVER_STEVEN, TRAIN_CSV,
                    TRAIN_SERIES_CSV, SILVER_PILKWANG)
from dataset import (build_study_index, build_volume_index, SliceDataset,
                     VolumeDataset, collate_series)
from model import SliceClassifier, AttnSeriesClassifier, unfreeze_last_blocks, aggregate_series


def build_conf_weights():
    """Per-study loss weight from pilkwang verdict confidence.

    Reports that name no finding (all UNK) pull that study's labels weakly;
    explicit YES/NO verdicts pull at full confidence.
    """
    pil = pd.read_csv(SILVER_PILKWANG).set_index("StudyInstanceUID")
    conf_cols = [c for c in pil.columns if c.endswith("__conf")]
    verd_cols = [c for c in pil.columns if c.endswith("__verdict")]
    known = pd.DataFrame(
        {l: (pil[l + "__verdict"] != "UNK") for l in LABELS})
    conf = pil[conf_cols].rename(columns=lambda c: c[:-6])
    conf = conf.where(known)
    w = 0.4 + 0.6 * conf.mean(axis=1).fillna(0.0)
    return w.rename("weight")


def study_auc(pred_df: pd.DataFrame, gold_df: pd.DataFrame, tag: str):
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
    print(f"[{tag}] per-label AUC:",
          {l: (round(a, 3) if a == a else None) for l, a in zip(LABELS, aucs)})
    print(f"[{tag}] MEAN AUC = {np.nanmean(aucs):.4f}")
    return np.nanmean(aucs)


@torch.no_grad()
def evaluate(model, loader, device, silver_df, gold_df, fold, mode="slice"):
    model.eval()
    rows = []
    for batch in loader:
        imgs = batch["imgs"].to(device, non_blocking=True)
        with torch.autocast("cuda"):
            if mode == "attn":
                feats = model.encode(imgs).float()
                logits = model.forward_groups(feats, batch["counts"])
            else:
                logits = model(imgs).float()
                logits = torch.stack([
                    v for v in aggregate_series(logits, batch["counts"])])
        for study, sp in zip(batch["series_studies"],
                             logits.sigmoid().cpu().numpy()):
            rows.append((study, *sp.tolist()))
    pred = pd.DataFrame(rows, columns=["study"] + LABELS).set_index("study")
    pred = pred.groupby(level=0).mean()
    silver_auc = study_auc(pred, silver_df.loc[silver_df.index.isin(pred.index)], f"fold{fold} silver")
    gold_auc = study_auc(pred, gold_df, f"fold{fold} gold")
    model.train()
    return pred


def run_fold(ds_cls, tr_idx, va_idx, series_df, targets, weights, gold_df, args, fold):
    device = "cuda"
    tr_ds = ds_cls(series_df.iloc[tr_idx], targets, size=args.size,
                   slices_per_series=args.sps, train=True)
    va_ds = ds_cls(series_df.iloc[va_idx], targets, size=args.size,
                   slices_per_series=args.sps, train=False)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_studies, shuffle=True,
                       num_workers=args.workers, collate_fn=collate_series,
                       pin_memory=True, drop_last=True, persistent_workers=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_studies, shuffle=False,
                       num_workers=args.workers, collate_fn=collate_series,
                       pin_memory=True, persistent_workers=True)

    mode = args.model
    if mode == "attn":
        model = AttnSeriesClassifier(args.backbone, unfreeze_last=args.unfreeze_last).to(device)
    else:
        model = SliceClassifier(args.backbone).to(device)
        model.encoder.set_grad_checkpointing(enable=bool(args.grad_ckpt))

    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("encoder.") and p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": args.lr},
         {"params": head_params, "lr": args.lr * 10}],
        weight_decay=1e-4)
    steps = (len(tr_dl) * args.epochs) // args.accum
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr, args.lr * 10], total_steps=max(steps, 1), pct_start=0.1)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    scaler = torch.amp.GradScaler()
    w_map = weights  # Series indexed by study

    model.train()
    for ep in range(args.epochs):
        for it, batch in enumerate(tr_dl):
            imgs = batch["imgs"].to(device, non_blocking=True)
            if mode == "attn":
                tgt = batch["series_targets"].to(device, non_blocking=True)
                w = torch.tensor(
                    w_map.reindex(batch["series_studies"]).to_numpy(np.float32),
                    device=device)
            else:
                tgt = batch["target"].to(device, non_blocking=True)
                w = None
            with torch.autocast("cuda"):
                if mode == "attn":
                    feats = model.encode(imgs).float()
                    logits = model.forward_groups(feats, batch["counts"])
                else:
                    logits = model(imgs).float()
                loss = lossf(logits, tgt)
                if w is not None:
                    loss = loss * w[:, None]
                loss = loss.mean() / args.accum
            scaler.scale(loss).backward()
            if (it + 1) % args.accum == 0:
                if mode == "attn":
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
        print(f"fold{fold} epoch{ep} done", flush=True)

    pred = evaluate(model, va_dl, device, targets, gold_df, fold, mode)
    torch.save({"state_dict": model.state_dict(), "fold": fold,
                "backbone": args.backbone, "mode": mode},
               args.out.format(fold))
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["slice", "attn"], default="slice")
    ap.add_argument("--backbone", default="tf_efficientnet_b0.ns_jft_in1k")
    ap.add_argument("--unfreeze-last", type=int, default=6)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--sps", type=int, default=8, help="slices per series")
    ap.add_argument("--batch-studies", type=int, default=6)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--grad-ckpt", type=int, default=0)
    ap.add_argument("--data", choices=["volumes", "jpeg"], default="volumes")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--limit-studies", type=int, default=0,
                    help="debug: use only the first N studies that exist on disk")
    ap.add_argument("--out", default="artifacts/b0_224_fold{}.pt")
    args = ap.parse_args()

    seed_everything(42)
    from common import JPEG_DIR, VOLUMES_DIR
    train = pd.read_csv(TRAIN_CSV).set_index("StudyInstanceUID")
    series_csv = pd.read_csv(TRAIN_SERIES_CSV)
    silver = pd.read_csv(SILVER_STEVEN).set_index("StudyInstanceUID")
    weights = build_conf_weights()
    gold = train.dropna(subset=LABELS, how="all")

    if args.data == "volumes":
        series_df = build_volume_index(series_csv, VOLUMES_DIR)
    else:
        series_df = build_study_index(series_csv, JPEG_DIR, "train_series")
    series_df = series_df[series_df.study.isin(silver.index)].reset_index(drop=True)
    if args.limit_studies:
        keep = series_df.study.drop_duplicates().head(args.limit_studies)
        series_df = series_df[series_df.study.isin(set(keep))].reset_index(drop=True)
    print(f"series index: {len(series_df)} series / {series_df.study.nunique()} studies")

    folds = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    studies = series_df.study.unique()
    study_pos = {s: i for i, s in enumerate(studies)}
    series_groups = series_df.study.map(study_pos)

    oof_rows = []
    for fold, (tr_g, va_g) in enumerate(folds.split(studies)):
        tr_idx = np.where(series_groups.isin(tr_g))[0]
        va_idx = np.where(series_groups.isin(va_g))[0]
        ds_cls = VolumeDataset if args.data == "volumes" else SliceDataset
        pred = run_fold(ds_cls, tr_idx, va_idx, series_df, silver, weights,
                        gold, args, fold)
        oof_rows.append(pred)
    oof = pd.concat(oof_rows)
    oof.to_csv("artifacts/oof.csv")
    study_auc(oof, gold, "OOF gold")


if __name__ == "__main__":
    main()
