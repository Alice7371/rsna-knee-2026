"""Study-level dataset over the preprocessed JPEG mirrors of the competition data.

Directory layout (identical to the raw competition):
    <root>/train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.jpeg
Series metadata (plane, fluid-sensitive, fat-suppression) comes from train_series.csv.
Slice files are SOP UIDs and are NOT anatomically ordered; slice-mean pooling is
order-free, so for v1 we sample evenly by file position.
"""
from pathlib import Path
import random

import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset

from common import LABELS


def build_study_index(series_csv: pd.DataFrame, root: Path, split_dir: str):
    """Return DataFrame: one row per series with study, plane, flags, file list."""
    rows = []
    base = root / split_dir
    for r in series_csv.itertuples(index=False):
        sdir = base / r.StudyInstanceUID / r.SeriesInstanceUID
        if not sdir.exists():
            continue
        files = sorted(sdir.glob("*.jpeg")) or sorted(sdir.glob("*.jpg"))
        if not files:
            continue
        rows.append({
            "study": r.StudyInstanceUID,
            "series": r.SeriesInstanceUID,
            "plane": getattr(r, "Anatomical_Plane", "unknown"),
            "fluid": int(getattr(r, "Fluid_Sensitive", 0)),
            "fs": int(getattr(r, "Fat_Suppression", 0)),
            "n_slices": len(files),
            "files": [str(f) for f in files],
        })
    return pd.DataFrame(rows)


def build_volume_index(series_csv: pd.DataFrame, volumes_dir: Path):
    """Index npz volumes (one per series, key='data', (N,224,224) uint8)."""
    rows = []
    for r in series_csv.itertuples(index=False):
        p = volumes_dir / f"{r.SeriesInstanceUID}.npz"
        if p.exists():
            rows.append({
                "study": r.StudyInstanceUID,
                "series": r.SeriesInstanceUID,
                "plane": getattr(r, "Anatomical_Plane", "unknown"),
                "fluid": int(getattr(r, "Fluid_Sensitive", 0)),
                "fs": int(getattr(r, "Fat_Suppression", 0)),
                "path": str(p),
            })
    return pd.DataFrame(rows)


class VolumeDataset(Dataset):
    """Series-level samples from preprocessed npz volumes."""

    def __init__(self, series_df: pd.DataFrame, targets: pd.DataFrame,
                 size=224, slices_per_series=8, train=True):
        self.series_df = series_df.reset_index(drop=True)
        self.targets = targets
        self.size = size
        self.sps = slices_per_series
        self.train = train

    def __len__(self):
        return len(self.series_df)

    def __getitem__(self, i):
        row = self.series_df.iloc[i]
        vol = np.load(row.path)["data"]  # (N, H, W) uint8, min-max normalized
        n_total = vol.shape[0]
        if self.train and n_total > self.sps:
            idx = sorted(random.sample(range(n_total), self.sps))
        elif n_total > self.sps:
            idx = np.linspace(0, n_total - 1, self.sps).round().astype(int).tolist()
        else:
            idx = list(range(n_total))
        imgs = vol[idx]
        if imgs.shape[1] != self.size:
            imgs = np.stack([cv2.resize(s, (self.size, self.size),
                                        interpolation=cv2.INTER_AREA) for s in imgs])
        t = self.targets.loc[row.study, LABELS].to_numpy(np.float32)
        return {
            "imgs": torch.from_numpy(np.ascontiguousarray(imgs)),
            "target": torch.from_numpy(t),
            "n": len(idx),
            "study": row.study,
            "plane": row.plane,
        }


def load_image(path: str, size: int):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"bad image {path}")
    # the alenic JPEGs are already min-max stretched 224x224; use as-is so that
    # train (JPEG) and test (DICOM -> resize -> minmax) share one transform
    if img.shape[0] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


class SliceDataset(Dataset):
    """Yields individual slices with their study's silver target (slice-level BCE)."""

    def __init__(self, series_df: pd.DataFrame, targets: pd.DataFrame, size=224,
                 slices_per_series=8, train=True):
        self.series_df = series_df.reset_index(drop=True)
        self.targets = targets
        self.size = size
        self.sps = slices_per_series
        self.train = train

    def __len__(self):
        return len(self.series_df)

    def __getitem__(self, i):
        row = self.series_df.iloc[i]
        files = row.files
        if self.train and len(files) > self.sps:
            files = random.sample(files, self.sps)
        elif len(files) > self.sps:
            step = len(files) / self.sps
            files = [files[int(j * step)] for j in range(self.sps)]
        imgs = np.stack([load_image(f, self.size) for f in files])  # (S, H, W)
        t = self.targets.loc[row.study, LABELS].to_numpy(np.float32)
        return {
            "imgs": torch.from_numpy(imgs),           # (S, H, W)
            "target": torch.from_numpy(t),            # (12,)
            "n": len(files),
            "study": row.study,
            "plane": row.plane,
        }


def collate_series(batch):
    """Flatten slices of all series in the batch; keep slice->series mapping."""
    imgs, targets, counts, planes, studies, series_targets, series_studies = \
        [], [], [], [], [], [], []
    for b in batch:
        n = b["n"]
        im = b["imgs"][:n]
        # 3-channel by repeating adjacent slice (prev, cur, next) for slight context
        if n == 1:
            im3 = torch.stack([im[0]] * 3)
        else:
            prev = torch.cat([im[:1], im[:-1]])
            nxt = torch.cat([im[1:], im[-1:]])
            im3 = torch.stack([prev, im, nxt], dim=1)  # (n, 3, H, W)
        imgs.append(im3)
        targets.append(b["target"].expand(n, -1))
        counts.append(n)
        planes.extend([b["plane"]] * n)
        studies.extend([b["study"]] * n)
        series_targets.append(b["target"])
        series_studies.append(b["study"])
    x = torch.cat(imgs).float().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    return {
        "imgs": x,                              # (sum_n, 3, H, W) float normalized
        "target": torch.cat(targets),           # (sum_n, 12)
        "series_targets": torch.stack(series_targets),  # (B, 12)
        "series_studies": series_studies,
        "counts": counts,
        "planes": planes,
        "studies": studies,
    }
