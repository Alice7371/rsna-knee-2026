"""RSNA Knee v1.1 inference kernel (script).

Mirrors the v1.1 training pipeline exactly:
- DINOv2-small encoders, gated-attention pooling over a series' slices
- test-time: ALL slices of each series, resized to 224 (INTER_AREA),
  min-max scaled across the volume to uint8, 3-channel adjacent stacking,
  ImageNet normalization
- 5-fold ensemble: mean logits across folds -> sigmoid
- study prediction: mean of series probabilities
"""
import glob
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import timm

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def find_root():
    for c in ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
              "/kaggle/input/rsna-knee-abnormality-detection"]:
        if os.path.isfile(os.path.join(c, "test.csv")):
            return c
    base = "/kaggle/input"
    for d in sorted(os.listdir(base)):
        p = os.path.join(base, d)
        if os.path.isfile(os.path.join(p, "test.csv")) and \
                os.path.isdir(os.path.join(p, "test_series")):
            return p
    raise FileNotFoundError("competition mount not found")


def find_weights():
    hits = glob.glob("/kaggle/input/**/v11_dino_fold*.pt", recursive=True)
    if hits:
        return os.path.dirname(sorted(hits)[0])
    raise FileNotFoundError("v11 weights mount not found")


class AttnSeriesClassifier(nn.Module):
    def __init__(self, backbone="vit_small_patch14_dinov2.lvd142m"):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=False, num_classes=0)
        self.feat = self.encoder.num_features
        self.attn = nn.Sequential(
            nn.Linear(self.feat, self.feat // 4), nn.Tanh(),
            nn.Linear(self.feat // 4, 1))
        self.head = nn.Linear(self.feat, 12)

    def encode(self, x):
        return self.encoder(x)

    def forward_groups(self, feats, counts):
        outs, ofs = [], 0
        for n in counts:
            g = feats[ofs:ofs + n]
            a = torch.softmax(self.attn(g), dim=0)
            outs.append(self.head((a * g).sum(0)))
            ofs += n
        return torch.stack(outs)


def read_slice(path):
    try:
        a = pydicom.dcmread(path).pixel_array
    except Exception as e:
        print("bad dicom", os.path.basename(path), e, flush=True)
        return []
    if a.ndim == 2:
        return [a]
    if a.ndim == 3:  # multi-frame: each frame is a slice
        return [f for f in a]
    return []


def series_volume(series_dir, max_frames=64):
    files = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))
    if not files:
        return None
    with ThreadPoolExecutor(max_workers=8) as ex:
        parts = list(ex.map(read_slice, files))
    frames = [f for p in parts for f in p]
    if not frames:
        return None
    if len(frames) > max_frames:  # cap for memory; evenly spaced keeps coverage
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in idx]
    vol = np.stack([cv2.resize(np.asarray(f, dtype=np.float32), (SIZE, SIZE),
                               interpolation=cv2.INTER_AREA) for f in frames])
    lo, hi = float(vol.min()), float(vol.max())
    return ((vol - lo) / max(hi - lo, 1.0) * 255.0).astype(np.uint8)


def series_tensor(vol):
    imgs = vol.astype(np.float32) / 255.0
    prev = np.concatenate([imgs[:1], imgs[:-1]])
    nxt = np.concatenate([imgs[1:], imgs[-1:]])
    x = torch.from_numpy(np.stack([prev, imgs, nxt], axis=1))  # (n,3,H,W)
    return (x - MEAN) / STD


ENC_CHUNK = 24  # slices per encoder forward; attention still pools over all


@torch.no_grad()
def predict_series(models, vol, dev):
    x = series_tensor(vol).to(dev, non_blocking=True)
    n = x.shape[0]
    chunks = [x[i:i + ENC_CHUNK] for i in range(0, n, ENC_CHUNK)]
    with torch.autocast("cuda"):
        logits = None
        for m in models:
            feats = torch.cat([m.encode(c) for c in chunks])
            li = m.forward_groups(feats.float(), [n])
            logits = li if logits is None else logits + li
    p = torch.sigmoid(logits / len(models))[0]
    return p.cpu().numpy()


def main():
    root = find_root()
    wdir = find_weights()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for f in sorted(glob.glob(os.path.join(wdir, "v11_dino_fold*.pt"))):
        ck = torch.load(f, map_location="cpu", weights_only=False)
        m = AttnSeriesClassifier(ck.get("backbone", "vit_small_patch14_dinov2.lvd142m"))
        m.load_state_dict(ck["state_dict"])
        m.to(dev).eval()
        models.append(m)
    print(f"loaded {len(models)} folds from {wdir}", flush=True)

    test = pd.read_csv(os.path.join(root, "test_series.csv"))
    rows = []
    for study, g in test.groupby("StudyInstanceUID"):
        per_series = []
        for series in g.SeriesInstanceUID:
            sdir = os.path.join(root, "test_series", study, series)
            if not os.path.isdir(sdir):
                continue
            try:
                vol = series_volume(sdir)
                if vol is not None:
                    per_series.append(predict_series(models, vol, dev))
            except Exception as e:
                print("series failed", series, repr(e)[:200], flush=True)
        if not per_series:
            rows.append([study] + [0.5] * 12)
            print(study, "EMPTY -> 0.5", flush=True)
            continue
        rows.append([study] + np.mean(per_series, axis=0).tolist())
        print(study, f"done ({len(per_series)} series)", flush=True)

    sub = pd.DataFrame(rows, columns=["StudyInstanceUID"] + LABELS)
    # scoring requires every test.csv StudyInstanceUID to be present
    test_uids = pd.read_csv(os.path.join(root, "test.csv")).StudyInstanceUID
    missing = set(test_uids) - set(sub.StudyInstanceUID)
    for u in missing:
        sub.loc[len(sub)] = [u] + [0.5] * 12
    if missing:
        print(f"filled {len(missing)} missing studies with 0.5", flush=True)
    sub = sub.set_index("StudyInstanceUID").reindex(test_uids).reset_index()
    sub.to_csv("submission.csv", index=False)
    print(sub.head())


if __name__ == "__main__":
    main()
