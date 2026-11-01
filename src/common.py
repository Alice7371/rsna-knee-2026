import os
from pathlib import Path

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
    "Baker's", "Contusion", "Fracture",
]

# All paths are overridable via env vars; defaults mirror the original layout.
DATA_ROOT = Path(os.environ.get("RSNA_DATA_ROOT", "E:/KaggleDataset"))
RAW_DIR = Path(os.environ.get("RSNA_RAW_DIR", DATA_ROOT / "rsna-knee"))
VOLUMES_DIR = Path(os.environ.get("RSNA_VOLUMES_DIR", DATA_ROOT / "volumes"))
META_DIR = Path(os.environ.get("RSNA_META_DIR", "data/rsna"))
WORK = Path(os.environ.get("RSNA_WORK_DIR", "artifacts"))
WORK.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = Path(os.environ.get("RSNA_TRAIN_CSV", META_DIR / "train.csv"))
TRAIN_SERIES_CSV = Path(os.environ.get("RSNA_TRAIN_SERIES_CSV", META_DIR / "train_series.csv"))
TEST_CSV = Path(os.environ.get("RSNA_TEST_CSV", META_DIR / "test.csv"))
TEST_SERIES_CSV = Path(os.environ.get("RSNA_TEST_SERIES_CSV", META_DIR / "test_series.csv"))
SAMPLE_SUB = Path(os.environ.get("RSNA_SAMPLE_SUB", META_DIR / "sample_submission.csv"))

# community LLM-mined silver labels (public Kaggle datasets, see README)
SILVER_STEVEN = Path(os.environ.get(
    "RSNA_SILVER_STEVEN", "data/labels/steven/llm_labels_v4_blend.csv"))
SILVER_PILKWANG = Path(os.environ.get(
    "RSNA_SILVER_PILKWANG", "data/labels/pilkwang/report_labels_v2.csv"))


def seed_everything(seed=42):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
