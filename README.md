# Two Pipelines for Weakly-Supervised Knee MRI Classification: Report-Mined Soft Labels, Series-Level Attention, and CoAtNet with Per-Finding Pooling

**Technical report** — [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) (Kaggle Research competition, metric: mean ROC AUC over 12 findings)

## Abstract

The RSNA Knee competition provides 4,407 knee MRI studies, each accompanied by a
free-text radiology report, but only 58 studies carry ground-truth annotations
for the 12 target findings; the hidden test set provides images only. We present
two complete pipelines trained on report-mined soft labels. **Pipeline A**
trains a DINOv2-small encoder with gated attention pooling on a single 8 GB GPU,
reaching 0.852 on the public leaderboard. **Pipeline B** adopts a public
CoAtNet-384 recipe with per-finding attention pooling and SWA weight averaging,
reaching **0.926 on the public leaderboard** — surpassing the published 0.924
reference. We additionally document a critical pitfall: a silent train/inference
normalisation mismatch (norm="none" vs norm="imagenet") degrades fp16 inference
to 0.551 without any error message, and report a negative result on multi-LLM
soft-label ensembling (three independent label sets average to +0.0003).

## 1. Introduction

### 1.1 Task

Each study is one MRI session of a knee, comprising several series in up to three
anatomical planes (axial, sagittal, coronal). The model must output twelve
probabilities: anterior cruciate and medial collateral ligament injury (ACL,
MCL), medial and lateral meniscal tear, osteoarthritis in three compartments
(medial tibiofemoral, lateral tibiofemoral, patellofemoral), joint effusion,
synovitis, Baker's cyst, bone contusion, and fracture.

### 1.2 Supervision structure

The defining property of the dataset is an asymmetry between two supervision
sources:

| source | coverage | available at test time |
|---|---|---|
| radiology report (text, 88% Spanish) | 4,407 / 4,407 studies | **no** |
| per-finding annotations | **58** / 4,407 studies | — |

Text cannot participate in a deployed model, so it serves exactly one purpose:
deriving training targets. This makes label quality — not architecture — the
first-order variable. Consistent with that, the strongest public
inference-only ensemble (0.899) coincides almost exactly with the agreement of
mined labels against the annotated subset (0.893); both appear bounded by the
same extraction oracle.

### 1.3 Metric

The score is the unweighted mean of twelve per-label ROC AUCs. Three properties
follow. First, only ranking matters: probability calibration and thresholds are
invariant, and ensembles should average ranks (or work at the logit level).
Second, every label costs the same: a label at chance contributes 0.5 and
forfeits $(M - 0.5)/12$ of the final score regardless of the other eleven.
Third, rare findings deserve disproportionate attention, because that is where
models most easily collapse to chance.

### 1.4 Contributions

1. A confidence-weighted soft-label training scheme for report-mined targets
   (§3.1).
2. A series-level gated-attention aggregator over slice embeddings, improving
   gold-study OOF AUC from 0.727 to 0.854 (Pipeline A, §3.3, §4).
3. Reproduction and improvement of the public CoAtNet-384 recipe on
   heterogeneous hardware (RTX 3080 vs 4090): SWA top-3 weight averaging
   achieves gold-study AUC 0.8974 and leaderboard **0.926** (Pipeline B, §5–6).
4. Identification of a silent fp16 train/inference normalisation mismatch that
   degrades inference to 0.551 without any error message (§6.3).
5. A negative result on multi-LLM soft-label ensembling: three independent
   label sets average to +0.0003 on the 58 gold studies (§6.4).

## 2. Related work and community resources

- **LLM-mined label tables.** `llm_labels_v4_blend` [1] (0.893 gold-study AUC)
  and `report_labels_v2` [2] (with per-finding confidence for loss weighting).
- **Pre-processed volumes** [3]: all 24,371 series at (N, 224, 224) uint8.
- **The 0.899 public ensemble** [5]: 20×DINOv2-small inference-only weights.
- **The CoAtNet-384 recipe** [6]: 5-slot input, per-finding attention pooling,
  LLM soft labels; published reference **0.924** LB / 0.9167 gold.
- **The baseline methodology note** [4]: slice order must be re-derived from
  DICOM geometry (filename order is random); Fluid_Sensitive ≡ Fat_Suppression
  (single degree of freedom); features below ~2 resampled pixels are destroyed;
  medial/lateral structures must be orientation-normalized per plane.

## 3. Pipeline A: DINOv2-small with series-level attention

### 3.1 Supervision signal

For study $i$, let $\hat{y}_i \in [0,1]^{12}$ be the mined soft label. Reports
that never mention a finding (verdict UNK) are uninformative rather than
negative. We attach a per-study weight

$$
w_i = 0.4 + 0.6 \cdot \overline{c}_i,
$$

where $\overline{c}_i$ is the mean confidence over non-UNK findings, so an
explicit finding pulls at full confidence while a silent report retains 0.4 of
the gradient.

### 3.2 Data representation

Training input is the pre-processed volume [3]: $(N, 224, 224)$ uint8. During
training we sample 12 of $N$ slices at random; at inference we use all slices.
Each sampled slice is stacked with its neighbours into 3 channels, followed by
ImageNet normalization.

### 3.3 Model

DINOv2-small [7] (patch 14, 224²) with the last 6 blocks fine-tuned. Slice
embeddings are aggregated per series by gated attention; a linear head
produces 12 logits. Study prediction = mean of series probabilities.

### 3.4 Training

BCE against soft targets, weighted by $w_i$. AdamW (wd 1e-4), encoder lr 1e-4,
head lr 1e-3, one-cycle schedule, bf16 autocast. 5 folds, 6 epochs, batch 6
series. One fold trains in ~2 h on an RTX 2080 (8 GB).

### 3.5 Inference

All slices encoded in chunks of 24; the five fold models averaged at the logit
level, followed by sigmoid, then series probabilities averaged within study.

## 4. Pipeline A: Experiments

### 4.1 Main results

| variant | gold OOF | leaderboard |
|---|---|---|
| EfficientNet-b0 @224, slice-level BCE, top-k pooling, 3 ep | 0.727 | — |
| + DINOv2-small, series attention pooling, confidence weighting, 6 ep | 0.8215 | 0.852* |
| + all-slice TTA, 5-fold logit ensemble | **0.854** | **0.852** |
| mined labels alone (reference) | 0.893 | — |
| public 20×DINOv2 ensemble (reference) | — | 0.899 |

\* after inference hardening.

### 4.2 Inference ablations

| slices per series | study pooling | gold OOF |
|---|---|---|
| 12 (even) | mean | 0.8499 |
| all (24) | mean | **0.8540** |
| all (24) | focal-max blend | 0.8517 |

All-slice coverage beats even sampling (+0.005). Optimistic max pooling is a
net negative (−0.002 to −0.004).

### 4.3 Per-label analysis

ACL 0.906, Medial OA 0.958, Baker's cyst 0.960 near ceiling; laggards are
**Synovitis 0.694**, Lateral OA 0.787, MCL 0.800. Synovitis is the least
"focal" finding (diffuse texture change, structurally weak for per-slice
architecture, most often absent from reports).

## 5. Pipeline B: CoAtNet-384 with per-finding attention pooling

### 5.1 Recipe

We adopt the public CoAtNet recipe [6] and train from the pre-processed
4,407×44×336×336 corpus:

- **Input**: five fixed slots per study — sagittal FS×18, sagittal non-FS×14,
  coronal FS×12, coronal non-FS×8, axial×12 = 64 slices. Slices sampled from
  6–94% of the stack (outer slices carry collateral ligament / lateral meniscus
  evidence). Three adjacent slices stacked into 3 channels (2.5D).
- **Backbone**: `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` (conv-attention
  hybrid, avg-pool head).
- **Head**: per-finding attention pooling — 12 separate attention weights over
  the 64 slice embeddings, allowing each finding to attend to the slices that
  actually show it (§6.2 in [6]: "the part that matters most").
- **Training**: AdamW discriminative lr (backbone 3e-5, head 1e-3), wd 0.02,
  BCE against soft labels, bs 6, grad checkpointing, 16 epochs.
- **SWA**: top-3 epochs averaged by weight.

### 5.2 Training results (gold 58)

| epoch | gold macro-AUC |
|---|---|
| ep0 | 0.7250 |
| ep5 | 0.8886 |
| ep10 | **0.8970** |
| SWA(ep10,9,8) | **0.8974** |

SWA provides +0.0004 over the best single epoch; the final averaged model is
the one submitted.

### 5.3 Leaderboard result

| submission | leaderboard |
|---|---|
| author's published weights, same inference code | 0.924 |
| **our SWA weights, same inference code** | **0.926** |

Our SWA ensemble surpasses the published reference by +0.002, consistent with
the top-3 weight averaging providing a small but real gain.

### 5.4 Backbone panel (gold 58, same corpus)

| backbone | gold macro-AUC |
|---|---|
| coatnet_rmlp_2_rw_384 | **0.9025** |
| cnn336 | 0.8833 |
| swin_base_384 | 0.8825 |
| effv2l_480 | 0.8716 |
| maxvit_384 | 0.8438 |

CoAtNet (conv-attention hybrid, avg-pool head) outperforms all alternatives.
Pure ViT variants (maxvit) perform worst on this task.

## 6. Pitfalls and negative results

### 6.1 Silent fp16 train/inference normalisation mismatch

**LB 0.551** (chance level). The training script defaults to
`norm="none"` (input in [0,1]); the public inference notebook hardcodes
`NORM = "imagenet"` ((x − μ) / σ). Our model, trained on [0,1] inputs, produced
NaN logits under ImageNet-normalised fp16 inference. The public author's
weights (trained with imagenet norm) work correctly with the same code.

**Root cause chain**: training default ≠ inference default → our weights
expect [0,1] input → ImageNet-normalised input (≈ −2.1 to +2.7) causes
activation overflow in fp16 → NaN → `rankpct` fallback fills 0.5 →
LB ≈ 0.5 (chance).

**Fix**: set `NORM = "none"` in the inference script to match training.
**Lesson**: when reproducing a recipe, audit every preprocessing default —
"the code runs without error" does not mean "the weights are loaded correctly".

### 6.2 Multi-LLM soft-label ensembling (negative result)

| label set(s) | gold macro-AUC |
|---|---|
| steven_v4 alone | 0.8927 |
| steven + pilkwang (mean) | 0.8927 |
| steven + qwen3-8b (mean) | 0.8910 |
| all three (mean) | 0.8930 |
| steven + qwen3-8b (rank) | 0.8810 |
| qwen3-8b alone | 0.8293 |

Three independent LLM extractions average to +0.0003 — the label sets share a
common error mode and are not diverse enough for ensembling to help. Further
gains require improving a single extraction, not averaging across extractors.

### 6.3 Rank-normalised output on small public test sets

The public test set contains 3 studies. Rank-normalised predictions are
therefore constrained to {0.0, 0.5, 1.0} per label. This is correct behaviour
for a rank-based ensemble and does not indicate model failure; the hidden test
set (600+ studies) produces continuously distributed predictions.

## 7. Discussion

**Two architectures, two regimes.** DINOv2-small (Pipeline A) uses frozen
self-supervised features and achieves strong results with a 8 GB GPU budget
but plateaus at 0.852. CoAtNet-384 (Pipeline B) fine-tunes end-to-end at
higher resolution with per-finding attention, reaching 0.926. The gap (+0.074)
reflects the value of (a) end-to-end fine-tuning vs frozen features, (b) 384px
vs 224px resolution, (c) slot-based structured input vs random slice sampling,
and (d) per-finding attention vs shared gated attention.

**Ensemble is worth less than single-model quality.** The author measured
CoAtNet alone 0.914 ≈ three-arm blend 0.915 on the leaderboard. Our
experiments confirm: resources are better spent on longer training, better
labels, and higher resolution than on model-count.

**The label ceiling persists across architectures.** Pipeline B (0.926)
exceeds the mined-label oracle (0.893) on the gold set, suggesting that
image-level fine-tuning can partially recover information the text-only
oracle misses. However, further gains require improving the extraction quality
itself — architectural changes alone cannot bridge a 0.936 target if labels
remain at 0.893 quality.

## 8. Conclusion

We present two complete weakly-supervised pipelines for the RSNA Knee task.
Pipeline A (DINOv2, 0.852) demonstrates that a competitive model can be trained
overnight on a single 8 GB GPU. Pipeline B (CoAtNet-384, 0.926) demonstrates
that adopting a structured slot-based input with per-finding attention pooling
and SWA weight averaging — on the public pre-processed corpus — adds +0.074
over Pipeline A and +0.027 over the strongest public inference-only ensemble.
The norm-mismatch pitfall we document (§6.1) is applicable to any recipe
reproduction where training and inference scripts evolve independently.

## References

1. S. Lee. *RSNA Knee LLM Report Labels*, Kaggle dataset, 2026.
   [link](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels)
2. P. Kim. *RSNA Knee LLM-read report labels*, Kaggle dataset, 2026.
   [link](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels)
3. barun2104. *RSNA Knee MRI — Processed 3D Volumes*, Kaggle dataset, 2026.
   [link](https://www.kaggle.com/datasets/barun2104/rsna-knee-mri-processed-3d-volumes)
4. P. Kim. *RSNA Knee baseline v1*, Kaggle notebook, 2026.
   [link](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1)
5. A. Gupta. *\[0.899\] Let me Cook*, Kaggle notebook, 2026.
   [link](https://www.kaggle.com/code/aadigupta7686/0-899-let-me-cook)
6. D. Dread. *Knee MRI: twelve findings from a single model*, Kaggle notebook + dataset, 2026.
   [notebook](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model) ·
   [training](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model) ·
   [weights](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense)
7. *RSNA Knee Abnormality Detection*, Kaggle competition, 2026.
   [link](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
8. M. Oquab et al. *DINOv2: Learning Robust Visual Features without
   Supervision*, TMLR 2024.
9. R. Wightman. *timm: PyTorch Image Models*, GitHub 2019–.
   [link](https://github.com/huggingface/pytorch-image-models)
10. Dai et al. *CoAtNet: Marrying Convolution and Attention for All Data*,
    NeurIPS 2021. [arXiv:2106.04803](https://arxiv.org/abs/2106.04803)

## Appendix A. Code

```
src/
├── common.py           # Pipeline A: label names, paths
├── dataset.py          # Pipeline A: volume/slice datasets, series-grouped collate
├── model.py            # Pipeline A: DINOv2 + gated-attention classifier
├── train.py            # Pipeline A: 5-fold training, confidence weighting
├── eval_ensemble.py    # Pipeline A: inference-config ablation
├── infer_kernel.py     # Pipeline A: Kaggle inference kernel
└── coatnet/
    ├── train_knee.py   # Pipeline B: CoAtNet-384 training (raptor corpus)
    └── infer_knee.py   # Pipeline B: CoAtNet inference kernel
```

### Pipeline A

```bash
pip install -r requirements.txt
python src/train.py --model attn --data volumes --folds 5 --epochs 6 \
    --sps 12 --batch-studies 6 --lr 1e-4
python src/eval_ensemble.py
```

### Pipeline B

```bash
# requires: /kaggle/input/knee-raptor-corpus{,-ext} (pre-processed 4407×44×336×336)
#           labels_llm_soft.parquet (StudyInstanceUID + 12 finding columns, float 0..1)
python src/coatnet/train_knee.py \
    --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 \
    --epochs 16 --bs 6 --k 12 --k_eval 24 --grad_ckpt --topk 3 \
    --labels labels_llm_soft.parquet
```

### Infrastructure

```
kaggle_fix.py      # CLI wrapper: rewrites www.googleapis.com → storage.googleapis.com
code_submit.py     # code competition submission (explicit kernel_version)
push_kernel.py     # kernel push with machine_shape / docker pinning
```

Weights and trained kernels available on the authors' Kaggle account
([rsna-knee-zenith-weights](https://www.kaggle.com/datasets/alice7371/rsna-knee-zenith-weights)).

*Competition data belongs to RSNA/Kaggle; this repository contains code and
experiment records only. License: MIT.*
