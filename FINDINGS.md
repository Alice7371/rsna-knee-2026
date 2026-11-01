# Research Findings

Validated conclusions from experiments conducted during the competition.
All AUC numbers are macro-averaged over 12 findings on the 58 gold-annotated studies unless noted otherwise.

## Label Extraction Quality

| extractor | gold macro-AUC | notes |
|---|---|---|
| steven_v4 (best single) | 0.893 | main training target |
| pilkwang | 0.870 | has per-finding confidence |
| qwen3-8b | 0.829 | Effusion 0.607 anomalously low |
| **3-set average** | **0.893** | +0.0003 over best single — not worth it |

Three independent LLM extractions average to +0.0003. The label sets share a common error mode and are not diverse enough for ensembling to help. Further gains require improving a single extractor, not averaging across extractors.

## Backbone Comparison (same corpus, same training)

| backbone | gold macro-AUC | type |
|---|---|---|
| CoAtNet-384 (rmlp_2_rw) | 0.9025 | conv-attention hybrid |
| ConvNeXt-336 | 0.8833 | pure conv |
| Swin-B-384 | 0.8825 | window attention |
| EffNetV2-L-480 | 0.8716 | pure conv |
| MaxViT-384 | 0.8438 | pure ViT variant |

CoAtNet (conv-attention hybrid, avg-pool head) outperforms all alternatives. Pure ViT variants underperform — conv stages handle low-level MRI texture better.

## SWA Weight Averaging

SWA top-3 weight averaging (epochs 10, 9, 8) achieved gold macro-AUC 0.8974, slightly above the best single epoch (0.8970). The improvement is small but consistent.

**Caveat**: on noisy soft labels, SWA can underperform. Consider EMA (exponential moving average, decay=0.999) as a more robust alternative for future runs.

## Vision-Language Label Generation (Negative Result)

A general-purpose VL model (Claude) reading MRI slice PNGs + reports achieved:

| config | gold macro-AUC |
|---|---|
| pure image (no report) | 0.669 |
| image + report text | 0.851 |
| 3-agent ensemble (image + report) | 0.843 |

**Conclusion**: VL models without MRI-specific fine-tuning cannot match text-only extraction (0.893). The approach is not viable for label generation within the competition timeframe.

**However**: per-label analysis shows VL is **complementary** for specific findings:
| finding | VL AUC | text AUC | VL advantage |
|---|---|---|---|
| Contusion | 0.950 | 0.860 | bone marrow edema directly visible |
| Fracture | 0.932 | 0.793 | fracture lines directly visible |
| Lateral OA | 0.966 | 0.833 | cartilage loss directly visible |

A future system combining text extraction (for ACL/MCL/meniscus) with VL assessment (for Contusion/Fracture/OA) could exceed either modality alone.

## Normalisation Mismatch (Critical Pitfall)

Training script defaults to `norm="none"` ([0,1] input). The public inference notebook hardcodes `NORM="imagenet"` ((x−μ)/σ). Our model, trained on [0,1] inputs, produced NaN logits under ImageNet-normalised fp16 inference.

- **Result**: LB 0.551 (chance level)
- **Fix**: set `NORM = "none"` in inference to match training
- **LB after fix**: 0.926

**Lesson**: when reproducing a recipe, audit every preprocessing default. "The code runs without error" does not mean "the weights are loaded correctly."

## Multi-LLM Label Ensembling (Negative Result)

Ensembling labels from three independent LLM extractions yielded no improvement:
- The rate-limiting step (API signing) was rate-limited to hour-scale windows
- Zombie processes continued burning quota after being "stopped"
- Exponential backoff with single-flight signing is required
- Label set quality is bounded by the weakest link in the extraction pipeline

## CoAtNet Training on Heterogeneous Hardware

The public CoAtNet-384 recipe (3h on RTX 4090) was reproduced on an RTX 3080 (5.5h) with SWA top-3 weight averaging achieving gold macro-AUC 0.8974. The submission scored 0.926 on the leaderboard, matching the author's published result.
