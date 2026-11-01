import torch
import torch.nn as nn
import timm


class SliceClassifier(nn.Module):
    """Per-slice 12-head classifier; study prob = aggregation of slice probs."""

    def __init__(self, backbone="tf_efficientnet_b0.ns_jft_in1k", pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.feat = self.encoder.num_features
        self.head = nn.Linear(self.feat, 12)

    def forward(self, x):  # x: (N, 3, H, W)
        f = self.encoder(x)
        return self.head(f)


def unfreeze_last_blocks(encoder, n=6):
    """Freeze everything except the last n transformer blocks + final norm.

    Works for timm vision transformers (dinov2); convnets are left untouched.
    """
    if hasattr(encoder, "blocks"):
        for p in encoder.parameters():
            p.requires_grad = False
        for blk in encoder.blocks[-n:]:
            for p in blk.parameters():
                p.requires_grad = True
        if hasattr(encoder, "norm"):
            for p in encoder.norm.parameters():
                p.requires_grad = True
        return
    # convnet fallback: freeze stem + first half of stages
    for p in encoder.parameters():
        p.requires_grad = False
    stages = [s for s in encoder.children() if hasattr(s, "parameters")]
    for s in stages[len(stages) // 2:]:
        for p in s.parameters():
            p.requires_grad = True


class AttnSeriesClassifier(nn.Module):
    """Encoder -> gated-attention pooling over a series' slices -> 12 logits."""

    def __init__(self, backbone="vit_small_patch14_dinov2.lvd142m", pretrained=True,
                 unfreeze_last=6):
        super().__init__()
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.feat = self.encoder.num_features
        if unfreeze_last:
            unfreeze_last_blocks(self.encoder, unfreeze_last)
        self.attn = nn.Sequential(
            nn.Linear(self.feat, self.feat // 4), nn.Tanh(),
            nn.Linear(self.feat // 4, 1))
        self.head = nn.Linear(self.feat, 12)

    def encode(self, x):  # (N, 3, H, W) -> (N, F)
        return self.encoder(x)

    def forward_groups(self, feats, counts):
        """feats (sum_n, F), counts list -> logits (B, 12)."""
        outs, ofs = [], 0
        for n in counts:
            g = feats[ofs:ofs + n]
            a = torch.softmax(self.attn(g), dim=0)
            outs.append(self.head((a * g).sum(0)))
            ofs += n
        return torch.stack(outs)


def aggregate_series(slice_logits, counts, topk=4):
    """slice_logits (N,12) -> per-series prob using mean of top-k slice probs.

    Returns list of (series_prob (12,), n_slices).
    """
    probs = torch.sigmoid(slice_logits)
    out, ofs = [], 0
    for n in counts:
        p = probs[ofs:ofs + n]                       # (n,12)
        k = min(topk, n)
        top = p.topk(k, dim=0).values.mean(0)        # optimistic pooling per label
        out.append(top)
        ofs += n
    return out
