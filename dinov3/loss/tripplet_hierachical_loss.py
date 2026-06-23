import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Any
import numpy as np
import logging

logger = logging.getLogger("dinov3.triplet")


def minmax_scale(x, eps=1e-8):
    t = torch.as_tensor(x, dtype=torch.float32)
    if t.numel() == 0:
        return t  # empty -> empty
    mn, mx = t.min(), t.max()
    return (t - mn) / (mx - mn + eps)


class TripletHCentroidLoss(nn.Module):
    """
    Triplet loss that consumes HDBSCAN path outputs (`positives`, `negatives`, `lambdas`).
    positives: list[N_total] of lists of centroid vectors (np.ndarray or list) length P_i each -> shape (P_i, D)
    negatives: list[N_total] of lists of centroid vectors (same)
    lambdas: list[N_total] of lists of floats aligned with positives
    neg_lambdas (optional): list[N_total] of lists of floats aligned with negatives
    local_indices: LongTensor (B_local,) indices for anchors in positives/negatives/lambdas lists
    """

    def __init__(
        self,
        margin: float = 0.2,
        weighting_mode: str = "weighted_mean",  # "none" | "weighted_mean" | "weighted_minmax"
        lambda_scaling: Optional[str] = "global",  # "global" | "local" | None
        negative_weighting: str = "uniform",  # "uniform" | "inverse_pos" | "based_on_pos" (simple heuristics)
        eps: float = 1e-8,
    ):
        super().__init__()
        assert weighting_mode in ("none", "weighted_mean", "weighted_minmax")
        assert lambda_scaling in ("global", "local", None)
        assert negative_weighting in ("uniform", "inverse_pos", "based_on_pos", "lambda_mean")
        self.margin = float(margin)
        self.weighting_mode = weighting_mode
        self.lambda_scaling = lambda_scaling
        self.negative_weighting = negative_weighting
        self.eps = float(eps)

    def forward(
        self,
        anchors: torch.Tensor,
        *,
        positives: Sequence[Sequence[Any]],
        negatives: Sequence[Sequence[Any]],
        lambdas: Sequence[Sequence[float]],
        neg_lambdas: Optional[Sequence[Sequence[float]]] = None,
        local_indices: torch.Tensor,
        margin: Optional[float] = None,
    ):
        if margin is None:
            margin = self.margin
        if anchors is None:
            return anchors.new_tensor(0.0), {
                "valid_count": 0,
                "total_anchors": 0,
                "weighting_mode": self.weighting_mode,
                "lambda_scaling": self.lambda_scaling,
                "negative_weighting": self.negative_weighting,
            }

        device = anchors.device
        anchors = anchors.contiguous().to(device)
        # list of indices on CPU for indexing python lists
        local_indices_list = local_indices.contiguous().cpu().tolist()
        B_local, D = anchors.shape

        # Compute quick diagnostics for the batch (counts, lambda stats)
        pos_counts = []
        neg_counts = []
        lambda_lens = []
        lambda_vals_flat = []
        for idx in local_indices_list:
            p = positives[idx] if idx < len(positives) else []
            n = negatives[idx] if idx < len(negatives) else []
            lam = lambdas[idx] if idx < len(lambdas) else []
            pos_counts.append(len(p))
            neg_counts.append(len(n))
            lambda_lens.append(len(lam))
            if lam:
                lambda_vals_flat.extend([float(x) for x in lam])

        pos_counts = np.array(pos_counts, dtype=int)
        neg_counts = np.array(neg_counts, dtype=int)
        lambda_lens = np.array(lambda_lens, dtype=int)

        # Basic logging summary
        logger.debug(f"[TripletH] batch pos_counts: mean={pos_counts.mean() if pos_counts.size else 0:.2f} "
                     f"zeros={(pos_counts==0).sum()}/{len(pos_counts)}")
        logger.debug(f"[TripletH] batch neg_counts: mean={neg_counts.mean() if neg_counts.size else 0:.2f} "
                     f"zeros={(neg_counts==0).sum()}/{len(neg_counts)}")
        if len(lambda_vals_flat) > 0:
            logger.debug(f"[TripletH] lambda stats: min={np.min(lambda_vals_flat):.6f} "
                         f"max={np.max(lambda_vals_flat):.6f} mean={np.mean(lambda_vals_flat):.6f}")
        else:
            logger.debug("[TripletH] lambda stats: no lambda values present")

        losses = []
        valid_flags = []

        # MAIN per-anchor loop
        for idx_in_batch, global_idx in enumerate(local_indices_list):
            pos_list = positives[global_idx] if global_idx < len(positives) else []
            neg_list = negatives[global_idx] if global_idx < len(negatives) else []
            lambda_list = lambdas[global_idx] if global_idx < len(lambdas) else []
            neg_lambda_list = (
                neg_lambdas[global_idx] if (neg_lambdas is not None and global_idx < len(neg_lambdas)) else []
            )

            # convert positives -> tensor
            if len(pos_list) == 0:
                valid_flags.append(False)
                continue
            try:
                pos_tensor = torch.tensor(pos_list, dtype=anchors.dtype, device=device)
            except Exception:
                pos_tensor = torch.stack(
                    [torch.as_tensor(p, dtype=anchors.dtype, device=device) for p in pos_list], dim=0
                )

            # convert negatives -> tensor, or try fallback
            if len(neg_list) == 0:
                valid_flags.append(False)
            else:
                try:
                    neg_tensor = torch.tensor(neg_list, dtype=anchors.dtype, device=device)
                except Exception:
                    neg_tensor = torch.stack(
                        [torch.as_tensor(n, dtype=anchors.dtype, device=device) for n in neg_list], dim=0
                    )

            lambda_scaled = minmax_scale(lambda_list).to(device=device, dtype=anchors.dtype)
            neg_lambda_scaled = minmax_scale(neg_lambda_list).to(device=device, dtype=anchors.dtype)

            anchor = F.normalize(anchors[idx_in_batch].float(), p=2, dim=0)   # (D,)
            pos_tensor = F.normalize(pos_tensor.float(),              p=2, dim=1) # (P,D)
            neg_tensor = F.normalize(neg_tensor.float(),              p=2, dim=1) # (N,D)

            # --- POSITIVE SUM (scalar) ---
            positive_sum = anchor.new_tensor(0.0, dtype=torch.float32)
            for i in range(pos_tensor.size(0)):
                # scalar squared L2 to the i-th positive
                d2 = (pos_tensor[i] - anchor).pow(2).sum()                        # scalar
                positive_sum = positive_sum + d2 * lambda_scaled[i]

            # --- NEGATIVE SUM (scalar) ---
            negative_sum = anchor.new_tensor(0.0, dtype=torch.float32)
            for i in range(neg_tensor.size(0)):
                # scalar L2 to the i-th negative (unit sphere: max dist ≈ 2)
                d = (neg_tensor[i] - anchor).pow(2).sum().sqrt()
                neg_term = (2 - d).clamp(min=0.0).pow(2)
                negative_sum = negative_sum + neg_term * neg_lambda_scaled[i]

            # combine (average by counts)
            raw = positive_sum / max(1, pos_tensor.size(0)) + negative_sum / max(1, neg_tensor.size(0))
            losses.append(raw)
            valid_flags.append(True)

        # If nothing valid -> large debug dump and return zero loss (but with diagnostics)
        if not any(valid_flags):
            # verbose logging
            logger.warning(
                f"[Triplet HDBScan] valid_anchors=0/{B_local} "
                f"pos_zero_fraction={(pos_counts==0).sum()}/{len(pos_counts)} "
                f"neg_zero_fraction={(neg_counts==0).sum()}/{len(neg_counts)}"
            )

            return anchors.new_tensor(0.0), {
                "valid_count": 0,
                "total_anchors": B_local,
                "weighting_mode": self.weighting_mode,
                "lambda_scaling": self.lambda_scaling,
                "negative_weighting": self.negative_weighting,
            }

        # Check if individual losses maintain gradients
        valid_losses = [l for l, v in zip(losses, valid_flags) if v]
        if valid_losses:
            # Check gradient connectivity of individual losses
            first_loss = valid_losses[0]
            logger.debug(f"First valid loss requires_grad: {first_loss.requires_grad}, grad_fn: {first_loss.grad_fn}")
        
        loss_tensor = torch.stack(valid_losses).mean()
        valid_count = int(sum(1 for v in valid_flags if v))
        stats = {
            "valid_count": valid_count,
            "total_anchors": B_local,
            "weighting_mode": self.weighting_mode,
            "lambda_scaling": self.lambda_scaling,
            "negative_weighting": self.negative_weighting,
        }
        return loss_tensor, stats