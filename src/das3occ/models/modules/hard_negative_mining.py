from typing import Optional

import torch
import torch.nn.functional as F


def hard_negative_suppression_loss(
    occ_pred: torch.Tensor,
    det_guidance_xy: Optional[torch.Tensor],
    mask_camera: torch.Tensor,
    empty_class_idx: int,
    guidance_threshold: float,
    loss_weight: float,
) -> torch.Tensor:
    """Penalize non-empty occupancy outside detector-supported regions."""
    if loss_weight <= 0 or det_guidance_xy is None:
        return occ_pred.new_tensor(0.0)

    probs = occ_pred.softmax(dim=-1)
    nonempty_prob = 1.0 - probs[..., empty_class_idx]

    # det_guidance_xy: [B, Dx, Dy, 1]
    det_mask_xy = det_guidance_xy[..., 0] >= guidance_threshold
    det_mask = det_mask_xy.unsqueeze(-1).expand_as(nonempty_prob)

    valid_mask = mask_camera.to(torch.bool)
    neg_mask = valid_mask & (~det_mask)

    if int(neg_mask.sum()) == 0:
        return occ_pred.new_tensor(0.0)

    pred_neg = nonempty_prob[neg_mask]
    target = torch.zeros_like(pred_neg)
    return F.binary_cross_entropy(pred_neg, target) * loss_weight
