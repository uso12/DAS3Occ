from typing import Optional

import einops
import torch
import torch.nn.functional as F

from mmdet3d.models.builder import HEADS
from mmdet3d.models.heads.occ.bev_occ_head import BEVOCCHead2D

from ..modules.detection_guidance import DetectionGuidanceProjector
from ..modules.hard_negative_mining import hard_negative_suppression_loss
from ..modules.temporal_memory import FeatureMemoryBank


@HEADS.register_module()
class HybridBEVOCCHead2D(BEVOCCHead2D):
    """Detection-guided + temporal-memory occupancy head."""

    def __init__(
        self,
        *args,
        guidance_gain: float = 1.5,
        guidance_threshold: float = 0.2,
        guidance_blur_kernel: int = 3,
        use_temporal_memory: bool = True,
        temporal_momentum: float = 0.9,
        temporal_blend: float = 0.25,
        max_memory_entries: int = 2048,
        hard_negative_weight: float = 0.2,
        hard_negative_threshold: float = 0.15,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.guidance_gain = guidance_gain
        self.guidance_threshold = guidance_threshold
        self.guidance_projector = DetectionGuidanceProjector(guidance_blur_kernel)

        self.use_temporal_memory = use_temporal_memory
        self.temporal_memory = FeatureMemoryBank(
            momentum=temporal_momentum,
            blend=temporal_blend,
            max_entries=max_memory_entries,
        )

        self.hard_negative_weight = hard_negative_weight
        self.hard_negative_threshold = hard_negative_threshold
        self.empty_class_idx = self.num_classes - 1
        self._cached_guidance_xy: Optional[torch.Tensor] = None

    def forward(
        self,
        img_feats,
        lidar_aug_matrix=None,
        lidar2ego=None,
        occ_aug_matrix=None,
        det_guidance_logits: Optional[torch.Tensor] = None,
        metas=None,
    ):
        if isinstance(img_feats, list):
            assert len(img_feats) == 1
            img_feats = img_feats[0]

        img_feats = einops.rearrange(img_feats, "bs c w h -> bs c h w")
        img_feats = torch.nan_to_num(img_feats, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.coordinate_transform is not None:
            img_feats = self.coordinate_transform(img_feats, lidar_aug_matrix, lidar2ego, occ_aug_matrix)
            img_feats = torch.nan_to_num(img_feats, nan=0.0, posinf=1e4, neginf=-1e4)

        guidance = self.guidance_projector(det_guidance_logits, img_feats.shape[-2:])
        if guidance is not None:
            guidance = torch.nan_to_num(guidance, nan=0.0, posinf=1.0, neginf=0.0)
            hard_gate = (guidance >= self.guidance_threshold).to(img_feats.dtype)
            soft_gate = 0.5 * guidance
            img_feats = img_feats * (1.0 + self.guidance_gain * hard_gate * guidance + soft_gate)
            img_feats = torch.nan_to_num(img_feats, nan=0.0, posinf=1e4, neginf=-1e4)
            # [B, 1, H, W] -> [B, Dx=W, Dy=H, 1]
            self._cached_guidance_xy = guidance.squeeze(1).permute(0, 2, 1).unsqueeze(-1).detach()
        else:
            self._cached_guidance_xy = None

        if self.use_temporal_memory:
            img_feats = self.temporal_memory(img_feats, metas=metas)
            img_feats = torch.nan_to_num(img_feats, nan=0.0, posinf=1e4, neginf=-1e4)

        occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
        bs, dx, dy = occ_pred.shape[:3]
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
            occ_pred = occ_pred.view(bs, dx, dy, self.Dz, self.num_classes)

        occ_pred = torch.nan_to_num(occ_pred, nan=0.0, posinf=1e4, neginf=-1e4)
        return occ_pred

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        occ_pred = torch.nan_to_num(occ_pred, nan=0.0, posinf=1e4, neginf=-1e4)
        loss_dict = super().loss(occ_pred, voxel_semantics, mask_camera)

        if self.hard_negative_weight > 0 and self._cached_guidance_xy is not None:
            if self._cached_guidance_xy.shape[:3] != occ_pred.shape[:3]:
                guidance = self._cached_guidance_xy.squeeze(-1).permute(0, 2, 1).unsqueeze(1)
                guidance = F.interpolate(
                    guidance,
                    size=(occ_pred.shape[2], occ_pred.shape[1]),
                    mode="bilinear",
                    align_corners=False,
                )
                guidance = guidance.squeeze(1).permute(0, 2, 1).unsqueeze(-1)
            else:
                guidance = self._cached_guidance_xy

            guidance = torch.nan_to_num(guidance, nan=0.0, posinf=1.0, neginf=0.0)
            loss_dict["loss_occ_hnm"] = hard_negative_suppression_loss(
                occ_pred=occ_pred,
                det_guidance_xy=guidance,
                mask_camera=mask_camera,
                empty_class_idx=self.empty_class_idx,
                guidance_threshold=self.hard_negative_threshold,
                loss_weight=self.hard_negative_weight,
            )

        return loss_dict
