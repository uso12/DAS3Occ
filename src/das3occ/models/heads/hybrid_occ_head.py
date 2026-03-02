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

    def _align_guidance_to_occ_bounds(
        self, det_guidance_logits: Optional[torch.Tensor], target_hw
    ) -> Optional[torch.Tensor]:
        if det_guidance_logits is None or det_guidance_logits.dim() != 4:
            return det_guidance_logits

        # Object head guidance is [B, C, X, Y], while occ features are [B, C, Y, X].
        guidance = det_guidance_logits.permute(0, 1, 3, 2).contiguous()

        # Without coordinate transform there is no physical-range remap to apply.
        if self.coordinate_transform is None:
            return guidance

        ctf = self.coordinate_transform
        required_attrs = ("ref_points", "lidar_x_min", "lidar_x_max", "lidar_y_min", "lidar_y_max")
        if not all(hasattr(ctf, k) for k in required_attrs):
            return guidance

        ref_points = ctf.ref_points
        if not torch.is_tensor(ref_points) or ref_points.numel() == 0:
            return guidance

        occ_x_min = float(ref_points[:, 0].min().item())
        occ_x_max = float(ref_points[:, 0].max().item())
        occ_y_min = float(ref_points[:, 1].min().item())
        occ_y_max = float(ref_points[:, 1].max().item())

        lidar_x_min = float(ctf.lidar_x_min)
        lidar_x_max = float(ctf.lidar_x_max)
        lidar_y_min = float(ctf.lidar_y_min)
        lidar_y_max = float(ctf.lidar_y_max)

        x_range = max(lidar_x_max - lidar_x_min, 1e-6)
        y_range = max(lidar_y_max - lidar_y_min, 1e-6)

        # Map desired occ-range endpoints into source guidance normalized coordinates.
        x0 = 2.0 * (occ_x_min - lidar_x_min) / x_range - 1.0
        x1 = 2.0 * (occ_x_max - lidar_x_min) / x_range - 1.0
        y0 = 2.0 * (occ_y_min - lidar_y_min) / y_range - 1.0
        y1 = 2.0 * (occ_y_max - lidar_y_min) / y_range - 1.0

        ax = 0.5 * (x1 - x0)
        bx = 0.5 * (x1 + x0)
        ay = 0.5 * (y1 - y0)
        by = 0.5 * (y1 + y0)

        bsz = guidance.shape[0]
        theta = guidance.new_tensor([[ax, 0.0, bx], [0.0, ay, by]]).unsqueeze(0).repeat(bsz, 1, 1)
        grid = F.affine_grid(
            theta,
            size=[bsz, guidance.shape[1], target_hw[0], target_hw[1]],
            align_corners=False,
        )
        guidance = F.grid_sample(
            guidance,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return guidance

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
        img_feats = torch.nan_to_num(img_feats, nan=0.0, posinf=10.0, neginf=-10.0)

        if self.coordinate_transform is not None:
            img_feats = self.coordinate_transform(img_feats, lidar_aug_matrix, lidar2ego, occ_aug_matrix)
            img_feats = torch.nan_to_num(img_feats, nan=0.0, posinf=10.0, neginf=-10.0)

        det_guidance_logits = self._align_guidance_to_occ_bounds(det_guidance_logits, img_feats.shape[-2:])

        guidance = self.guidance_projector(det_guidance_logits, img_feats.shape[-2:])
        if guidance is not None:
            guidance = torch.nan_to_num(guidance, nan=0.0, posinf=1.0, neginf=0.0)
            hard_gate = (guidance >= self.guidance_threshold).to(img_feats.dtype)
            soft_gate = 0.5 * guidance
            img_feats = img_feats * (1.0 + self.guidance_gain * hard_gate * guidance + soft_gate)
            img_feats = torch.nan_to_num(img_feats, nan=0.0, posinf=10.0, neginf=-10.0)
            # [B, 1, H, W] -> [B, Dx=W, Dy=H, 1]
            self._cached_guidance_xy = guidance.squeeze(1).permute(0, 2, 1).unsqueeze(-1).detach()
        else:
            self._cached_guidance_xy = None

        if self.use_temporal_memory:
            img_feats = self.temporal_memory(img_feats, metas=metas)
            img_feats = torch.nan_to_num(img_feats, nan=0.0, posinf=10.0, neginf=-10.0)

        occ_pred = self.final_conv(img_feats).permute(0, 3, 2, 1)
        bs, dx, dy = occ_pred.shape[:3]
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
            occ_pred = occ_pred.view(bs, dx, dy, self.Dz, self.num_classes)

        occ_pred = torch.nan_to_num(occ_pred, nan=0.0, posinf=10.0, neginf=-10.0)
        return occ_pred

    def loss(self, occ_pred, voxel_semantics, mask_camera):
        occ_pred = torch.nan_to_num(occ_pred, nan=0.0, posinf=10.0, neginf=-10.0)
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
                voxel_semantics=voxel_semantics,
                empty_class_idx=self.empty_class_idx,
                guidance_threshold=self.hard_negative_threshold,
                loss_weight=self.hard_negative_weight,
            )

        return loss_dict
