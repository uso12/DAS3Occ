from typing import Iterable, Optional

import torch

from mmcv.runner import auto_fp16
from mmdet3d.models import FUSIONMODELS
from mmdet3d.models.fusion_models.bevfusion import BEVFusion


@FUSIONMODELS.register_module()
class HybridBEVFusion(BEVFusion):
    """BEVFusion variant that injects detector priors into occupancy head."""

    def __init__(self, use_detection_guidance: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.use_detection_guidance = use_detection_guidance

    @staticmethod
    def _iter_prediction_dicts(pred) -> Iterable[dict]:
        if pred is None:
            return
        if isinstance(pred, dict):
            yield pred
            for value in pred.values():
                if isinstance(value, (dict, list, tuple)):
                    yield from HybridBEVFusion._iter_prediction_dicts(value)
            return
        if isinstance(pred, (list, tuple)):
            for item in pred:
                yield from HybridBEVFusion._iter_prediction_dicts(item)

    @staticmethod
    def _extract_detection_guidance(pred_dict) -> Optional[torch.Tensor]:
        if pred_dict is None:
            return None

        hm_list = []
        for item in HybridBEVFusion._iter_prediction_dicts(pred_dict):
            heatmap = item.get("heatmap", None)
            if heatmap is None or not torch.is_tensor(heatmap):
                continue
            if heatmap.dim() != 4:
                continue
            hm_list.append(heatmap.sigmoid().amax(dim=1, keepdim=True))

        if not hm_list:
            return None

        return torch.cat(hm_list, dim=1).amax(dim=1, keepdim=True)

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        return self.forward_single(
            img,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            metas,
            gt_masks_bev,
            gt_bboxes_3d,
            gt_labels_3d,
            **kwargs,
        )

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = []
        for sensor in (self.encoders if self.training else list(self.encoders.keys())[::-1]):
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                    **kwargs,
                )
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)

        if not self.training:
            features = features[::-1]

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        if self.decoder is not None:
            x = self.decoder["backbone"](x)
            x = self.decoder["neck"](x)

        batch_size = x.shape[0]

        object_pred = None
        if "object" in self.heads:
            object_pred = self.heads["object"](x, metas)

        det_guidance = None
        if self.use_detection_guidance:
            det_guidance = self._extract_detection_guidance(object_pred)

        if self.training:
            outputs = {}

            if "object" in self.heads:
                obj_losses = self.heads["object"].loss(gt_bboxes_3d, gt_labels_3d, object_pred)
                for name, val in obj_losses.items():
                    key = f"loss/object/{name}" if val.requires_grad else f"stats/object/{name}"
                    outputs[key] = val * self.loss_scale["object"] if val.requires_grad else val

            if "map" in self.heads:
                map_losses = self.heads["map"](x, gt_masks_bev)
                for name, val in map_losses.items():
                    key = f"loss/map/{name}" if val.requires_grad else f"stats/map/{name}"
                    outputs[key] = val * self.loss_scale["map"] if val.requires_grad else val

            if "occ" in self.heads:
                occ_head = self.heads["occ"]
                try:
                    occ_pred = occ_head(
                        x,
                        lidar_aug_matrix,
                        lidar2ego,
                        kwargs["occ_aug_matrix"],
                        det_guidance_logits=det_guidance,
                        metas=metas,
                    )
                except TypeError:
                    occ_pred = occ_head(x, lidar_aug_matrix, lidar2ego, kwargs["occ_aug_matrix"])

                occ_losses = occ_head.loss(occ_pred, kwargs["voxel_semantics"], kwargs["mask_camera"])
                for name, val in occ_losses.items():
                    key = f"loss/occ/{name}" if val.requires_grad else f"stats/occ/{name}"
                    outputs[key] = val * self.loss_scale["occ"] if val.requires_grad else val

            if det_guidance is not None:
                outputs["stats/occ/det_guidance_mean"] = det_guidance.detach().mean()
                outputs["stats/occ/det_guidance_max"] = det_guidance.detach().max()
            else:
                outputs["stats/occ/det_guidance_mean"] = x.detach().new_tensor(0.0)
                outputs["stats/occ/det_guidance_max"] = x.detach().new_tensor(0.0)

            return outputs

        outputs = [{} for _ in range(batch_size)]

        if "object" in self.heads:
            bboxes = self.heads["object"].get_bboxes(object_pred, metas)
            for k, (boxes, scores, labels) in enumerate(bboxes):
                outputs[k].update(
                    {
                        "boxes_3d": boxes.to("cpu"),
                        "scores_3d": scores.cpu(),
                        "labels_3d": labels.cpu(),
                    }
                )

        if "map" in self.heads:
            logits = self.heads["map"](x)
            for k in range(batch_size):
                outputs[k].update({"masks_bev": logits[k].cpu(), "gt_masks_bev": gt_masks_bev[k].cpu()})

        if "occ" in self.heads:
            occ_head = self.heads["occ"]
            try:
                occ_pred = occ_head(
                    x,
                    lidar_aug_matrix,
                    lidar2ego,
                    kwargs["occ_aug_matrix"],
                    det_guidance_logits=det_guidance,
                    metas=metas,
                )
            except TypeError:
                occ_pred = occ_head(x, lidar_aug_matrix, lidar2ego, kwargs["occ_aug_matrix"])
            occ_pred = occ_head.get_occ(occ_pred)
            for k in range(batch_size):
                outputs[k].update({"occ_pred": occ_pred[k]})

        return outputs
