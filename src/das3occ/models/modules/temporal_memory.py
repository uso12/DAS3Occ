from collections import OrderedDict
from typing import Any, Dict, Iterable

import torch
import torch.nn as nn


class FeatureMemoryBank(nn.Module):
    """Sequence-keyed EMA feature memory for lightweight temporal stabilization."""

    def __init__(
        self,
        momentum: float = 0.9,
        blend: float = 0.25,
        max_entries: int = 2048,
    ) -> None:
        super().__init__()
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend must be in [0, 1]")
        self.momentum = momentum
        self.blend = blend
        self.max_entries = max_entries
        self._memory: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    @staticmethod
    def _key_from_meta(meta: Dict[str, Any], sample_idx: int) -> str:
        for key in ("sequence_group_idx", "scene_token", "scene_name", "sample_idx"):
            if key in meta:
                return str(meta[key])
        return f"sample_{sample_idx}"

    def _prune(self) -> None:
        while len(self._memory) > self.max_entries:
            self._memory.popitem(last=False)

    @staticmethod
    def _sanitize(x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

    @torch.no_grad()
    def forward(self, feats: torch.Tensor, metas: Iterable[Dict[str, Any]] = None) -> torch.Tensor:
        if metas is None:
            return feats

        metas = list(metas)
        if len(metas) != feats.shape[0]:
            return feats

        feats = self._sanitize(feats)
        fused = feats.clone()
        for i, meta in enumerate(metas):
            key = self._key_from_meta(meta, i)
            hist = self._memory.get(key)

            if hist is not None:
                hist = self._sanitize(hist.to(device=feats.device, dtype=feats.dtype))
                fused_i = (1.0 - self.blend) * feats[i] + self.blend * hist
                fused[i] = self._sanitize(fused_i)
                updated = self.momentum * hist + (1.0 - self.momentum) * feats[i].detach()
            else:
                updated = feats[i].detach()

            updated = self._sanitize(updated)
            if torch.isfinite(updated).all():
                self._memory[key] = updated.cpu()
            elif key in self._memory:
                self._memory.pop(key, None)

        self._prune()
        return fused
