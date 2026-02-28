# Gap Analysis Across Existing Approaches

## Snapshot of Strengths and Gaps

1. `DAOcc`
- Strengths: strongest reported Occ3D mIoU, explicit detector-assisted occupancy sharpening, deployment-friendly backbone.
- Gaps: weak temporal modeling; occupancy head is mostly frame-wise; false positives can persist in texture-heavy regions.

2. `STCOcc`
- Strengths: explicit sparse spatial-temporal cascade, strong sequence-level stability and ray metrics.
- Gaps: depends heavily on coarse occupancy for voxel sampling; misses thin/small objects when early sparse selection fails; no explicit object-instance prior.

3. `ALOcc`
- Strengths: strong depth-lifting pipeline, strong occupancy-flow joint modeling, good temporal history usage.
- Gaps: heavier training stack and complex multi-loss coupling; lower reported mIoU than DAOcc in the referenced configs.

4. `GaussianFlowOcc`
- Strengths: weak/sparse supervision path, temporal Gaussian representation.
- Gaps: pseudo-label pipeline is storage-heavy and noisy (large generated depth/semantic caches), increasing label-noise risk and engineering overhead.

5. `OccMamba`
- Strengths: efficient global modeling with state-space blocks; good scalability for large volumes.
- Gaps: 3D-to-1D ordering introduces locality bias tradeoffs; reported OpenOccupancy mIoU remains below DAOcc’s top Occ3D mIoU reference.

6. `SelfOcc`
- Strengths: self-supervised 3D representation learning and strong world-model alignment.
- Gaps: objective is broader than pure supervised mIoU maximization; direct supervised occupancy leaderboard performance is typically not its primary strength.

## Key Opportunity

The clearest path to higher supervised mIoU is:
- keep DAOcc’s high-quality object-aware semantics
- add STCOcc-style temporal stabilization
- explicitly suppress detector-unsupported occupancy hallucinations

## DAS3Occ Strategy

1. Detection-guided occupancy gating
- Use detector heatmaps to bias occupancy feature updates toward likely object regions.

2. Sequence-aware temporal memory
- Add lightweight sequence memory fusion to reduce frame-to-frame flicker.

3. Hard-negative suppression
- Penalize high-confidence non-empty predictions outside detector-supported regions (and valid camera mask).

4. Long-schedule optimization for 8 GPUs
- Use longer schedule, higher image resolution, and EMA to target mIoU gains over DAOcc baseline.

## Expected Behavioral Gains

- Small dynamic objects: better recall from detector-guided gating.
- Temporal consistency: reduced occupancy flicker via memory fusion.
- Cleaner free-space prediction: fewer ghost voxels via hard-negative suppression.
