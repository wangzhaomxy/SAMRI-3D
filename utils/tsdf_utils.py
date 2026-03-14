"""
TSDF (Truncated Signed Distance Field) utilities for GVT v3.

Provides:
  - compute_tsdf: per-slice binary mask → TSDF
  - prepare_tsdf_targets: batch TSDF for a full 3D volume
  - tsdf_loss: SmoothL1 + boundary-weighting loss
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn.functional as F


def compute_tsdf(binary_mask: np.ndarray, tau: float = 32.0) -> np.ndarray:
    """
    Compute Truncated Signed Distance Field from a 2D binary mask.

    Sign convention: positive inside foreground, negative outside (Kervadec convention).

    Args:
        binary_mask: (H, W) binary array, 1=foreground, 0=background.
                     Should already be at target resolution (e.g. 256×256).
        tau: Truncation threshold in pixels. Output is clamped to [-tau, +tau].

    Returns:
        tsdf: (H, W) float32 array in [-tau, +tau].

    For an empty slice (all zeros), returns all -tau.
    """
    binary_mask = binary_mask.astype(np.float32)

    # Distance for inside (foreground) voxels; 0 outside
    dist_inside = distance_transform_edt(binary_mask)

    # Distance for outside (background) voxels; 0 inside
    dist_outside = distance_transform_edt(1.0 - binary_mask)

    # SDF: positive inside, negative outside
    sdf = dist_inside - dist_outside

    # Truncate to [-tau, +tau]
    tsdf = np.clip(sdf, -tau, tau)

    return tsdf.astype(np.float32)


def prepare_tsdf_targets(gt_masks_3d: np.ndarray, label_ids: list,
                         target_size: int = 256, tau: float = 32.0) -> dict:
    """
    Prepare per-label TSDF targets for a 3D volume.

    Args:
        gt_masks_3d: (D, H_orig, W_orig) integer label map.
        label_ids: List of label integers to compute TSDF for.
        target_size: Spatial resolution for TSDF (default 256 = mask decoder output).
        tau: Truncation threshold.

    Returns:
        dict mapping label_id -> torch.Tensor of shape (D, target_size, target_size).
    """
    D = gt_masks_3d.shape[0]
    tsdf_targets = {}

    for label_id in label_ids:
        binary = (gt_masks_3d == label_id).astype(np.float32)  # (D, H, W)
        tsdf_vol = np.zeros((D, target_size, target_size), dtype=np.float32)

        for s in range(D):
            # Resize binary mask to target resolution using nearest interpolation
            slice_mask = torch.from_numpy(binary[s:s+1, :, :]).unsqueeze(0)  # (1, 1, H, W)
            slice_resized = F.interpolate(
                slice_mask, size=(target_size, target_size), mode='nearest'
            ).squeeze().numpy()  # (target_size, target_size)

            tsdf_vol[s] = compute_tsdf(slice_resized, tau=tau)

        tsdf_targets[label_id] = torch.from_numpy(tsdf_vol)  # (D, target_size, target_size)

    return tsdf_targets


def tsdf_loss(tsdf_pred: torch.Tensor,
              tsdf_gt: torch.Tensor,
              alpha: float = 4.0,
              sigma: float = 10.67,
              label_area: float = None) -> torch.Tensor:
    """
    SmoothL1 loss with boundary-focused weighting for TSDF prediction.

    Weight function: w(x) = 1 + alpha * exp(-tsdf_gt(x)^2 / (2 * sigma^2))
    - Near boundary (|tsdf_gt| ≈ 0): weight ≈ 1 + alpha = 5.0
    - Far from boundary (|tsdf_gt| ≈ tau): weight ≈ 1.0

    Args:
        tsdf_pred: (B, S, H, W) predicted TSDF
        tsdf_gt:   (B, S, H, W) ground truth TSDF, must be same shape
        alpha: Boundary upweighting factor (default 4.0)
        sigma: Gaussian width for boundary band (default tau/3 ≈ 10.67 for tau=32)
        label_area: If provided, normalize by sqrt(area) for multi-label balance.
                    Pass the foreground voxel count for this label.

    Returns:
        Scalar loss tensor.
    """
    # Boundary-focused weighting: high weight near boundary, low weight far away
    w = 1.0 + alpha * torch.exp(-tsdf_gt.pow(2) / (2.0 * sigma ** 2))

    # SmoothL1 (Huber) loss: robust to annotation noise at blurry boundaries
    elementwise_loss = F.smooth_l1_loss(tsdf_pred, tsdf_gt, reduction='none')

    # Weighted mean
    weighted_loss = (w * elementwise_loss).mean()

    # Optional label-size normalization to balance small vs large structures
    if label_area is not None and label_area > 0:
        weighted_loss = weighted_loss / (label_area ** 0.5)

    return weighted_loss
