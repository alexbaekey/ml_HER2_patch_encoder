from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import openslide


@dataclass(frozen=True)
class PatchSpec:
    """
    If level is None, we auto-select the OpenSlide level closest to target_mpp.
    patch_size/stride are specified in pixels at the selected read level.
    """
    level: Optional[int] = None
    target_mpp: Optional[float] = 0.50  # ~20x equivalent if mpp0 ~0.25
    patch_size: int = 256
    stride: int = 256
    tissue_threshold: float = 0.25
    mask_level: Optional[int] = None  # if None, use coarsest


def get_slide_mpp(slide: openslide.OpenSlide) -> Optional[float]:
    """
    Return microns-per-pixel (MPP) at level 0 if available.
    TCGA SVS often includes openslide.mpp-x / openslide.mpp-y.
    """
    props = slide.properties
    mpp_x = props.get("openslide.mpp-x")
    mpp_y = props.get("openslide.mpp-y")
    if mpp_x is None and mpp_y is None:
        return None
    try:
        mpp_xf = float(mpp_x) if mpp_x is not None else None
        mpp_yf = float(mpp_y) if mpp_y is not None else None
        if mpp_xf is None:
            return mpp_yf
        if mpp_yf is None:
            return mpp_xf
        return 0.5 * (mpp_xf + mpp_yf)
    except Exception:
        return None


def choose_level_for_target_mpp(slide: openslide.OpenSlide, target_mpp: float = 0.50) -> int:
    """
    Choose OpenSlide level whose effective MPP is closest to target_mpp.
    If MPP missing, fall back to level 0.
    """
    mpp0 = get_slide_mpp(slide)
    if mpp0 is None:
        return 0

    best_level = 0
    best_diff = float("inf")
    for lvl, ds in enumerate(slide.level_downsamples):
        eff_mpp = mpp0 * float(ds)
        diff = abs(eff_mpp - target_mpp)
        if diff < best_diff:
            best_diff = diff
            best_level = lvl
    return best_level


def _rgb_to_tissue_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Simple tissue detector for H&E: background is bright and low-saturation.
    Returns boolean mask (H,W): True = tissue.
    """
    intensity = rgb.astype(np.float32).mean(axis=-1)
    dark = intensity < 220.0

    mx = rgb.astype(np.float32).max(axis=-1)
    mn = rgb.astype(np.float32).min(axis=-1)
    sat = (mx - mn) > 10.0
    return dark & sat


def generate_patch_coords(slide: openslide.OpenSlide, spec: PatchSpec) -> Tuple[List[Tuple[int, int]], int]:
    """
    Returns (coords, level) where coords are patch top-left positions in level-0 frame (x0,y0).
    """
    level = spec.level if spec.level is not None else choose_level_for_target_mpp(
        slide, target_mpp=float(spec.target_mpp or 0.50)
    )

    # Pick a coarse mask level
    mask_level = spec.mask_level if spec.mask_level is not None else (slide.level_count - 1)

    dims_mask = slide.level_dimensions[mask_level]
    thumb = slide.read_region((0, 0), mask_level, dims_mask).convert("RGB")
    thumb_np = np.asarray(thumb)
    tissue_mask = _rgb_to_tissue_mask(thumb_np)

    downsample_mask = float(slide.level_downsamples[mask_level])
    downsample_read = float(slide.level_downsamples[level])

    w0, h0 = slide.dimensions
    wL = int(np.floor(w0 / downsample_read))
    hL = int(np.floor(h0 / downsample_read))

    patch = spec.patch_size
    stride = spec.stride

    coords: List[Tuple[int, int]] = []

    # Grid in read-level coords; convert each top-left back to level-0
    for yL in range(0, max(hL - patch + 1, 0), stride):
        for xL in range(0, max(wL - patch + 1, 0), stride):
            x0 = int(xL * downsample_read)
            y0 = int(yL * downsample_read)

            # Corresponding region in mask coordinates
            xM0 = int(x0 / downsample_mask)
            yM0 = int(y0 / downsample_mask)

            # Patch extent in level-0 pixels is patch*downsample_read
            # Convert extent to mask pixels
            wM = max(int(np.ceil((patch * downsample_read) / downsample_mask)), 1)
            hM = max(int(np.ceil((patch * downsample_read) / downsample_mask)), 1)

            xM1 = min(xM0 + wM, tissue_mask.shape[1])
            yM1 = min(yM0 + hM, tissue_mask.shape[0])
            if xM0 >= xM1 or yM0 >= yM1:
                continue

            frac = float(tissue_mask[yM0:yM1, xM0:xM1].mean())
            if frac >= spec.tissue_threshold:
                coords.append((x0, y0))

    return coords, level


def read_patch_rgb(slide: openslide.OpenSlide, x0: int, y0: int, level: int, spec: PatchSpec) -> Image.Image:
    """
    Read a patch at (x0,y0) in level-0 frame, from the selected level.
    """
    return slide.read_region((x0, y0), level, (spec.patch_size, spec.patch_size)).convert("RGB")
