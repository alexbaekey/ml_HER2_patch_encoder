from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from torchvision import transforms
import openslide

from patching import PatchSpec, generate_patch_coords, read_patch_rgb
from encoders import EncoderSpec, build_frozen_encoder


@dataclass(frozen=True)
class ExtractConfig:
    csv_path: str               # columns: slide_path, patient_id, label, split(optional)
    out_dir: str                # writes <patient_id>.h5
    patch: PatchSpec
    encoder: EncoderSpec
    batch_size: int = 128
    max_patches: Optional[int] = 4096
    seed: int = 0


def main(cfg: ExtractConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    df = pd.read_csv(cfg.csv_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, feat_dim = build_frozen_encoder(cfg.encoder, device=device)

    tfm = transforms.Compose([
        transforms.Resize((cfg.encoder.img_size, cfg.encoder.img_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    rng = np.random.default_rng(cfg.seed)

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Extract"):
        slide_path = str(row.slide_path)
        patient_id = str(row.patient_id)
        label = int(row.label)

        out_path = os.path.join(cfg.out_dir, f"{patient_id}.h5")
        if os.path.exists(out_path):
            continue

        if not os.path.exists(slide_path):
            raise FileNotFoundError(f"Slide path not found: {slide_path}")

        slide = openslide.OpenSlide(slide_path)
        coords, level = generate_patch_coords(slide, cfg.patch)

        if len(coords) == 0:
            slide.close()
            # still write an empty bag (or skip); here we skip
            print(f"[WARN] No tissue patches found for {patient_id} ({slide_path})")
            continue

        if cfg.max_patches is not None and len(coords) > cfg.max_patches:
            idx = rng.choice(len(coords), size=cfg.max_patches, replace=False)
            coords = [coords[i] for i in idx]

        feats = np.zeros((len(coords), feat_dim), dtype=np.float32)

        encoder.eval()
        with torch.no_grad():
            for i in range(0, len(coords), cfg.batch_size):
                batch_coords = coords[i:i + cfg.batch_size]
                imgs = []
                for (x0, y0) in batch_coords:
                    patch_img = read_patch_rgb(slide, x0, y0, level, cfg.patch)
                    imgs.append(tfm(patch_img))
                x = torch.stack(imgs, dim=0).to(device, non_blocking=True)
                y = encoder(x).float().cpu().numpy()
                feats[i:i + len(batch_coords)] = y

        with h5py.File(out_path, "w") as f:
            f.create_dataset("features", data=feats, compression="gzip")
            f.create_dataset("coords", data=np.asarray(coords, dtype=np.int64), compression="gzip")
            f.attrs["label"] = label
            f.attrs["patient_id"] = patient_id
            f.attrs["slide_path"] = slide_path
            f.attrs["level"] = int(level)
            f.attrs["target_mpp"] = float(cfg.patch.target_mpp or -1.0)
            f.attrs["patch_size"] = int(cfg.patch.patch_size)
            f.attrs["stride"] = int(cfg.patch.stride)

        slide.close()


if __name__ == "__main__":
    # Example config: ~20x equivalent via mpp targeting.
    cfg = ExtractConfig(
        csv_path="configs/tcga_example.csv",
        out_dir="bags_resnet50_mpp0p5",
        patch=PatchSpec(
            level=None,
            target_mpp=0.50,
            patch_size=256,
            stride=256,
            tissue_threshold=0.25,
            mask_level=None
        ),
        encoder=EncoderSpec(name="resnet50", pretrained=True, img_size=224),
        batch_size=128,
        max_patches=4096,
        seed=0,
    )
    main(cfg)
