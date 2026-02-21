from __future__ import annotations
import os
from typing import List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class H5BagDataset(Dataset):
    """
    Each item is one patient bag stored as <patient_id>.h5.
    """
    def __init__(self, bag_dir: str, ids: List[str]):
        self.bag_dir = bag_dir
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        pid = str(self.ids[idx])
        path = os.path.join(self.bag_dir, f"{pid}.h5")
        with h5py.File(path, "r") as f:
            feats = f["features"][...].astype(np.float32)
            label = int(f.attrs["label"])
        return torch.from_numpy(feats), label, pid
