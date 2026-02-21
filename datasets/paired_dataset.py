import os
from PIL import Image
from .base_dataset import BaseDataset


class PairedDataset(BaseDataset):
    def __init__(self, data_root, phase="train", transforms=None):
        super().__init__(data_root, transforms)
        self.phase = phase
        self.raw_dir = os.path.join(data_root, "raw")
        self.gt_dir = os.path.join(data_root, "gt")

        filenames = sorted(os.listdir(self.raw_dir))
        for fname in filenames:
            self.raw_list.append(os.path.join(self.raw_dir, fname))
            self.gt_list.append(os.path.join(self.gt_dir, fname))

    def __getitem__(self, idx):
        raw_path = self.raw_list[idx]
        gt_path = self.gt_list[idx]

        raw_img = Image.open(raw_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        if self.transforms:
            raw_img = self.transforms(raw_img)
            gt_img = self.transforms(gt_img)

        return {"raw": raw_img, "gt": gt_img, "path": raw_path}