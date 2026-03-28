import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torchvision.tv_tensors as tv

class FaceSegDataset(Dataset):
    def __init__(self, root: str, split: str = "train"):
        self.img_dir  = Path(root) / split / "images"
        self.mask_dir = Path(root) / split / "masks"
        self.ids      = sorted([p.stem for p in self.img_dir.glob("*.jpg")])

        self.split = split
        self.train_size = 256
        self.val_size   = 512

        self._setup_transforms()

    def _setup_transforms(self):
        size = self.train_size if self.split == "train" else self.val_size

        base = v2.Compose([
            v2.ToDtype({tv.Image: torch.float32, tv.Mask: torch.int64}, scale=True),
            v2.Normalize(mean=[0.5081, 0.4134, 0.3607],
                         std=[0.3013, 0.2757, 0.2692]),
        ])
        if self.split == "train":
            aug = v2.Compose([
                v2.RandomAffine(
                    degrees=30,
                    translate=(0.1, 0.1),
                    scale=(0.5, 3.0),
                    shear=20,
                    fill=0,
                ),
                v2.ColorJitter(
                    brightness=(0.5, 1.5),
                    contrast=(0.0, 2.0),
                    saturation=(0.5, 1.5),
                    hue=(-0.3, 0.3),
                ),
            ])
            self.transforms = v2.Compose([v2.Resize((size, size)), aug, base])
        else:
            self.transforms = v2.Compose([v2.Resize((size, size)), base])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        stem = self.ids[idx]
        image = tv.Image(
            torch.from_numpy(
                np.array(Image.open(self.img_dir / f"{stem}.jpg").convert("RGB"))
            ).permute(2, 0, 1)
        )
        mask = tv.Mask(
            torch.from_numpy(
                np.array(Image.open(self.mask_dir / f"{stem}.png"))
            ).long()
        )
        image, mask = self.transforms(image, mask)
        return image, mask.squeeze(0)