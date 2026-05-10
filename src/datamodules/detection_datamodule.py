import lightning as L
import torch
import torchvision
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np


def collate_fn(batch):
    return tuple(zip(*batch))


class FakeCOCODataset(Dataset):
    """Small synthetic dataset for object detection testing."""
    def __init__(self, size=1000, num_classes=91):
        self.size = size
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = torch.rand(3, 224, 224)
        num_boxes = torch.randint(1, 5, (1,)).item()
        boxes = torch.rand(num_boxes, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes = torch.clamp(boxes, 0, 1) * 224
        labels = torch.randint(1, self.num_classes, (num_boxes,))
        target = {"boxes": boxes, "labels": labels}
        return img, target


class VOCDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 4, num_workers: int = 4, dataset_name: str = "voc"):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = FakeCOCODataset(size=1000)
            self.val_dataset = FakeCOCODataset(size=200)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=collate_fn)
