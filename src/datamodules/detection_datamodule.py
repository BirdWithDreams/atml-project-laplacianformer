import lightning as L
import torch
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


def collate_fn(batch):
    return tuple(zip(*batch))


class COCODetectionDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item["image"].convert("RGB")
        img = F.to_tensor(img)

        objects = item["objects"]
        boxes = torch.tensor(objects["bbox"], dtype=torch.float32)
        labels = torch.tensor(objects["category"], dtype=torch.int64)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        return img, target


class VOCDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 4, num_workers: int = 4, dataset_name: str = "voc"):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        dataset = load_dataset("detection-datasets/coco", split="train[:5000]", trust_remote_code=True)
        val_dataset = load_dataset("detection-datasets/coco", split="validation[:1000]", trust_remote_code=True)

        if stage == "fit" or stage is None:
            self.train_dataset = COCODetectionDataset(dataset)
            self.val_dataset = COCODetectionDataset(val_dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=collate_fn)
