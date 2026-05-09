import lightning as L
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def collate_fn(batch):
    return tuple(zip(*batch))


class VOCDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 4, num_workers: int = 4, dataset_name: str = "voc"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def prepare_data(self):
        torchvision.datasets.VOCDetection(self.data_dir, year="2012", image_set="train", download=True)
        torchvision.datasets.VOCDetection(self.data_dir, year="2012", image_set="val", download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = torchvision.datasets.VOCDetection(
                self.data_dir, year="2012", image_set="train",
                download=False, transforms=self._transforms()
            )
            self.val_dataset = torchvision.datasets.VOCDetection(
                self.data_dir, year="2012", image_set="val",
                download=False, transforms=self._transforms()
            )

    def _transforms(self):
        def transforms(img, target):
            img = torchvision.transforms.functional.to_tensor(img)
            boxes = []
            labels = []
            VOC_CLASSES = [
                "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
                "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
                "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
                "train", "tvmonitor"
            ]
            objects = target["annotation"]["object"]
            if isinstance(objects, dict):
                objects = [objects]
            for obj in objects:
                bbox = obj["bndbox"]
                boxes.append([
                    float(bbox["xmin"]), float(bbox["ymin"]),
                    float(bbox["xmax"]), float(bbox["ymax"])
                ])
                labels.append(VOC_CLASSES.index(obj["name"]))
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
            }
            return img, target
        return transforms

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, collate_fn=collate_fn)
