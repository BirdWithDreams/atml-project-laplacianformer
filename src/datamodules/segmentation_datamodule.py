import os
import random
from typing import Optional

import lightning as L
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SegmentationTransform:
    def __init__(
            self,
            image_size: int = 224,
            train: bool = False,
            mean: tuple[float, float, float] = IMAGENET_MEAN,
            std: tuple[float, float, float] = IMAGENET_STD,
            ):
        self.image_size = image_size
        self.train = train
        self.mean = mean
        self.std = std

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.convert("RGB")

        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )

        if self.train and random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image = TF.normalize(TF.to_tensor(image), self.mean, self.std)
        mask_tensor = torch.as_tensor(np.array(mask, dtype=np.int64), dtype=torch.long)
        return image, mask_tensor


class VOCSegmentationWrapper(Dataset):
    def __init__(self, dataset: Dataset, transform: SegmentationTransform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.dataset[index]
        return self.transform(image, mask)


class CityscapesSegmentationWrapper(Dataset):
    def __init__(
            self,
            dataset: Dataset,
            transform: SegmentationTransform,
            ignore_index: int = 255,
            ):
        self.dataset = dataset
        self.transform = transform
        self.label_id_to_train_id = self._build_label_id_to_train_id(ignore_index)

    @staticmethod
    def _build_label_id_to_train_id(ignore_index: int) -> np.ndarray:
        label_id_to_train_id = np.full(256, ignore_index, dtype=np.uint8)
        for cityscapes_class in torchvision.datasets.Cityscapes.classes:
            label_id = cityscapes_class.id
            train_id = cityscapes_class.train_id
            if 0 <= label_id < label_id_to_train_id.shape[0]:
                label_id_to_train_id[label_id] = (
                    train_id if 0 <= train_id < ignore_index else ignore_index
                )
        return label_id_to_train_id

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.dataset[index]
        mask_array = np.array(mask, dtype=np.uint8)
        mask = Image.fromarray(self.label_id_to_train_id[mask_array])
        return self.transform(image, mask)


class SegmentationDataModule(L.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "./data",
            dataset_name: str = "voc2012",
            batch_size: int = 16,
            num_workers: int = 4,
            image_size: int = 224,
            num_classes: Optional[int] = None,
            ignore_index: int = 255,
            download: bool = True,
            cityscapes_mode: str = "fine",
            max_train_samples: Optional[int] = None,
            max_val_samples: Optional[int] = None,
            max_test_samples: Optional[int] = None,
            subset_seed: int = 42,
            ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name.lower()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.num_classes = num_classes or self._infer_num_classes(self.dataset_name)
        self.ignore_index = ignore_index
        self.download = download
        self.cityscapes_mode = str(cityscapes_mode)
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        self.subset_seed = subset_seed

        self.train_transform = SegmentationTransform(image_size=image_size, train=True)
        self.eval_transform = SegmentationTransform(image_size=image_size, train=False)

    @staticmethod
    def _infer_num_classes(dataset_name: str) -> int:
        if dataset_name in {"voc", "voc2012", "pascal_voc", "pascal_voc2012"}:
            return 21
        if dataset_name in {"cityscapes", "cityscapes_fine"}:
            return 19
        raise ValueError(f"Unknown segmentation dataset: {dataset_name}")

    def prepare_data(self):
        if self._is_voc:
            torchvision.datasets.VOCSegmentation(
                self.data_dir,
                year="2012",
                image_set="train",
                download=self.download,
            )
            torchvision.datasets.VOCSegmentation(
                self.data_dir,
                year="2012",
                image_set="val",
                download=self.download,
            )
        elif self._is_cityscapes and self.download:
            raise ValueError(
                "Cityscapes cannot be downloaded automatically. Download the fine annotations "
                "and leftImg8bit files manually, then place them under data_dir/cityscapes."
            )

    @property
    def _is_voc(self) -> bool:
        return self.dataset_name in {"voc", "voc2012", "pascal_voc", "pascal_voc2012"}

    @property
    def _is_cityscapes(self) -> bool:
        return self.dataset_name in {"cityscapes", "cityscapes_fine"}

    def setup(self, stage=None):
        if self._is_voc:
            train_dataset = self._build_voc_dataset("train", self.train_transform)
            val_dataset = self._build_voc_dataset("val", self.eval_transform)
            test_dataset = self._build_voc_dataset("val", self.eval_transform)
        elif self._is_cityscapes:
            train_dataset = self._build_cityscapes_dataset("train", self.train_transform)
            val_dataset = self._build_cityscapes_dataset("val", self.eval_transform)
            test_dataset = self._build_cityscapes_dataset("val", self.eval_transform)
        else:
            raise ValueError(f"Unknown segmentation dataset: {self.dataset_name}")

        if stage == "fit" or stage is None:
            self.train_data = self._subset(train_dataset, self.max_train_samples, "train")
            self.val_data = self._subset(val_dataset, self.max_val_samples, "val")
        if stage == "test" or stage is None:
            self.test_data = self._subset(test_dataset, self.max_test_samples, "test")

    def _build_voc_dataset(self, split: str, transform: SegmentationTransform) -> Dataset:
        dataset = torchvision.datasets.VOCSegmentation(
            self.data_dir,
            year="2012",
            image_set=split,
            download=False,
        )
        return VOCSegmentationWrapper(dataset, transform)

    def _build_cityscapes_dataset(self, split: str, transform: SegmentationTransform) -> Dataset:
        dataset = torchvision.datasets.Cityscapes(
            root=os.path.join(self.data_dir, "cityscapes"),
            split=split,
            mode=self.cityscapes_mode,
            target_type="semantic",
        )
        return CityscapesSegmentationWrapper(
            dataset,
            transform,
            ignore_index=self.ignore_index,
        )

    def _subset(self, dataset: Dataset, max_samples: Optional[int], split_name: str) -> Dataset:
        if max_samples is None:
            return dataset

        max_samples = int(max_samples)
        if max_samples <= 0 or max_samples >= len(dataset):
            return dataset

        split_offsets = {"train": 0, "val": 1, "test": 2}
        generator = torch.Generator().manual_seed(self.subset_seed + split_offsets.get(split_name, 0))
        # Use a seeded random permutation so max_*_samples is not biased toward the first files.
        indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
        return Subset(dataset, indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
