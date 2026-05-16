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
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SegmentationTransform:
    def __init__(
            self,
            image_size: int = 224,
            train: bool = False,
            min_scale: float = 0.5,
            max_scale: float = 2.0,
            ignore_index: int = 255,
            mean: tuple[float, float, float] = IMAGENET_MEAN,
            std: tuple[float, float, float] = IMAGENET_STD,
            ):
        self.image_size = image_size
        self.train = train
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.ignore_index = int(ignore_index)
        self.mean = mean
        self.std = std

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.convert("RGB")

        if self.train:
            short_size = max(1, int(round(self.image_size * random.uniform(self.min_scale, self.max_scale))))
            image, mask = self._resize_short_side(image, mask, short_size)
            image, mask = self._pad_to_crop_size(image, mask)
            image, mask = self._random_crop(image, mask)
        else:
            image, mask = self._resize_long_side(image, mask, self.image_size)
            image, mask = self._pad_to_crop_size(image, mask, center=True)
            image, mask = self._center_crop(image, mask)

        if self.train and random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image = TF.normalize(TF.to_tensor(image), self.mean, self.std)
        mask_tensor = torch.as_tensor(np.array(mask, dtype=np.int64), dtype=torch.long)
        return image, mask_tensor

    def _resize_short_side(
            self, image: Image.Image, mask: Image.Image, short_size: int
            ) -> tuple[Image.Image, Image.Image]:
        width, height = image.size
        scale = float(short_size) / float(min(height, width))
        resized_height = max(1, int(round(height * scale)))
        resized_width = max(1, int(round(width * scale)))
        size = [resized_height, resized_width]
        image = TF.resize(image, size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)
        return image, mask

    def _resize_long_side(
            self, image: Image.Image, mask: Image.Image, long_size: int
            ) -> tuple[Image.Image, Image.Image]:
        width, height = image.size
        scale = float(long_size) / float(max(height, width))
        resized_height = max(1, int(round(height * scale)))
        resized_width = max(1, int(round(width * scale)))
        size = [resized_height, resized_width]
        image = TF.resize(image, size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)
        return image, mask

    def _pad_to_crop_size(
            self, image: Image.Image, mask: Image.Image, center: bool = False
            ) -> tuple[Image.Image, Image.Image]:
        pad_width = max(self.image_size - image.size[0], 0)
        pad_height = max(self.image_size - image.size[1], 0)
        if pad_width == 0 and pad_height == 0:
            return image, mask

        if center:
            left = pad_width // 2
            top = pad_height // 2
            padding = [left, top, pad_width - left, pad_height - top]
        else:
            padding = [0, 0, pad_width, pad_height]
        image = TF.pad(image, padding, fill=0)
        mask = TF.pad(mask, padding, fill=self.ignore_index)
        return image, mask

    def _random_crop(
            self, image: Image.Image, mask: Image.Image
            ) -> tuple[Image.Image, Image.Image]:
        width, height = image.size
        top = random.randint(0, height - self.image_size)
        left = random.randint(0, width - self.image_size)
        return self._crop(image, mask, top, left)

    def _center_crop(
            self, image: Image.Image, mask: Image.Image
            ) -> tuple[Image.Image, Image.Image]:
        width, height = image.size
        top = max((height - self.image_size) // 2, 0)
        left = max((width - self.image_size) // 2, 0)
        return self._crop(image, mask, top, left)

    def _crop(
            self, image: Image.Image, mask: Image.Image, top: int, left: int
            ) -> tuple[Image.Image, Image.Image]:
        image = TF.crop(image, top, left, self.image_size, self.image_size)
        mask = TF.crop(mask, top, left, self.image_size, self.image_size)
        return image, mask


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


class StanfordBackgroundSegmentation(Dataset):
    url = "http://dags.stanford.edu/data/iccv09Data.tar.gz"
    filename = "iccv09Data.tar.gz"

    def __init__(
            self,
            root: str,
            split: str,
            transform: SegmentationTransform,
            val_fraction: float = 0.1,
            test_fraction: float = 0.1,
            subset_seed: int = 42,
            download: bool = False,
            ):
        self.root = root
        self.split = split
        self.transform = transform
        if download:
            self.download()

        self.data_root = self._find_data_root(root)
        self.image_dir = os.path.join(self.data_root, "images")
        self.label_dir = os.path.join(self.data_root, "labels")
        self.samples = self._split_samples(
            self._discover_samples(),
            split=split,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            subset_seed=subset_seed,
        )

    def download(self):
        if self._has_data(self.root):
            return

        os.makedirs(self.root, exist_ok=True)
        download_and_extract_archive(
            self.url,
            download_root=self.root,
            filename=self.filename,
        )

    @staticmethod
    def _has_data(root: str) -> bool:
        candidate_roots = [root, os.path.join(root, "iccv09Data")]
        return any(
            os.path.isdir(os.path.join(candidate_root, "images"))
            and os.path.isdir(os.path.join(candidate_root, "labels"))
            for candidate_root in candidate_roots
        )

    @classmethod
    def _find_data_root(cls, root: str) -> str:
        candidate_roots = [root, os.path.join(root, "iccv09Data")]
        for candidate_root in candidate_roots:
            if (
                os.path.isdir(os.path.join(candidate_root, "images"))
                and os.path.isdir(os.path.join(candidate_root, "labels"))
            ):
                return candidate_root

        raise FileNotFoundError(
            "Could not find Stanford Background data. Expected images/ and labels/ "
            f"under {root!r} or {os.path.join(root, 'iccv09Data')!r}. "
            "Set download=true to fetch iccv09Data.tar.gz."
        )

    def _discover_samples(self) -> list[tuple[str, str]]:
        image_paths = []
        for filename in sorted(os.listdir(self.image_dir)):
            stem, extension = os.path.splitext(filename)
            if extension.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            label_path = self._find_label_path(stem)
            if label_path is not None:
                image_paths.append((os.path.join(self.image_dir, filename), label_path))

        if not image_paths:
            raise FileNotFoundError(
                "No Stanford Background image/label pairs found under "
                f"{self.image_dir!r} and {self.label_dir!r}."
            )

        return image_paths

    def _find_label_path(self, stem: str) -> Optional[str]:
        for suffix in (".regions.txt", ".png", ".jpg", ".jpeg"):
            candidate = os.path.join(self.label_dir, f"{stem}{suffix}")
            if os.path.exists(candidate):
                return candidate
        return None

    @staticmethod
    def _split_samples(
            samples: list[tuple[str, str]],
            split: str,
            val_fraction: float,
            test_fraction: float,
            subset_seed: int,
            ) -> list[tuple[str, str]]:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown Stanford Background split: {split!r}")

        if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
            raise ValueError(
                "Stanford Background split fractions must be non-negative and sum to < 1; "
                f"got val_fraction={val_fraction}, test_fraction={test_fraction}."
            )

        rng = random.Random(subset_seed)
        shuffled = list(samples)
        rng.shuffle(shuffled)

        total = len(shuffled)
        test_count = int(round(total * test_fraction))
        val_count = int(round(total * val_fraction))
        train_end = total - val_count - test_count
        val_end = train_end + val_count

        if split == "train":
            return shuffled[:train_end]
        if split == "val":
            return shuffled[train_end:val_end]
        return shuffled[val_end:]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label_path = self.samples[index]
        image = Image.open(image_path)
        mask = self._load_mask(label_path)
        return self.transform(image, mask)

    @staticmethod
    def _load_mask(label_path: str) -> Image.Image:
        if label_path.endswith(".regions.txt"):
            mask_array = np.loadtxt(label_path, dtype=np.int64)
            return Image.fromarray(mask_array.astype(np.uint8))

        mask = Image.open(label_path)
        mask_array = np.array(mask)
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]
        return Image.fromarray(mask_array.astype(np.uint8))


class SegmentationDataModule(L.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "./data",
            dataset_name: str = "voc2012",
            batch_size: int = 16,
            num_workers: int = 4,
            image_size: int = 224,
            train_min_scale: float = 0.5,
            train_max_scale: float = 2.0,
            num_classes: Optional[int] = None,
            ignore_index: int = 255,
            download: bool = True,
            cityscapes_mode: str = "fine",
            val_fraction: float = 0.1,
            test_fraction: float = 0.1,
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
        self.train_min_scale = train_min_scale
        self.train_max_scale = train_max_scale
        self.num_classes = num_classes or self._infer_num_classes(self.dataset_name)
        self.ignore_index = ignore_index
        self.download = download
        self.cityscapes_mode = str(cityscapes_mode)
        self.val_fraction = float(val_fraction)
        self.test_fraction = float(test_fraction)
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        self.subset_seed = subset_seed

        self.train_transform = SegmentationTransform(
            image_size=image_size,
            train=True,
            min_scale=train_min_scale,
            max_scale=train_max_scale,
            ignore_index=ignore_index,
        )
        self.eval_transform = SegmentationTransform(
            image_size=image_size,
            train=False,
            ignore_index=ignore_index,
        )

    @staticmethod
    def _infer_num_classes(dataset_name: str) -> int:
        if dataset_name in {"voc", "voc2012", "pascal_voc", "pascal_voc2012"}:
            return 21
        if dataset_name in {"cityscapes", "cityscapes_fine"}:
            return 19
        if dataset_name in {"stanford_background", "stanford_background_segmentation"}:
            return 9
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
                "torchvision.datasets.Cityscapes does not support automatic download. "
                "Download the fine annotations and leftImg8bit files manually, then place "
                "them under data_dir/cityscapes."
            )
        elif self._is_cityscapes:
            for split in ("train", "val"):
                torchvision.datasets.Cityscapes(
                    root=os.path.join(self.data_dir, "cityscapes"),
                    split=split,
                    mode=self.cityscapes_mode or "fine",
                    target_type="semantic",
                )
        elif self._is_stanford_background:
            StanfordBackgroundSegmentation(
                self.data_dir,
                split="train",
                transform=self.train_transform,
                val_fraction=self.val_fraction,
                test_fraction=self.test_fraction,
                subset_seed=self.subset_seed,
                download=self.download,
            )

    @property
    def _is_voc(self) -> bool:
        return self.dataset_name in {"voc", "voc2012", "pascal_voc", "pascal_voc2012"}

    @property
    def _is_cityscapes(self) -> bool:
        return self.dataset_name in {"cityscapes", "cityscapes_fine"}

    @property
    def _is_stanford_background(self) -> bool:
        return self.dataset_name in {"stanford_background", "stanford_background_segmentation"}

    def setup(self, stage=None):
        if self._is_voc:
            train_dataset = self._build_voc_dataset("train", self.train_transform)
            val_dataset = self._build_voc_dataset("val", self.eval_transform)
            test_dataset = self._build_voc_dataset("val", self.eval_transform)
        elif self._is_cityscapes:
            train_dataset = self._build_cityscapes_dataset("train", self.train_transform)
            val_dataset = self._build_cityscapes_dataset("val", self.eval_transform)
            test_dataset = self._build_cityscapes_dataset("val", self.eval_transform)
        elif self._is_stanford_background:
            train_dataset = self._build_stanford_background_dataset("train", self.train_transform)
            val_dataset = self._build_stanford_background_dataset("val", self.eval_transform)
            test_dataset = self._build_stanford_background_dataset("test", self.eval_transform)
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
            mode=self.cityscapes_mode or "fine",
            target_type="semantic",
        )
        return CityscapesSegmentationWrapper(
            dataset,
            transform,
            ignore_index=self.ignore_index,
        )

    def _build_stanford_background_dataset(
            self, split: str, transform: SegmentationTransform
            ) -> Dataset:
        return StanfordBackgroundSegmentation(
            self.data_dir,
            split=split,
            transform=transform,
            val_fraction=self.val_fraction,
            test_fraction=self.test_fraction,
            subset_seed=self.subset_seed,
            download=False,
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
