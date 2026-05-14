import lightning as L
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os


def _split_dir(root_dir: str, split_names: tuple[str, ...]) -> str:
    for split_name in split_names:
        candidate = os.path.join(root_dir, split_name)
        if os.path.isdir(candidate):
            return candidate

    expected = ", ".join(os.path.join(root_dir, split_name) for split_name in split_names)
    raise FileNotFoundError(f"Could not find dataset split directory. Expected one of: {expected}")


class CVDataModule(L.LightningDataModule):
    def __init__(
            self, data_dir: str = "./data", dataset_name: str = "cifar100", batch_size: int = 128,
            num_workers: int = 4, image_size: int = 224
            ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name.lower()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.transform_test = transforms.Compose([
            transforms.Resize(int(self.image_size * 256 / 224)),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def prepare_data(self):
        if self.dataset_name == "cifar100":
            torchvision.datasets.CIFAR100(self.data_dir, train=True, download=True)
            torchvision.datasets.CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Inicialización de Datasets basada en el nombre
        if self.dataset_name == "cifar100":
            train_dataset = torchvision.datasets.CIFAR100(self.data_dir, train=True, transform=self.transform_train)
            val_dataset = torchvision.datasets.CIFAR100(self.data_dir, train=False, transform=self.transform_test)
            test_dataset = torchvision.datasets.CIFAR100(self.data_dir, train=False, transform=self.transform_test)
            
        elif self.dataset_name == "imagenet":
            imagenet_root = os.path.join(self.data_dir, "imagenet")
            if not os.path.isdir(imagenet_root) and os.path.isdir(os.path.join(self.data_dir, "train")):
                imagenet_root = self.data_dir

            train_dir = _split_dir(imagenet_root, ("train",))
            val_dir = _split_dir(imagenet_root, ("val", "validation"))
            train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=self.transform_train)
            val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=self.transform_test)
            test_dataset = val_dataset  # Típicamente en ImageNet usamos val para test
            
        else:
            # Opción por defecto (Genérica) para cualquier dataset organizado en carpetas
            dataset_root = os.path.join(self.data_dir, self.dataset_name)
            train_dir = _split_dir(dataset_root, ("train",))
            val_dir = _split_dir(dataset_root, ("val", "validation"))
            train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=self.transform_train)
            val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=self.transform_test)
            test_dataset = val_dataset

        # Asignación para los DataLoaders
        if stage == "fit" or stage is None:
            self.train_data = train_dataset
            self.val_data = val_dataset
        if stage == "test":
            self.test_data = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
