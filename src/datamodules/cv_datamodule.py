import lightning as L
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class CVDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data", dataset_name: str = "cifar100", batch_size: int = 128, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def prepare_data(self):
        if self.dataset_name == "cifar100":
            torchvision.datasets.CIFAR100(self.data_dir, train=True, download=True)
            torchvision.datasets.CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if self.dataset_name == "cifar100":
            if stage == "fit" or stage is None:
                self.cifar_train = torchvision.datasets.CIFAR100(self.data_dir, train=True, transform=self.transform_train)
                self.cifar_val = torchvision.datasets.CIFAR100(self.data_dir, train=False, transform=self.transform_test)
            if stage == "test":
                self.cifar_test = torchvision.datasets.CIFAR100(self.data_dir, train=False, transform=self.transform_test)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
