import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm

import wandb
import argparse
import time
from models.vit_wrapper import CompareViT


def get_dataloaders(dataset_name, batch_size):
    """Fetches CIFAR100 or ImageNet."""
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    if dataset_name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 100
    elif dataset_name == "imagenet":
        # Requires manual download of ImageNet to ./data/imagenet
        trainset = torchvision.datasets.ImageNet(root='./data/imagenet', split='train', transform=transform_train)
        testset = torchvision.datasets.ImageNet(root='./data/imagenet', split='val', transform=transform_test)
        num_classes = 1000
    else:
        raise ValueError("Unsupported dataset")

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

    return trainloader, testloader, num_classes


def train(args):
    # 1. Initialize Weights & Biases
    wandb.init(project="laplacian-vs-vanilla", config=args)
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Setup Data
    trainloader, testloader, num_classes = get_dataloaders(config.dataset, config.batch_size)

    # 3. Initialize Model
    model = CompareViT(
        num_classes=num_classes,
        attn_type=config.attn_type,
        dim=384,  # Small ViT dimensions
        depth=6,
        num_heads=6
    ).to(device)

    # Optional: Log model gradients to wandb
    wandb.watch(model, log="all", log_freq=100)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    print(f"Starting training for {config.attn_type} attention on {config.dataset}...")

    # 4. Training Loop
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 50 == 49:
                step_time = (time.time() - start_time) / 50
                wandb.log(
                    {
                        "train_loss": running_loss / 50,
                        "step_time_sec": step_time,
                        "epoch": epoch
                    }
                )
                running_loss = 0.0
                start_time = time.time()

        # 5. Validation Loop
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        wandb.log(
            {
                "val_loss": val_loss / len(testloader),
                "val_accuracy": val_acc,
                "epoch": epoch
            }
        )
        print(f"Epoch {epoch + 1} | Val Acc: {val_acc:.2f}%")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn_type", type=str, default="laplacian", choices=["vanilla", "laplacian"])
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar100", "imagenet"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=30)

    args = parser.parse_args()
    train(args)