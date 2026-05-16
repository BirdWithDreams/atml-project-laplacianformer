from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def safe_name(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name[:120]


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if image.mode != "RGB":
        image = image.convert("RGB")

    image.save(path, format="JPEG", quality=95)


def select_class_ids(num_classes: int, total_classes: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return sorted(rng.sample(range(total_classes), num_classes))


def download_train_split(
    *,
    dataset_name: str,
    out_dir: Path,
    selected_class_ids: list[int],
    class_names: list[str],
    images_per_class: int,
    seed: int,
    shuffle_buffer: int,
) -> dict:
    selected_set = set(selected_class_ids)
    selected_folders = {
        class_id: safe_name(class_names[class_id])
        for class_id in selected_class_ids
    }

    ds = load_dataset(
        dataset_name,
        split="train",
        streaming=True,
        token=True,
    )

    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    counts = defaultdict(int)
    total_needed = len(selected_class_ids) * images_per_class
    total_saved = 0

    progress = tqdm(total=total_needed, desc="Saving train")

    for example in ds:
        label = int(example["label"])

        if label not in selected_set:
            continue

        if counts[label] >= images_per_class:
            continue

        class_folder = selected_folders[label]
        image_idx = counts[label]

        save_path = out_dir / "train" / class_folder / f"{image_idx:05d}.jpg"
        save_image(example["image"], save_path)

        counts[label] += 1
        total_saved += 1
        progress.update(1)

        if total_saved >= total_needed:
            break

    progress.close()

    incomplete = {
        selected_folders[class_id]: counts[class_id]
        for class_id in selected_class_ids
        if counts[class_id] < images_per_class
    }

    return {
        "source_split": "train",
        "output_split": "train",
        "requested_images_per_class": images_per_class,
        "saved_total": total_saved,
        "saved_per_class": {
            selected_folders[class_id]: counts[class_id]
            for class_id in selected_class_ids
        },
        "incomplete_classes": incomplete,
    }


def split_validation_into_val_and_test(
    *,
    dataset_name: str,
    out_dir: Path,
    selected_class_ids: list[int],
    class_names: list[str],
    val_images_per_class: int,
    test_images_per_class: int,
    seed: int,
    shuffle_buffer: int,
) -> dict:
    """
    Uses the official ImageNet validation split as the source.

    For each selected class:
      - first val_images_per_class images go to output validation/
      - next test_images_per_class images go to output test/

    Because we stream each image only once, validation and test are disjoint.
    """

    selected_set = set(selected_class_ids)
    selected_folders = {
        class_id: safe_name(class_names[class_id])
        for class_id in selected_class_ids
    }

    ds = load_dataset(
        dataset_name,
        split="validation",
        streaming=True,
        token=True,
    )

    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    val_counts = defaultdict(int)
    test_counts = defaultdict(int)

    total_needed = len(selected_class_ids) * (
        val_images_per_class + test_images_per_class
    )
    total_saved = 0

    progress = tqdm(total=total_needed, desc="Splitting validation into val/test")

    for example in ds:
        label = int(example["label"])

        if label not in selected_set:
            continue

        class_folder = selected_folders[label]

        if val_counts[label] < val_images_per_class:
            image_idx = val_counts[label]
            save_path = out_dir / "validation" / class_folder / f"{image_idx:05d}.jpg"

            save_image(example["image"], save_path)

            val_counts[label] += 1
            total_saved += 1
            progress.update(1)

        elif test_counts[label] < test_images_per_class:
            image_idx = test_counts[label]
            save_path = out_dir / "test" / class_folder / f"{image_idx:05d}.jpg"

            save_image(example["image"], save_path)

            test_counts[label] += 1
            total_saved += 1
            progress.update(1)

        if total_saved >= total_needed:
            break

    progress.close()

    incomplete_validation = {
        selected_folders[class_id]: val_counts[class_id]
        for class_id in selected_class_ids
        if val_counts[class_id] < val_images_per_class
    }

    incomplete_test = {
        selected_folders[class_id]: test_counts[class_id]
        for class_id in selected_class_ids
        if test_counts[class_id] < test_images_per_class
    }

    return {
        "source_split": "validation",
        "output_splits": {
            "validation": {
                "requested_images_per_class": val_images_per_class,
                "saved_total": sum(val_counts.values()),
                "saved_per_class": {
                    selected_folders[class_id]: val_counts[class_id]
                    for class_id in selected_class_ids
                },
                "incomplete_classes": incomplete_validation,
            },
            "test": {
                "requested_images_per_class": test_images_per_class,
                "saved_total": sum(test_counts.values()),
                "saved_per_class": {
                    selected_folders[class_id]: test_counts[class_id]
                    for class_id in selected_class_ids
                },
                "incomplete_classes": incomplete_test,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="ILSVRC/imagenet-1k",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="../data/imagenet_subset",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--train-images-per-class",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--val-images-per-class",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--test-images-per-class",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=20_000,
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
    )
    parser.add_argument(
        "--skip-val-test",
        action="store_true",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.val_images_per_class + args.test_images_per_class > 50:
        raise ValueError(
            "ImageNet validation has only 50 images per class. "
            "The sum of --val-images-per-class and --test-images-per-class "
            "should be <= 50."
        )

    print("Loading dataset metadata...")

    meta_ds = load_dataset(
        args.dataset,
        split="train",
        streaming=True,
        token=True,
    )

    class_names = meta_ds.features["label"].names
    total_classes = len(class_names)

    if args.num_classes > total_classes:
        raise ValueError(
            f"Requested {args.num_classes} classes, but dataset has only {total_classes}."
        )

    selected_class_ids = select_class_ids(
        num_classes=args.num_classes,
        total_classes=total_classes,
        seed=args.seed,
    )

    selected_classes = [
        {
            "class_id": class_id,
            "class_name": class_names[class_id],
            "folder": safe_name(class_names[class_id]),
        }
        for class_id in selected_class_ids
    ]

    results = {
        "dataset": args.dataset,
        "num_classes": args.num_classes,
        "seed": args.seed,
        "selected_classes": selected_classes,
        "splits": {},
    }

    if not args.skip_train:
        train_result = download_train_split(
            dataset_name=args.dataset,
            out_dir=out_dir,
            selected_class_ids=selected_class_ids,
            class_names=class_names,
            images_per_class=args.train_images_per_class,
            seed=args.seed,
            shuffle_buffer=args.shuffle_buffer,
        )
        results["splits"]["train"] = train_result

    if not args.skip_val_test:
        val_test_result = split_validation_into_val_and_test(
            dataset_name=args.dataset,
            out_dir=out_dir,
            selected_class_ids=selected_class_ids,
            class_names=class_names,
            val_images_per_class=args.val_images_per_class,
            test_images_per_class=args.test_images_per_class,
            seed=args.seed,
            shuffle_buffer=args.shuffle_buffer,
        )
        results["splits"]["validation_test_from_validation"] = val_test_result

    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Done. Saved subset to: {out_dir}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()