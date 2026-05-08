from __future__ import annotations

import csv
import glob
import json
import os
import resource
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = PROJECT_ROOT / "configs"


def resolve_path(path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_named_config(group: str, name: str) -> DictConfig:
    path = CONFIG_ROOT / group / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Unknown {group} config '{name}': {path}")
    return OmegaConf.load(path)


def to_plain_container(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def expand_checkpoint_paths(run_cfg: DictConfig) -> list[Path]:
    if run_cfg.get("checkpoint_glob") is not None:
        pattern = str(run_cfg.checkpoint_glob)
        if not Path(pattern).is_absolute():
            pattern = str(PROJECT_ROOT / pattern)
        paths = [Path(path) for path in sorted(glob.glob(pattern))]
    elif run_cfg.get("checkpoint_paths") is not None:
        paths = [resolve_path(path) for path in run_cfg.checkpoint_paths]
    elif run_cfg.get("checkpoint_path") is not None:
        paths = [resolve_path(run_cfg.checkpoint_path)]
    else:
        raise ValueError(
            f"Benchmark run '{run_cfg.get('name', '<unnamed>')}' must define "
            "checkpoint_path, checkpoint_paths, or checkpoint_glob."
        )

    if len(paths) == 0:
        raise FileNotFoundError(
            f"Benchmark run '{run_cfg.get('name', '<unnamed>')}' did not match any checkpoints."
        )
    return paths


def expand_runs(cfg: DictConfig) -> list[DictConfig]:
    expanded = []
    for run_cfg in cfg.get("runs", []):
        paths = expand_checkpoint_paths(run_cfg)
        for index, checkpoint_path in enumerate(paths):
            run = OmegaConf.create(to_plain_container(run_cfg))
            run.checkpoint_path = str(checkpoint_path)
            if len(paths) > 1:
                base_name = str(run_cfg.get("name", run_cfg.task))
                run.name = f"{base_name}_{index:03d}_{checkpoint_path.stem}"
            else:
                run.name = str(run_cfg.get("name", checkpoint_path.stem))
            expanded.append(run)
    return expanded


def load_datamodule_config(run_cfg: DictConfig) -> DictConfig:
    if run_cfg.get("datamodule_config") is not None:
        datamodule_cfg = OmegaConf.create(run_cfg.datamodule_config)
    elif run_cfg.get("datamodule") is not None:
        datamodule_cfg = load_named_config("datamodule", str(run_cfg.datamodule))
    else:
        raise ValueError(f"Run '{run_cfg.name}' must define datamodule or datamodule_config.")

    if run_cfg.get("datamodule_overrides") is not None:
        datamodule_cfg = OmegaConf.merge(datamodule_cfg, run_cfg.datamodule_overrides)
    return datamodule_cfg


def checkpoint_hparams(checkpoint_path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return dict(checkpoint.get("hyper_parameters", {}))


def build_datamodule(task_name: str, datamodule_cfg: DictConfig, hparams: dict[str, Any]):
    if task_name == "cv_classification":
        from src.datamodules.cv_datamodule import CVDataModule

        model_cfg = hparams.get("model_cfg", {}) or {}
        return CVDataModule(
            data_dir=datamodule_cfg.get("data_dir", "./data"),
            dataset_name=datamodule_cfg.dataset_name,
            batch_size=datamodule_cfg.batch_size,
            num_workers=datamodule_cfg.num_workers,
            image_size=model_cfg.get("img_size", datamodule_cfg.get("image_size", 224)),
        )

    if task_name == "nlp_classification":
        from src.datamodules.nlp_datamodule import NLPDataModule

        return NLPDataModule(
            model_name=datamodule_cfg.get("model_name", "bert-base-uncased"),
            dataset_name=datamodule_cfg.dataset_name,
            batch_size=datamodule_cfg.batch_size,
            num_workers=datamodule_cfg.num_workers,
            max_length=datamodule_cfg.max_length,
            dataset_path=datamodule_cfg.get("dataset_path", None),
            dataset_config_name=datamodule_cfg.get("dataset_config_name", None),
            text_column=datamodule_cfg.get("text_column", None),
            text_pair_column=datamodule_cfg.get("text_pair_column", None),
            label_column=datamodule_cfg.get("label_column", "label"),
            train_split=datamodule_cfg.get("train_split", "train"),
            validation_split=datamodule_cfg.get("validation_split", None),
            test_split=datamodule_cfg.get("test_split", "test"),
            validation_size=datamodule_cfg.get("validation_size", 0.1),
            subset_seed=datamodule_cfg.get("subset_seed", 42),
        )

    if task_name == "ner_task":
        from src.datamodules.ner_datamodule import NERDataModule

        return NERDataModule(
            dataset_name=datamodule_cfg.dataset_name,
            batch_size=datamodule_cfg.batch_size,
            num_workers=datamodule_cfg.num_workers,
            max_length=datamodule_cfg.max_length,
            label_column=datamodule_cfg.get("label_column", None),
            min_freq=datamodule_cfg.get("min_freq", 1),
            unk_replace_prob=0.0,
        )

    if task_name == "semantic_segmentation":
        from src.datamodules.segmentation_datamodule import SegmentationDataModule

        model_cfg = hparams.get("model_cfg", {}) or {}
        return SegmentationDataModule(
            data_dir=datamodule_cfg.get("data_dir", "./data"),
            dataset_name=datamodule_cfg.dataset_name,
            batch_size=datamodule_cfg.batch_size,
            num_workers=datamodule_cfg.num_workers,
            image_size=model_cfg.get("img_size", datamodule_cfg.get("image_size", 224)),
            num_classes=datamodule_cfg.num_classes,
            ignore_index=datamodule_cfg.get("ignore_index", 255),
            download=False,
            coco_year=datamodule_cfg.get("coco_year", "2017"),
            max_test_samples=datamodule_cfg.get("max_test_samples", None),
            subset_seed=datamodule_cfg.get("subset_seed", 42),
        )

    raise ValueError(f"Unsupported task for benchmarking: {task_name}")


def load_task(task_name: str, checkpoint_path: Path):
    if task_name == "cv_classification":
        from src.tasks.classification_cv import CVClassificationTask

        return CVClassificationTask.load_from_checkpoint(str(checkpoint_path), map_location="cpu")

    if task_name == "nlp_classification":
        from src.tasks.classification_nlp import NLPClassificationTask

        return NLPClassificationTask.load_from_checkpoint(str(checkpoint_path), map_location="cpu")

    if task_name == "ner_task":
        from src.tasks.ner_task import NERTask

        return NERTask.load_from_checkpoint(str(checkpoint_path), map_location="cpu")

    if task_name == "semantic_segmentation":
        from src.tasks.segmentation import SemanticSegmentationTask

        return SemanticSegmentationTask.load_from_checkpoint(
            str(checkpoint_path),
            map_location="cpu",
            log_segmentation_images=False,
        )

    raise ValueError(f"Unsupported task for benchmarking: {task_name}")


def setup_test_dataloader(datamodule, max_samples: int | None, subset_seed: int, subset_shuffle: bool):
    datamodule.prepare_data()
    datamodule.setup("test")
    if not getattr(datamodule, "has_test_labels", True):
        raise ValueError("Configured test split has no public labels.")

    test_dataset = getattr(datamodule, "test_dataset", None)
    test_data = getattr(datamodule, "test_data", None)
    if test_dataset is None and test_data is None:
        raise ValueError("Datamodule did not create a test dataset.")

    dataloader = datamodule.test_dataloader()
    if max_samples is None:
        return dataloader

    dataset = dataloader.dataset
    max_samples = int(max_samples)
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataloader

    if subset_shuffle:
        generator = torch.Generator().manual_seed(int(subset_seed))
        indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    else:
        indices = list(range(max_samples))

    subset = Subset(dataset, indices)
    loader_kwargs = {
        "batch_size": dataloader.batch_size,
        "shuffle": False,
        "num_workers": dataloader.num_workers,
        "pin_memory": dataloader.pin_memory,
        "drop_last": dataloader.drop_last,
        "collate_fn": dataloader.collate_fn,
    }
    if dataloader.num_workers > 0:
        loader_kwargs["persistent_workers"] = getattr(dataloader, "persistent_workers", False)
    return DataLoader(subset, **loader_kwargs)


def select_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def maybe_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device: torch.device, precision: Any):
    precision = str(precision).lower()
    if device.type != "cuda":
        return nullcontext()
    if precision in {"16", "16-mixed", "fp16", "float16"}:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision in {"bf16", "bf16-mixed", "bfloat16"}:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def batch_size(batch: Any) -> int:
    if torch.is_tensor(batch):
        return int(batch.shape[0]) if batch.ndim > 0 else 1
    if isinstance(batch, dict):
        for value in batch.values():
            if torch.is_tensor(value):
                return batch_size(value)
    if isinstance(batch, (tuple, list)):
        for value in batch:
            if torch.is_tensor(value):
                return batch_size(value)
    raise ValueError("Could not infer batch size.")


def tensor_to_python(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: tensor_to_python(item) for key, item in value.items()}
    return value


def current_rss_mb() -> float | None:
    statm = Path("/proc/self/statm")
    if statm.is_file():
        pages = int(statm.read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 ** 2)
    return None


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return peak / (1024 ** 2)
    return peak / 1024


class RunningLoss:
    def __init__(self):
        self.total = 0.0
        self.weight = 0

    def update(self, loss: torch.Tensor, weight: int):
        self.total += float(loss.detach().cpu().item()) * int(weight)
        self.weight += int(weight)

    def compute(self) -> float:
        if self.weight == 0:
            return 0.0
        return self.total / self.weight


def classification_num_classes(model, datamodule_cfg: DictConfig) -> int:
    value = getattr(getattr(model, "hparams", {}), "num_classes", None)
    if value is None:
        value = datamodule_cfg.get("num_classes")
    if value is None:
        raise ValueError("Could not infer num_classes for classification metrics.")
    return int(value)


def classification_metric_bundle(num_classes: int, device: torch.device) -> dict[str, Any]:
    return {
        "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro").to(device),
        "precision_macro": MulticlassPrecision(
            num_classes=num_classes, average="macro", zero_division=0
        ).to(device),
        "recall_macro": MulticlassRecall(
            num_classes=num_classes, average="macro", zero_division=0
        ).to(device),
        "f1_macro": MulticlassF1Score(
            num_classes=num_classes, average="macro", zero_division=0
        ).to(device),
    }


def init_metric_state(task_name: str, model, datamodule_cfg: DictConfig, device: torch.device):
    if task_name in {"cv_classification", "nlp_classification"}:
        return {
            "loss": RunningLoss(),
            "metrics": classification_metric_bundle(
                classification_num_classes(model, datamodule_cfg), device
            ),
        }

    if task_name == "ner_task":
        from src.tasks.ner_task import SeqevalMetric

        id2label = getattr(model, "id2label", None)
        if id2label is None:
            raise ValueError("NER checkpoint does not expose id2label.")
        return {
            "loss": RunningLoss(),
            "seqeval": SeqevalMetric(id2label=id2label).to(device),
            "token_correct": 0,
            "token_total": 0,
        }

    if task_name == "semantic_segmentation":
        from src.tasks.segmentation import SegmentationMetrics

        num_classes = int(datamodule_cfg.num_classes)
        ignore_index = int(datamodule_cfg.get("ignore_index", getattr(model.hparams, "ignore_index", 255)))
        return {
            "loss": RunningLoss(),
            "metrics": SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index).to(device),
        }

    raise ValueError(f"Unsupported task: {task_name}")


def forward_for_task(task_name: str, model, batch: Any):
    if task_name == "cv_classification":
        images, _ = batch
        return model(images)
    if task_name == "nlp_classification":
        return model(batch["input_ids"], batch["attention_mask"])
    if task_name == "ner_task":
        return model(batch["input_ids"], batch["attention_mask"])
    if task_name == "semantic_segmentation":
        images, _ = batch
        return model(images)
    raise ValueError(f"Unsupported task: {task_name}")


def update_metrics_for_task(task_name: str, model, batch: Any, logits: torch.Tensor, state: dict[str, Any]):
    current_batch_size = batch_size(batch)

    if task_name == "cv_classification":
        _, labels = batch
        loss = model.criterion(logits, labels)
        state["loss"].update(loss, current_batch_size)
        for metric in state["metrics"].values():
            metric.update(logits, labels)
        return

    if task_name == "nlp_classification":
        labels = batch["label"]
        loss = model.criterion(logits, labels)
        state["loss"].update(loss, current_batch_size)
        for metric in state["metrics"].values():
            metric.update(logits, labels)
        return

    if task_name == "ner_task":
        labels = batch["labels"]
        loss = model.criterion(logits.reshape(-1, model.hparams.num_classes), labels.reshape(-1))
        state["loss"].update(loss, current_batch_size)
        preds = torch.argmax(logits, dim=-1)
        state["seqeval"].update(preds, labels)
        valid = labels != -100
        state["token_correct"] += int((preds[valid] == labels[valid]).sum().detach().cpu().item())
        state["token_total"] += int(valid.sum().detach().cpu().item())
        return

    if task_name == "semantic_segmentation":
        _, masks = batch
        loss = model.criterion(logits, masks)
        state["loss"].update(loss, current_batch_size)
        preds = torch.argmax(logits, dim=1)
        state["metrics"].update(preds, masks)
        return

    raise ValueError(f"Unsupported task: {task_name}")


def compute_metrics(task_name: str, state: dict[str, Any]) -> dict[str, float]:
    metrics = {"loss": state["loss"].compute()}

    if task_name in {"cv_classification", "nlp_classification"}:
        metrics.update(
            {name: tensor_to_python(metric.compute()) for name, metric in state["metrics"].items()}
        )
        return metrics

    if task_name == "ner_task":
        metrics.update(tensor_to_python(state["seqeval"].compute()))
        token_total = max(int(state["token_total"]), 1)
        metrics["token_accuracy"] = float(state["token_correct"]) / token_total
        return metrics

    if task_name == "semantic_segmentation":
        metrics.update(tensor_to_python(state["metrics"].compute()))
        return metrics

    raise ValueError(f"Unsupported task: {task_name}")


def memory_start(device: torch.device) -> dict[str, float | None]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        return {
            "cuda_allocated_start_mb": torch.cuda.memory_allocated(device) / (1024 ** 2),
            "cuda_reserved_start_mb": torch.cuda.memory_reserved(device) / (1024 ** 2),
            "cpu_rss_start_mb": current_rss_mb(),
        }
    return {"cpu_rss_start_mb": current_rss_mb()}


def memory_end(device: torch.device, start: dict[str, float | None]) -> dict[str, float | None]:
    result = dict(start)
    if device.type == "cuda":
        result.update(
            {
                "cuda_allocated_end_mb": torch.cuda.memory_allocated(device) / (1024 ** 2),
                "cuda_reserved_end_mb": torch.cuda.memory_reserved(device) / (1024 ** 2),
                "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
                "cuda_peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 ** 2),
            }
        )
    result.update({"cpu_rss_end_mb": current_rss_mb(), "cpu_rss_peak_mb": peak_rss_mb()})
    return result


def run_warmup(task_name: str, model, dataloader, device: torch.device, precision: Any, num_batches: int):
    if num_batches <= 0:
        return
    model.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= num_batches:
                break
            batch = move_to_device(batch, device)
            with autocast_context(device, precision):
                _ = forward_for_task(task_name, model, batch)
            maybe_sync(device)


def benchmark_run(
        run_cfg: DictConfig,
        cfg: DictConfig,
        device: torch.device,
        ) -> dict[str, Any]:
    task_name = str(run_cfg.task)
    checkpoint_path = resolve_path(run_cfg.checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    hparams = checkpoint_hparams(checkpoint_path)
    datamodule_cfg = load_datamodule_config(run_cfg)
    max_samples = run_cfg.get("max_samples", cfg.get("max_samples", None))
    subset_seed = int(run_cfg.get("subset_seed", cfg.get("subset_seed", cfg.seed)))
    subset_shuffle = bool(run_cfg.get("subset_shuffle", cfg.get("subset_shuffle", True)))

    datamodule = build_datamodule(task_name, datamodule_cfg, hparams)
    dataloader = setup_test_dataloader(datamodule, max_samples, subset_seed, subset_shuffle)

    model = load_task(task_name, checkpoint_path)
    model.to(device)
    if bool(run_cfg.get("compile", cfg.get("compile", False))):
        model = torch.compile(model)
    model.eval()

    precision = run_cfg.get("precision", cfg.get("precision", 32))
    warmup_batches = int(run_cfg.get("warmup_batches", cfg.get("warmup_batches", 5)))
    max_batches = run_cfg.get("max_batches", cfg.get("max_batches", None))
    max_batches = int(max_batches) if max_batches is not None else None

    run_warmup(task_name, model, dataloader, device, precision, warmup_batches)
    state = init_metric_state(task_name, model, datamodule_cfg, device)
    memory = memory_start(device)

    total_forward_time = 0.0
    total_samples = 0
    total_batches = 0
    maybe_sync(device)
    with torch.inference_mode():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            batch = move_to_device(batch, device)
            samples = batch_size(batch)
            maybe_sync(device)
            start_time = time.perf_counter()
            with autocast_context(device, precision):
                logits = forward_for_task(task_name, model, batch)
            maybe_sync(device)
            total_forward_time += time.perf_counter() - start_time

            update_metrics_for_task(task_name, model, batch, logits, state)
            total_samples += samples
            total_batches += 1

    maybe_sync(device)
    memory = memory_end(device, memory)
    metrics = compute_metrics(task_name, state)
    avg_ms_per_batch = (total_forward_time / max(total_batches, 1)) * 1000.0
    avg_ms_per_sample = (total_forward_time / max(total_samples, 1)) * 1000.0

    return {
        "status": "ok",
        "name": str(run_cfg.name),
        "task": task_name,
        "checkpoint_path": str(checkpoint_path),
        "datamodule": to_plain_container(datamodule_cfg),
        "device": str(device),
        "precision": str(precision),
        "num_samples": total_samples,
        "num_batches": total_batches,
        "timing": {
            "forward_time_sec": total_forward_time,
            "ms_per_batch": avg_ms_per_batch,
            "ms_per_sample": avg_ms_per_sample,
            "samples_per_sec": total_samples / total_forward_time if total_forward_time > 0 else 0.0,
            "batches_per_sec": total_batches / total_forward_time if total_forward_time > 0 else 0.0,
            "warmup_batches": warmup_batches,
        },
        "memory": memory,
        "metrics": metrics,
    }


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    flat = {
        "status": result.get("status"),
        "name": result.get("name"),
        "task": result.get("task"),
        "checkpoint_path": result.get("checkpoint_path"),
        "device": result.get("device"),
        "precision": result.get("precision"),
        "num_samples": result.get("num_samples"),
        "num_batches": result.get("num_batches"),
        "error": result.get("error"),
    }
    for group in ("timing", "memory", "metrics"):
        for key, value in result.get(group, {}).items():
            flat[f"{group}.{key}"] = value
    return flat


def write_outputs(results: list[dict[str, Any]], cfg: DictConfig):
    output_dir = resolve_path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / str(cfg.output_jsonl)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, sort_keys=True) + "\n")

    csv_path = output_dir / str(cfg.output_csv)
    rows = [flatten_result(result) for result in results]
    fieldnames = sorted({field for row in rows for field in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return jsonl_path, csv_path


@hydra.main(version_base="1.3", config_path="configs/benchmark", config_name="default")
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision("high")
    if cfg.get("seed") is not None:
        torch.manual_seed(int(cfg.seed))

    device = select_device(str(cfg.device))
    runs = expand_runs(cfg)
    if len(runs) == 0:
        print("No benchmark runs configured. Add entries under runs.")
        return

    results = []
    failures = []
    for run_cfg in runs:
        print(f"Benchmarking {run_cfg.name} on {device}...")
        try:
            result = benchmark_run(run_cfg, cfg, device)
            metrics = result["metrics"]
            timing = result["timing"]
            print(
                f"  ok: {result['num_samples']} samples, "
                f"{timing['ms_per_sample']:.3f} ms/sample, metrics={metrics}"
            )
        except Exception as exc:
            if bool(cfg.get("fail_fast", False)):
                raise
            result = {
                "status": "failed",
                "name": str(run_cfg.get("name", "<unnamed>")),
                "task": str(run_cfg.get("task", "<unknown>")),
                "checkpoint_path": str(run_cfg.get("checkpoint_path", "")),
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(result)
            print(f"  failed: {result['error']}")
        results.append(result)

    jsonl_path, csv_path = write_outputs(results, cfg)
    print(f"Wrote benchmark JSONL: {jsonl_path}")
    print(f"Wrote benchmark CSV: {csv_path}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
