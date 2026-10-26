#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import EquivariantGraph, MyModel, SequenceConv
from protacDataset import NUM_ATOM_TYPES, PROTACDataset, collater

DEFAULT_CHECKPOINT = "model/full2026.pt"
DEFAULT_DATA_DIR = "data/protacs_mol2/"
DEFAULT_CLEAN_DATA = "data/clean_data.csv"
DEFAULT_FEATURE_ROOT = "data/features"
DEFAULT_OUT = "results/test_metrics.json"

SPLIT_DESCRIPTION = "8:1:1 (row-level random split, no compound id isolation)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model() -> MyModel:
    return MyModel(
        EquivariantGraph(num_embeddings=NUM_ATOM_TYPES, n_layers=3),
        SequenceConv(MAX_LENGTH=1000),
        EquivariantGraph(num_embeddings=NUM_ATOM_TYPES, n_layers=3),
        SequenceConv(MAX_LENGTH=1000),
        EquivariantGraph(num_embeddings=NUM_ATOM_TYPES, n_layers=3),
    )


def build_model_inputs(data_sample, device):
    return (
        data_sample["ligase_ligand"].to(device),
        data_sample["ligase"].to(device),
        data_sample["target_ligand"].to(device),
        data_sample["target"].to(device),
        data_sample["linker"].to(device),
        data_sample["target_plm"].to(device),
        data_sample["e3_plm"].to(device),
        data_sample["linker_conf"].to(device),
        data_sample["smiles_chemberta"].to(device),
    )


@torch.no_grad()
def evaluate(model: MyModel, loader: DataLoader, device: torch.device):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    y_true, y_pred, y_score, losses = [], [], [], []
    for data_sample in loader:
        labels = data_sample["label"].to(device)
        outputs, _, _ = model(*build_model_inputs(data_sample, device))
        losses.append(criterion(outputs, labels).item())
        probs = torch.nn.functional.softmax(outputs, 1)[:, 1]
        y_score.extend(probs.cpu().tolist())
        y_pred.extend(torch.max(outputs, 1)[1].cpu().tolist())
        y_true.extend(labels.cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auroc = roc_auc_score(y_true, y_score)
    except ValueError:
        auroc = 0.0
    try:
        aupr = average_precision_score(y_true, y_score)
    except ValueError:
        aupr = 0.0

    return float(np.mean(losses)), acc, auroc, precision, recall, aupr, f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the released DCAP-PROTACs checkpoint on the 8:1:1 test split"
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="trained checkpoint (.pt)")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="directory with fragment mol2 files")
    parser.add_argument("--clean-data", default=DEFAULT_CLEAN_DATA, help="input CSV with 1086 records")
    parser.add_argument("--feature-root", default=DEFAULT_FEATURE_ROOT, help="root of precomputed features")
    parser.add_argument("--seed", type=int, default=137, help="seed of the 8:1:1 split")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out", default=DEFAULT_OUT, help="path of the metrics JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint}\n"
            "The capsule mounts uploaded data at /data; keep the checkpoint at "
            "/data/model/full2026.pt or pass --checkpoint."
        )

    feature_root = Path(args.feature_root)
    if not feature_root.is_dir():
        raise FileNotFoundError(f"feature directory not found: {feature_root}")

    dataset = PROTACDataset(args.data_dir, args.clean_data, feature_root=str(feature_root))
    total = len(dataset)
    train_size = int(total * 0.8)
    valid_size = int(total * 0.1)
    test_size = total - train_size - valid_size

    _, _, test_set = random_split(
        dataset,
        lengths=[train_size, valid_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collater,
        num_workers=args.num_workers,
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = build_model()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device)

    loss, acc, auroc, precision, recall, aupr, f1 = evaluate(model, loader, device)

    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {device.type}")
    print(f"Seed: {args.seed} | Total samples: {total} | Test samples: {test_size}")
    print(
        f"Test Metrics | AUROC: {auroc:.4f} | ACC: {acc:.4f} | Precision: {precision:.4f} "
        f"| Recall: {recall:.4f} | AUPR: {aupr:.4f} | F1: {f1:.4f}"
    )

    report = {
        "checkpoint": str(checkpoint),
        "device": device.type,
        "seed": args.seed,
        "train_rate": round(train_size / total, 4),
        "batch_size": args.batch_size,
        "total_samples": total,
        "test_samples": test_size,
        "metrics": {
            "loss": loss,
            "acc": acc,
            "auroc": auroc,
            "precision": precision,
            "recall": recall,
            "aupr": aupr,
            "f1": f1,
        },
        "split": SPLIT_DESCRIPTION,
        "train_samples": train_size,
        "valid_samples": valid_size,
        "checkpoint_sha256": sha256(checkpoint),
        "feature_root": str(feature_root),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=4) + "\n", encoding="utf-8")
    print(f"Results saved to: {out}")


if __name__ == "__main__":
    main()
