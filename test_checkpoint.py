import argparse
import os
import random
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

from model import EquivariantGraph, MyModel, SequenceConv
from protacDataset import NUM_ATOM_TYPES, PROTACDataset, collater


DEFAULT_CHECKPOINT = (
    "/home/admin641/disk_2T/0528代码备份/model/full2026-05-29_13-52-01.pt"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved ET-PROTACs checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to .pt checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/protacs_mol2/", help="Path to mol2 files")
    parser.add_argument("--clean-data", type=str, default="data/clean_data.csv", help="CSV path")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed used for the train/test split")
    parser.add_argument("--train-rate", type=float, default=0.8, help="Train split ratio used during training")
    parser.add_argument("--batch-size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device")
    parser.add_argument(
        "--sequence-mode",
        type=str,
        default="dual",
        choices=["dual", "cnn", "lstm"],
        help="Sequence encoder mode used by the checkpoint",
    )
    parser.add_argument(
        "--use-cross-attention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable cross-attention in the model",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_model(sequence_mode="dual", use_cross_attention=True):
    return MyModel(
        EquivariantGraph(num_embeddings=NUM_ATOM_TYPES, n_layers=3),
        SequenceConv(MAX_LENGTH=1000, sequence_mode=sequence_mode),
        EquivariantGraph(num_embeddings=NUM_ATOM_TYPES, n_layers=3),
        SequenceConv(MAX_LENGTH=1000, sequence_mode=sequence_mode),
        EquivariantGraph(num_embeddings=NUM_ATOM_TYPES, n_layers=3),
        use_cross_attention=use_cross_attention,
    )


def load_state_dict(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint

    raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")


def build_test_loader(data_dir, clean_data, seed, train_rate, batch_size):
    dataset = PROTACDataset(data_dir, clean_data)
    data_size = len(dataset)
    train_size = int(data_size * train_rate)
    test_size = data_size - train_size

    generator = torch.Generator().manual_seed(seed)
    _, test_dataset = random_split(dataset, lengths=[train_size, test_size], generator=generator)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
        collate_fn=collater,
    )
    return test_loader, data_size, test_size


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


def evaluate(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    y_true = []
    y_pred = []
    y_score = []
    losses = []

    with torch.no_grad():
        for data_sample in test_loader:
            labels = data_sample["label"].to(device)
            model_inputs = build_model_inputs(data_sample, device)
            outputs, _, _ = model(*model_inputs)
            losses.append(criterion(outputs, labels).item())

            probs = torch.softmax(outputs, dim=1)[:, 1]
            y_score.extend(probs.detach().cpu().tolist())
            y_pred.extend(torch.argmax(outputs, dim=1).detach().cpu().tolist())
            y_true.extend(labels.detach().cpu().tolist())

    avg_loss = float(np.mean(losses)) if losses else 0.0
    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    precision = precision_score(y_true, y_pred, zero_division=0) if y_true else 0.0
    recall = recall_score(y_true, y_pred, zero_division=0) if y_true else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0) if y_true else 0.0

    try:
        auroc = roc_auc_score(y_true, y_score)
    except ValueError:
        auroc = 0.0

    try:
        aupr = average_precision_score(y_true, y_score)
    except ValueError:
        aupr = 0.0

    return {
        "loss": avg_loss,
        "acc": float(acc),
        "auroc": float(auroc),
        "precision": float(precision),
        "recall": float(recall),
        "aupr": float(aupr),
        "f1": float(f1),
    }


def print_metrics(metrics):
    print(
        "Test Metrics | "
        f"AUROC: {metrics['auroc']:.4f} | "
        f"ACC: {metrics['acc']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | "
        f"AUPR: {metrics['aupr']:.4f} | "
        f"F1: {metrics['f1']:.4f}"
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    test_loader, data_size, test_size = build_test_loader(
        args.data_dir,
        args.clean_data,
        args.seed,
        args.train_rate,
        args.batch_size,
    )

    model = build_model(
        sequence_mode=args.sequence_mode,
        use_cross_attention=args.use_cross_attention,
    ).to(device)
    state_dict = load_state_dict(checkpoint_path, device)
    model.load_state_dict(state_dict)

    metrics = evaluate(model, test_loader, device)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed} | Total samples: {data_size} | Test samples: {test_size}")
    print_metrics(metrics)


if __name__ == "__main__":
    main()