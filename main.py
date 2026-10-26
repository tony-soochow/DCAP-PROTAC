import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from model import EquivariantGraph, MyModel, SequenceConv
from protacDataset import NUM_ATOM_TYPES, PROTACDataset, collater
from train_and_test import train

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

BATCH_SIZE = 4
EPOCH = 100
TRAIN_RATE = 0.8
VALID_RATE = 0.1
LEARNING_RATE = 0.0001
RANDOM_SEED = [42, 24, 2025, 2026, 2027]
DATA_DIR = 'data/protacs_mol2/'
CLEAN_DATA = 'data/clean_data.csv'
NUM_WORKERS = 4


def build_run_name(seed):
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    return f"smiles-bert-ET-PROTAC-(linker+plm)batch4-row811-seed{seed}-{timestamp}"


def ensure_dirs():
    for folder in ('log', 'model', 'plots', 'runs', 'results'):
        Path(folder).mkdir(exist_ok=True)


def build_model():
    return MyModel(
        EquivariantGraph(num_embeddings=NUM_ATOM_TYPES, n_layers=3),
        SequenceConv(MAX_LENGTH=1000),
        EquivariantGraph(num_embeddings=NUM_ATOM_TYPES, n_layers=3),
        SequenceConv(MAX_LENGTH=1000),
        EquivariantGraph(num_embeddings=NUM_ATOM_TYPES, n_layers=3),
    )


def build_loaders(seed):
    protac_set = PROTACDataset(DATA_DIR, CLEAN_DATA)
    data_size = len(protac_set)
    train_size = int(data_size * TRAIN_RATE)
    valid_size = int(data_size * VALID_RATE)
    test_size = data_size - train_size - valid_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, valid_dataset, test_dataset = random_split(
        protac_set, lengths=[train_size, valid_size, test_size], generator=generator
    )

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        drop_last=False,
        collate_fn=collater,
        num_workers=NUM_WORKERS,
        persistent_workers=NUM_WORKERS > 0,
    )
    if NUM_WORKERS > 0:
        loader_kwargs['prefetch_factor'] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return {
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'test_loader': test_loader,
        'data_size': data_size,
        'train_size': train_size,
        'valid_size': valid_size,
        'test_size': test_size,
    }


def run_one_seed(seed, epoch=EPOCH):
    train_name = build_run_name(seed)
    log_path = f'log/{train_name}.log'
    logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO, force=True)

    loaders = build_loaders(seed)
    logging.info(f'seed: {seed}')
    logging.info('split: row-level 8:1:1 (no compound id isolation)')
    logging.info(f"all data: {loaders['data_size']}")
    logging.info(f"train data: {loaders['train_size']}")
    logging.info(f"valid data: {loaders['valid_size']}")
    logging.info(f"test data: {loaders['test_size']}")
    print(f"[INFO] seed {seed} split: train={loaders['train_size']} "
          f"valid={loaders['valid_size']} test={loaders['test_size']} "
          f"(total={loaders['data_size']}, row-level 8:1:1)")

    model = build_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with SummaryWriter(f'runs/{train_name}') as writer:
        _, details = train(
            model,
            train_loader=loaders['train_loader'],
            valid_loader=loaders['valid_loader'],
            test_loader=loaders['test_loader'],
            device=device,
            writer=writer,
            LOSS_NAME=train_name,
            batch_size=BATCH_SIZE,
            epoch=epoch,
            lr=LEARNING_RATE,
            accumulation_steps=1,
            return_details=True,
        )

    summary = {
        'seed': seed,
        'train_name': train_name,
        'split': 'row-level 8:1:1 (no compound id isolation)',
        'data_size': loaders['data_size'],
        'train_size': loaders['train_size'],
        'valid_size': loaders['valid_size'],
        'test_size': loaders['test_size'],
        'log_path': log_path,
        'model_path': f'model/{train_name}.pt',
        'plot_path': f'plots/{train_name}/training_curves.png',
        'best_metrics': details['best_metrics'],
        'test_metrics': details['test_metrics'],
        'history': details['history'],
    }
    with open(f'results/{train_name}.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def save_summary_table(rows):
    csv_path = 'results/summary.csv'
    fieldnames = [
        'seed', 'train_name', 'train_size', 'valid_size', 'test_size',
        'best_epoch',
        'val_loss', 'val_acc', 'val_auroc', 'val_precision', 'val_recall', 'val_aupr', 'val_f1',
        'test_loss', 'test_acc', 'test_auroc', 'test_precision', 'test_recall', 'test_aupr', 'test_f1',
        'model_path', 'log_path', 'plot_path',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = {}
            metrics.update(row.get('best_metrics') or {})
            metrics.update(row.get('test_metrics') or {})
            writer.writerow({
                'seed': row['seed'],
                'train_name': row['train_name'],
                'train_size': row.get('train_size', ''),
                'valid_size': row.get('valid_size', ''),
                'test_size': row.get('test_size', ''),
                'best_epoch': metrics.get('best_epoch', ''),
                'val_loss': metrics.get('val_loss', ''),
                'val_acc': metrics.get('val_acc', ''),
                'val_auroc': metrics.get('val_auroc', ''),
                'val_precision': metrics.get('val_precision', ''),
                'val_recall': metrics.get('val_recall', ''),
                'val_aupr': metrics.get('val_aupr', ''),
                'val_f1': metrics.get('val_f1', ''),
                'test_loss': metrics.get('test_loss', ''),
                'test_acc': metrics.get('test_acc', ''),
                'test_auroc': metrics.get('test_auroc', ''),
                'test_precision': metrics.get('test_precision', ''),
                'test_recall': metrics.get('test_recall', ''),
                'test_aupr': metrics.get('test_aupr', ''),
                'test_f1': metrics.get('test_f1', ''),
                'model_path': row['model_path'],
                'log_path': row['log_path'],
                'plot_path': row['plot_path'],
            })


def main():
    parser = argparse.ArgumentParser(description='lzhdcap1001v1 Full 模型训练 (行级 8:1:1)')
    parser.add_argument('--seeds', type=str, default=','.join(map(str, RANDOM_SEED)),
                        help='逗号分隔的 seed 列表, 例如 42,24,2025')
    parser.add_argument('--epoch', type=int, default=EPOCH, help='训练轮数')
    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]

    ensure_dirs()
    summaries = []
    for seed in seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        summary = run_one_seed(seed, epoch=args.epoch)
        summaries.append(summary)
    save_summary_table(summaries)


if __name__ == '__main__':
    main()
