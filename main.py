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
LEARNING_RATE = 0.0001
RANDOM_SEED = [42, 24, 2025, 2026, 2027]
DATA_DIR = 'data/protacs_mol2/'
CLEAN_DATA = 'data/clean_data.csv'


def build_run_name(seed):
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    return f"smiles-bert-ET-PROTAC-(linker+plm)batch4-seed{seed}-{timestamp}"


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


def run_one_seed(seed):
    train_name = build_run_name(seed)
    log_path = f'log/{train_name}.log'
    logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO, force=True)

    protac_set = PROTACDataset(DATA_DIR, CLEAN_DATA)
    data_size = len(protac_set)
    train_size = int(data_size * TRAIN_RATE)
    test_size = data_size - train_size
    logging.info(f'seed: {seed}')
    logging.info(f'all data: {data_size}')
    logging.info(f'train data: {train_size}')
    logging.info(f'test data: {test_size}')

    generator = torch.Generator().manual_seed(seed)
    train_dataset, test_dataset = random_split(protac_set, lengths=[train_size, test_size], generator=generator)
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=True, collate_fn=collater)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False, collate_fn=collater)

    model = build_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with SummaryWriter(f'runs/{train_name}') as writer:
        _, details = train(
            model,
            train_loader=trainloader,
            valid_loader=testloader,
            device=device,
            writer=writer,
            LOSS_NAME=train_name,
            batch_size=BATCH_SIZE,
            epoch=EPOCH,
            lr=LEARNING_RATE,
            accumulation_steps=1,
            return_details=True,
        )

    summary = {
        'seed': seed,
        'train_name': train_name,
        'log_path': log_path,
        'model_path': f'model/{train_name}.pt',
        'plot_path': f'plots/{train_name}/training_curves.png',
        'best_metrics': details['best_metrics'],
        'history': details['history'],
    }
    with open(f'results/{train_name}.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def save_summary_table(rows):
    csv_path = 'results/summary.csv'
    fieldnames = [
        'seed', 'train_name', 'best_epoch', 'val_loss', 'val_acc', 'val_auroc',
        'val_precision', 'val_recall', 'val_aupr', 'val_f1', 'model_path', 'log_path', 'plot_path',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = row.get('best_metrics') or {}
            writer.writerow({
                'seed': row['seed'],
                'train_name': row['train_name'],
                'best_epoch': metrics.get('best_epoch', ''),
                'val_loss': metrics.get('val_loss', ''),
                'val_acc': metrics.get('val_acc', ''),
                'val_auroc': metrics.get('val_auroc', ''),
                'val_precision': metrics.get('val_precision', ''),
                'val_recall': metrics.get('val_recall', ''),
                'val_aupr': metrics.get('val_aupr', ''),
                'val_f1': metrics.get('val_f1', ''),
                'model_path': row['model_path'],
                'log_path': row['log_path'],
                'plot_path': row['plot_path'],
            })


def main():
    ensure_dirs()
    summaries = []
    for seed in RANDOM_SEED:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        summary = run_one_seed(seed)
        summaries.append(summary)
    save_summary_table(summaries)


if __name__ == '__main__':
    main()

