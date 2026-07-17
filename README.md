# DCAP-PROTAC



## Overview

- `main.py`: train the model
- `test_checkpoint.py`: evaluate a saved checkpoint
- `protacDataset.py`: load data and build batches
- `model.py`, `egnn.py`: model definitions

## Requirements

Python 3.10+ is recommended.

- `torch`
- `torch-geometric`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `tqdm`
- `tensorboard`
- `egnn-pytorch`

Install with:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric pandas numpy scikit-learn matplotlib tqdm tensorboard egnn-pytorch
```

## Data Preparation

Expected files:

- `data/clean_data.csv`
- `data/protacs_mol2/`
- `data/features/linker/`
- `data/features/plm/`
- `data/features/smiles/`

Required feature:

- `smiles_chemberta` in `data/features/smiles/{id}.npz`

Optional features:

- `target_plm` and `e3_plm` in `data/features/plm/{id}.npz`
- `linker_conf` in `data/features/linker/{id}.npz`

Missing optional features are replaced with zeros.

## Training

Run:

```bash
python main.py
```

Outputs are saved to:

- `log/`
- `model/`
- `plots/`
- `runs/`
- `results/`



## Evaluation

Run:

```bash
python test_checkpoint.py --checkpoint model/your_checkpoint.pt
```

Common options:

- `--checkpoint`: checkpoint path
- `--data-dir`: default `data/protacs_mol2/`
- `--clean-data`: default `data/clean_data.csv`


## Reported Metrics

The evaluation script prints:

- Loss
- Accuracy
- AUROC
- Precision
- Recall
- AUPR
- F1



