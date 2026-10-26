# DCAP-PROTAC

## Overview

- `main.py`: train the model
- `test_checkpoint.py`: evaluate a saved checkpoint
- `protacDataset.py`: load data and build batches
- `model.py`: model definitions

## Requirements

Python 3.10+, plus `torch`, `torch-geometric`, `pandas`, `numpy`,
`scikit-learn`, `matplotlib`, `tqdm`, `tensorboard` and `egnn-pytorch`.

## Data

- `data/clean_data.csv`
- `data/protacs_mol2/`
- `data/features/{linker,smiles,plm_target_seq,plm_e3_seq}/`

Only a few example files are included. The full dataset is available at
https://codeocean.com/capsule/6887573/tree

## Training

```bash
python main.py
```

Row-level 8:1:1 split. Outputs go to `log/`, `model/`, `plots/`, `runs/` and
`results/`.

## Evaluation

```bash
python test_checkpoint.py --checkpoint model/full2026.pt
```

The checkpoint is hosted on Code Ocean (see `model/read.txt`).


