import logging
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score, f1_score)
from tqdm import tqdm


def build_model_inputs(data_sample, device, ablation_flags=None):
    """Build model inputs and optionally drop selected auxiliary features."""
    flags = ablation_flags or {}

    target_plm = data_sample['target_plm'].to(device) if flags.get('use_plm', True) else None
    e3_plm = data_sample['e3_plm'].to(device) if flags.get('use_plm', True) else None
    linker_conf = data_sample['linker_conf'].to(device) if flags.get('use_linker_conf', True) else None
    smiles_chemberta = data_sample['smiles_chemberta'].to(device) if flags.get('use_smiles', True) else None

    return (
        data_sample['ligase_ligand'].to(device),
        data_sample['ligase'].to(device),
        data_sample['target_ligand'].to(device),
        data_sample['target'].to(device),
        data_sample['linker'].to(device),
        target_plm,
        e3_plm,
        linker_conf,
        smiles_chemberta,
    )


def valids(model, test_loader, device, ablation_flags=None):
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        model.eval()
        y_true = []
        y_pred = []
        y_score = []
        losses = []

        for data_sample in test_loader:
            y = data_sample['label'].to(device)
            model_inputs = build_model_inputs(data_sample, device, ablation_flags)
            outputs, _, _ = model(*model_inputs)
            loss_val = criterion(outputs, y)
            losses.append(loss_val.item())
            probs = torch.nn.functional.softmax(outputs, 1)[:, 1]
            y_score.extend(probs.cpu().tolist())
            y_pred.extend(torch.max(outputs, 1)[1].cpu().tolist())
            y_true.extend(y.cpu().tolist())

        model.train()

    avg_loss = np.mean(losses)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall_val = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)


    try:
        auroc = roc_auc_score(y_true, y_score) 
    except ValueError:
        auroc = 0.0
    try:
        aupr = average_precision_score(y_true, y_score)
    except ValueError:
        aupr = 0.0

    return avg_loss, acc, auroc, precision, recall_val, aupr, f1


def plot_metrics(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', markersize=2, label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-o', markersize=2, label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(epochs, history['val_acc'], 'g-o', markersize=2, label='Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # AUROC
    axes[0, 2].plot(epochs, history['val_auroc'], 'm-o', markersize=2, label='AUROC')
    axes[0, 2].set_title('AUROC')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Precision & Recall
    axes[1, 0].plot(epochs, history['val_precision'], 'c-o', markersize=2, label='Precision')
    axes[1, 0].plot(epochs, history['val_recall'], 'y-o', markersize=2, label='Recall')
    axes[1, 0].set_title('Precision & Recall')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # F1 Score
    axes[1, 1].plot(epochs, history['val_f1'], 'k-o', markersize=2, label='F1 Score')
    axes[1, 1].set_title('F1 Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # AUPR + Learning Rate
    ax_aupr = axes[1, 2]
    ax_aupr.plot(epochs, history['val_aupr'], 'r-o', markersize=2, label='AUPR')
    ax_aupr.set_title('AUPR & Learning Rate')
    ax_aupr.set_xlabel('Epoch')
    ax_aupr.set_ylabel('AUPR', color='r')
    ax_aupr.grid(True, alpha=0.3)
    ax_lr = ax_aupr.twinx()
    ax_lr.plot(epochs, history['lr'], 'b--', alpha=0.6, label='LR')
    ax_lr.set_ylabel('Learning Rate', color='b')
    lines1, labels1 = ax_aupr.get_legend_handles_labels()
    lines2, labels2 = ax_lr.get_legend_handles_labels()
    ax_aupr.legend(lines1 + lines2, labels1 + labels2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()


def compute_class_weights(train_loader, device):
    labels = []
    for data_sample in train_loader:
        labels.extend(data_sample['label'].tolist())
    labels = np.array(labels)
    n_samples = len(labels)
    n_classes = len(np.unique(labels))
    class_counts = np.bincount(labels, minlength=2)
    weights = n_samples / (n_classes * class_counts.astype(float))
    return torch.FloatTensor(weights).to(device)


def train(model, lr=0.0001, epoch=30, train_loader=None, valid_loader=None,
          device=None, writer=None, LOSS_NAME=None, batch_size=None,
          accumulation_steps=16, ablation_flags=None, return_details=False,
          artifact_dir=None):
    model = model.to(device)
    artifact_dir = artifact_dir or '.'
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )


    weight = compute_class_weights(train_loader, device)
    logging.info(f"Class weights: {weight.cpu().tolist()}")
    print(f"[INFO] Class weights: {weight.cpu().tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weight)

    # Early stopping
    best_model_params = None
    best_val_auc = float('-inf')
    patience = 15
    patience_counter = 0


    history = {
        'train_loss': [], 'val_loss': [],
        'val_acc': [], 'val_auroc': [], 'val_precision': [],
        'val_recall': [], 'val_aupr': [], 'val_f1': [], 'lr': []
    }
    plot_dir = artifact_dir if artifact_dir is not None else f'plots/{LOSS_NAME}'


    val_loss, val_acc, auroc, precision, recall, AUPR, f1 = valids(
        model, valid_loader, device, ablation_flags=ablation_flags
    )
    print(f'[Init] Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUROC: {auroc:.4f}')

    best_metrics = None

    for epo in range(epoch):
        model.train()
        running_loss = 0.0
        n_batches = 0


        pbar = tqdm(train_loader, desc=f'Epoch {epo+1}/{epoch}', ncols=100, leave=True)
        opt.zero_grad() 

        for data_sample in pbar:
            model_inputs = build_model_inputs(data_sample, device, ablation_flags)
            outputs, _, _ = model(*model_inputs)
            y = data_sample['label'].to(device)
            loss = criterion(outputs, y)

            loss = loss / accumulation_steps


            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f'Epoch {epo+1}: NaN/Inf loss detected, skipping batch')
                continue

            loss.backward()
            n_batches += 1


            if n_batches % accumulation_steps == 0:

                has_nan_grad = False
                for p in model.parameters():
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        has_nan_grad = True
                        break
                if has_nan_grad:
                    opt.zero_grad()
                    logging.warning(f'Epoch {epo+1}: NaN gradient detected, skipping update')
                    continue

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                opt.zero_grad()

            running_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})


        if n_batches % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            opt.zero_grad()

        avg_train_loss = running_loss / max(n_batches, 1)


        val_loss, val_acc, auroc, precision, recall, AUPR, f1 = valids(
            model, valid_loader, device, ablation_flags=ablation_flags
        )

        current_lr = opt.param_groups[0]['lr']
        scheduler.step(auroc)
        new_lr = opt.param_groups[0]['lr']
        if new_lr < current_lr:
            print(f'  LR reduced: {current_lr:.6f} -> {new_lr:.6f}')


        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auroc'].append(auroc)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['val_aupr'].append(AUPR)
        history['val_f1'].append(f1)
        history['lr'].append(current_lr)


        if writer is not None:
            writer.add_scalar(f'{LOSS_NAME}/train_loss', avg_train_loss, epo)
            writer.add_scalar(f'{LOSS_NAME}/val_loss', val_loss, epo)
            writer.add_scalar(f'{LOSS_NAME}/val_acc', val_acc, epo)
            writer.add_scalar(f'{LOSS_NAME}/val_auroc', auroc, epo)
            writer.add_scalar(f'{LOSS_NAME}/precision', precision, epo)
            writer.add_scalar(f'{LOSS_NAME}/recall', recall, epo)
            writer.add_scalar(f'{LOSS_NAME}/AUPR', AUPR, epo)
            writer.add_scalar(f'{LOSS_NAME}/F1', f1, epo)
            writer.add_scalar(f'{LOSS_NAME}/lr', current_lr, epo)


        print(f'[Epoch {epo+1}/{epoch}] '
              f'Train: {avg_train_loss:.4f} | '
              f'Val: {val_loss:.4f} | '
              f'Acc: {val_acc:.4f} | '
              f'AUROC: {auroc:.4f} | '
              f'AUPR: {AUPR:.4f} | '
              f'F1: {f1:.4f}')

        logging.info(
            f'Epoch {epo+1}, train_loss: {avg_train_loss:.4f}, val_loss: {val_loss:.4f}, '
            f'acc: {val_acc:.4f}, auroc: {auroc:.4f}, precision: {precision:.4f}, '
            f'recall: {recall:.4f}, AUPR: {AUPR:.4f}, F1: {f1:.4f}, lr: {current_lr:.6f}')


        if auroc > best_val_auc:
            best_val_auc = auroc
            best_model_params = model.state_dict()
            patience_counter = 0
            best_metrics = {
                'best_epoch': epo + 1,
                'val_loss': float(val_loss),
                'val_acc': float(val_acc),
                'val_auroc': float(auroc),
                'val_precision': float(precision),
                'val_recall': float(recall),
                'val_aupr': float(AUPR),
                'val_f1': float(f1),
            }
            print(f'  * New best AUROC: {auroc:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\n[STOP] Early stopping at epoch {epo+1} '
                      f'(no improvement for {patience} epochs)')
                logging.info(f'Early stopping at epoch {epo+1}')
                break


        if (epo + 1) % 5 == 0 or epo == 0:
            plot_metrics(history, plot_dir)


    plot_metrics(history, plot_dir)
    print(f'[PLOT] Training curves saved to {plot_dir}/training_curves.png')


    if best_model_params is not None:
        model_dir = os.path.join(artifact_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'{LOSS_NAME}.pt')
        torch.save(best_model_params, model_path)
        print(f'[SAVED] Best model (AUROC: {best_val_auc:.4f}) -> {model_path}')
        model.load_state_dict(best_model_params)

    details = {
        'best_metrics': best_metrics,
        'history': history,
    }
    if return_details:
        return model, details

    return model
