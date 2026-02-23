import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from pathlib import Path

def plot_training_curves(histories: Dict[Tuple[int, int], Dict], 
                        save_path: str = None) -> plt.Figure:
    """
    Plot training curves for all subjects and folds.
    
    Args:
        histories: Training histories dictionary
        save_path: Path to save the plot (optional)
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot training loss
    for (subject_id, fold_idx), history in histories.items():
        epochs = range(1, len(history['train_loss']) + 1)
        axes[0].plot(epochs, history['train_loss'], 
                    label=f'S{subject_id}F{fold_idx+1}', alpha=0.7)
    
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Plot validation loss
    for (subject_id, fold_idx), history in histories.items():
        epochs = range(1, len(history['val_loss']) + 1)
        axes[1].plot(epochs, history['val_loss'], 
                    label=f'S{subject_id}F{fold_idx+1}', alpha=0.7)
    
    axes[1].set_title('Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    return fig

def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Calculate reconstruction metrics.
    
    Args:
        pred: Predicted tensor
        target: Target tensor
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    # Convert to numpy
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # Flatten
    pred_flat = pred_np.flatten()
    target_flat = target_np.flatten()
    
    # Calculate metrics
    mse = np.mean((pred_flat - target_flat) ** 2)
    mae = np.mean(np.abs(pred_flat - target_flat))
    
    # Correlation coefficient
    correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
    
    return {
        'mse': mse,
        'mae': mae,
        'correlation': correlation if not np.isnan(correlation) else 0.0
    }

def evaluate_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, 
                  device: torch.device) -> Dict[str, float]:
    """
    Evaluate model on dataset.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device to use
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(data_loader):
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            
            batch_metrics = calculate_metrics(pred, yb)
            all_metrics.append(batch_metrics)
    
    # Average metrics across batches
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    return avg_metrics

def save_predictions(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, 
                    device: torch.device, save_path: str):
    """
    Save model predictions and targets.
    
    Args:
        model: Model to use
        data_loader: Data loader
        device: Device to use
        save_path: Path to save predictions
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            
            all_predictions.append(pred.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Save to file
    np.savez(save_path, predictions=predictions, targets=targets)
    print(f"Predictions saved to {save_path}")

def create_summary_report(histories: Dict[Tuple[int, int], Dict], 
                         save_path: str = None) -> str:
    """
    Create a summary report of training results.
    
    Args:
        histories: Training histories dictionary
        save_path: Path to save the report (optional)
        
    Returns:
        str: Summary report
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("TRAINING SUMMARY REPORT")
    report_lines.append("=" * 60)
    
    # Overall statistics
    all_final_train_losses = []
    all_final_val_losses = []
    all_best_val_losses = []
    
    for (subject_id, fold_idx), history in histories.items():
        final_train_loss = history['train_loss'][-1]
        final_val_loss = history['val_loss'][-1]
        best_val_loss = min(history['val_loss'])
        
        all_final_train_losses.append(final_train_loss)
        all_final_val_losses.append(final_val_loss)
        all_best_val_losses.append(best_val_loss)
    
    report_lines.append(f"\nOVERALL STATISTICS:")
    report_lines.append(f"Total subjects/folds: {len(histories)}")
    report_lines.append(f"Average final training loss: {np.mean(all_final_train_losses):.6f} ± {np.std(all_final_train_losses):.6f}")
    report_lines.append(f"Average final validation loss: {np.mean(all_final_val_losses):.6f} ± {np.std(all_final_val_losses):.6f}")
    report_lines.append(f"Average best validation loss: {np.mean(all_best_val_losses):.6f} ± {np.std(all_best_val_losses):.6f}")
    
    # Per-subject details
    report_lines.append(f"\nPER-SUBJECT DETAILS:")
    report_lines.append("-" * 60)
    report_lines.append(f"{'Subject':<8} {'Fold':<6} {'Final Train':<12} {'Final Val':<12} {'Best Val':<12}")
    report_lines.append("-" * 60)
    
    for (subject_id, fold_idx), history in sorted(histories.items()):
        final_train_loss = history['train_loss'][-1]
        final_val_loss = history['val_loss'][-1]
        best_val_loss = min(history['val_loss'])
        
        report_lines.append(f"{subject_id:<8} {fold_idx+1:<6} {final_train_loss:<12.6f} {final_val_loss:<12.6f} {best_val_loss:<12.6f}")
    
    report_lines.append("=" * 60)
    
    report = "\n".join(report_lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Summary report saved to {save_path}")
    
    return report
