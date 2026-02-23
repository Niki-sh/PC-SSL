#!/usr/bin/env python3
"""
Evaluation script for predictive coding models.
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.predictive_coding import CNNPredictiveCodingDE_Attn
from src.data.dataset import DataLoaderFactory
from src.training.utils import evaluate_model, save_predictions, plot_training_curves, create_summary_report
from src.utils.config import Config
from src.utils.seed import set_seed

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate predictive coding model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--model_path", type=str,
                       help="Path to trained model checkpoint")
    parser.add_argument("--model_type", type=str, default="cnn_predictive_coding",
                       choices=["cnn_predictive_coding", "lightweight"],
                       help="Model architecture")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device to use for evaluation")
    parser.add_argument("--subject", type=int,
                       help="Evaluate specific subject only")
    parser.add_argument("--fold", type=int,
                       help="Evaluate specific fold only")
    parser.add_argument("--save_predictions", action="store_true",
                       help="Save model predictions")
    parser.add_argument("--plot_curves", action="store_true",
                       help="Plot training curves if history available")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Set seed for reproducibility
    set_seed(config.get('seed', 42))
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create model
    if args.model_type == "cnn_predictive_coding":
        model = CNNPredictiveCodingDE_Attn(
            num_channels=config.get('data.eeg_channels', 62),
            num_bands=config.get('data.frequency_bands', 5)
        )
    elif args.model_type == "lightweight":
        from src.models.predictive_coding import LightweightPredictiveCoding
        model = LightweightPredictiveCoding(
            num_channels=config.get('data.eeg_channels', 62),
            num_bands=config.get('data.frequency_bands', 5)
        )
    
    model.to(device)
    
    # Create data loader factory
    data_loader_factory = DataLoaderFactory()
    
    # Get paths and parameters
    data_dir = config.get('paths.data_dir', '')
    split_type = config.get('cross_validation.strategy', 'trial')
    model_save_dir = Path(config.get('paths.model_save_dir', 'models'))
    
    # Determine subjects and folds to evaluate
    if args.subject is not None:
        subjects = [args.subject]
    else:
        subjects = list(range(1, config.get('data.subjects', 16) + 1))
    
    folds_to_eval = [args.fold] if args.fold is not None else range(config.get('cross_validation.folds', 3))
    
    # Evaluation results
    all_results = {}
    
    for subject_id in subjects:
        print(f"\n{'='*50}")
        print(f"Evaluating Subject {subject_id}")
        print(f"{'='*50}")
        
        for fold_idx in folds_to_eval:
            print(f"\nFold {fold_idx+1}")
            
            # Load model
            if args.model_path:
                model_path = args.model_path
            else:
                model_path = model_save_dir / f"subject{subject_id}_fold{fold_idx+1}_best.pt"
            
            if not Path(model_path).exists():
                print(f"Model not found: {model_path}")
                continue
            
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from: {model_path}")
            
            # Create data loaders
            _, val_loader = data_loader_factory.create_predictive_coding_loaders(
                data_dir, subject_id, fold_idx, 
                config.get('training.batch_size', 256), 
                split_type
            )
            
            # Evaluate
            metrics = evaluate_model(model, val_loader, device)
            all_results[(subject_id, fold_idx)] = metrics
            
            print(f"Evaluation Results:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.6f}")
            
            # Save predictions if requested
            if args.save_predictions:
                pred_save_path = model_save_dir / f"predictions_subject{subject_id}_fold{fold_idx+1}.npz"
                save_predictions(model, val_loader, device, pred_save_path)
            
            # Cleanup
            del val_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Print summary
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    
    if all_results:
        # Calculate average metrics
        all_metrics = list(all_results.values())[0].keys()
        print(f"{'Subject':<8} {'Fold':<6}", end="")
        for metric in all_metrics:
            print(f" {metric:<12}", end="")
        print()
        print("-" * (8 + 6 + len(all_metrics) * 13))
        
        metric_sums = {metric: 0.0 for metric in all_metrics}
        count = 0
        
        for (subject_id, fold_idx), metrics in sorted(all_results.items()):
            print(f"{subject_id:<8} {fold_idx+1:<6}", end="")
            for metric in all_metrics:
                value = metrics[metric]
                metric_sums[metric] += value
                print(f" {value:<12.6f}", end="")
            print()
            count += 1
        
        # Print averages
        print("-" * (8 + 6 + len(all_metrics) * 13))
        print(f"{'Average':<8} {'':<6}", end="")
        for metric in all_metrics:
            avg_value = metric_sums[metric] / count
            print(f" {avg_value:<12.6f}", end="")
        print()
    
    # Plot training curves if requested and history available
    if args.plot_curves:
        history_path = model_save_dir / "training_history.pkl"
        if history_path.exists():
            import pickle
            with open(history_path, "rb") as f:
                histories = pickle.load(f)
            
            curves_path = model_save_dir / "training_curves.png"
            plot_training_curves(histories, curves_path)
            
            # Create summary report
            report_path = model_save_dir / "summary_report.txt"
            report = create_summary_report(histories, report_path)
            print(f"\n{report}")
        else:
            print("Training history not found. Cannot plot curves.")
    
    print(f"\n{'='*50}")
    print("Evaluation completed!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
