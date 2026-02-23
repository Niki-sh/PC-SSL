#!/usr/bin/env python3
"""
Training script for predictive coding models.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.predictive_coding import CNNPredictiveCodingDE_Attn
from src.training.trainer import PredictiveCodingTrainer
from src.data.dataset import DataLoaderFactory
from src.utils.config import Config
from src.utils.seed import set_seed

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train predictive coding model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--model_type", type=str, default="cnn_predictive_coding",
                       choices=["cnn_predictive_coding", "lightweight"],
                       help="Model architecture to train")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device to use for training")
    parser.add_argument("--subject", type=int,
                       help="Train specific subject only")
    parser.add_argument("--fold", type=int,
                       help="Train specific fold only")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Set seed for reproducibility
    set_seed(config.get('seed', 42))
    
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
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Create trainer
    device = None if args.device == "auto" else args.device
    trainer = PredictiveCodingTrainer(config, model, device)
    
    # Create data loader factory
    data_loader_factory = DataLoaderFactory()
    
    # Train specific subject/fold or all
    if args.subject is not None:
        # Train specific subject
        subjects = [args.subject]
    else:
        # Train all subjects
        subjects = list(range(1, config.get('data.subjects', 16) + 1))
    
    split_type = config.get('cross_validation.strategy', 'trial')
    data_dir = config.get('paths.data_dir', '')
    
    for subject_id in subjects:
        print(f"\n{'='*50}")
        print(f"Training Subject {subject_id}")
        print(f"{'='*50}")
        
        folds_to_train = [args.fold] if args.fold is not None else range(config.get('cross_validation.folds', 3))
        
        for fold_idx in folds_to_train:
            print(f"\nFold {fold_idx+1}")
            
            # Create data loaders
            train_loader, val_loader = data_loader_factory.create_predictive_coding_loaders(
                data_dir, subject_id, fold_idx, 
                config.get('training.batch_size', 256), 
                split_type
            )
            
            # Train fold
            history = trainer.train_fold(train_loader, val_loader, subject_id, fold_idx)
            
            # Cleanup
            del train_loader, val_loader
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    print(f"\n{'='*50}")
    print("Training completed!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
