import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict
import pickle
import gc
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

from ..utils.config import Config
from ..utils.seed import set_seed

class PredictiveCodingTrainer:
    """
    Trainer for predictive coding models.
    """
    
    def __init__(self, config: Config, model: nn.Module, device: Optional[str] = None):
        """
        Initialize trainer.
        
        Args:
            config (Config): Configuration object
            model (nn.Module): Model to train
            device (Optional[str]): Device to use (auto-detected if None)
        """
        self.config = config
        self.model = model
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Training parameters
        self.epochs = config.get('training.epochs', 30)
        self.batch_size = config.get('training.batch_size', 256)
        self.learning_rate = config.get('training.learning_rate', 5e-4)
        self.weight_decay = config.get('training.weight_decay', 1e-5)
        self.max_grad_norm = config.get('training.max_grad_norm', 5.0)
        
        # Logging
        self.train_loss_log = defaultdict(list)
        self.val_loss_log = defaultdict(list)
        self.setup_logging()
        
        # Create save directory
        self.save_dir = Path(config.get('paths.model_save_dir', 'models'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Using device: {self.device}")
        print(f"Model: {model.get_model_info()}")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config.get('logging.log_dir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, 
                   loss_fn: nn.Module, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            loss_fn: Loss function
            epoch: Current epoch number
            
        Returns:
            float: Average training loss
        """
        self.model.train()
        train_loss = 0.0
        
        for batch_idx, (xb, yb) in enumerate(tqdm(train_loader, 
                                                   desc=f"Epoch {epoch} [Train]", 
                                                   leave=False)):
            xb, yb = xb.to(self.device), yb.to(self.device)

            optimizer.zero_grad()
            pred = self.model(xb)

            # Check for NaN/Inf
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                self.logger.warning(f"NaN/Inf in prediction at Epoch {epoch}, Batch {batch_idx}")
                continue

            loss = loss_fn(pred, yb)
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(f"NaN/Inf in loss at Epoch {epoch}, Batch {batch_idx}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
            optimizer.step()
            train_loss += loss.item()

        return train_loss / len(train_loader)
    
    def validate_epoch(self, val_loader: DataLoader, loss_fn: nn.Module, epoch: int) -> float:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            loss_fn: Loss function
            epoch: Current epoch number
            
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.model(xb)
                val_loss += loss_fn(pred, yb).item()

        return val_loss / len(val_loader)
    
    def train_fold(self, train_loader: DataLoader, val_loader: DataLoader, 
                  subject_id: int, fold_idx: int) -> Dict[str, List[float]]:
        """
        Train a single fold.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            subject_id: Subject ID
            fold_idx: Fold index
            
        Returns:
            Dict[str, List[float]]: Training history
        """
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                    lr=self.learning_rate, 
                                    weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        best_val_loss = float('inf')
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(1, self.epochs + 1):
            # Training
            avg_train_loss = self.train_epoch(train_loader, optimizer, loss_fn, epoch)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            avg_val_loss = self.validate_epoch(val_loader, loss_fn, epoch)
            history['val_loss'].append(avg_val_loss)
            
            # Log
            print(f"Epoch {epoch:02d} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")
            self.logger.info(f"Subject {subject_id}, Fold {fold_idx+1}, Epoch {epoch}: "
                           f"Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = self.save_dir / f"subject{subject_id}_fold{fold_idx+1}_best.pt"
                torch.save(self.model.state_dict(), model_path)
                print(f"Saved best model to {model_path}")
            
            # Memory cleanup
            if epoch % 5 == 0:
                if torch.cuda.is_available():
                    print(f"GPU Mem: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
                    torch.cuda.empty_cache()
                gc.collect()
        
        return history
    
    def train_all_subjects(self, data_loader_factory) -> Dict[Tuple[int, int], Dict]:
        """
        Train all subjects and folds.
        
        Args:
            data_loader_factory: Factory for creating data loaders
            
        Returns:
            Dict: Training history for all subjects and folds
        """
        # Set seed for reproducibility
        set_seed(self.config.get('seed', 42))
        
        # Get subjects
        subjects = list(range(1, self.config.get('data.subjects', 16) + 1))
        split_type = self.config.get('cross_validation.strategy', 'trial')
        data_dir = self.config.get('paths.data_dir', '')
        
        all_histories = {}
        
        for subject_id in subjects:
            print(f"\n{'='*50}")
            print(f"Training Subject {subject_id}")
            print(f"{'='*50}")
            
            for fold_idx in range(self.config.get('cross_validation.folds', 3)):
                print(f"\nFold {fold_idx+1}/3")
                
                # Create data loaders
                train_loader, val_loader = data_loader_factory.create_predictive_coding_loaders(
                    data_dir, subject_id, fold_idx, self.batch_size, split_type
                )
                
                # Reset model for each fold
                self.model.apply(self._weights_init)
                
                # Train fold
                history = self.train_fold(train_loader, val_loader, subject_id, fold_idx)
                all_histories[(subject_id, fold_idx)] = history
                
                # Cleanup
                del train_loader, val_loader
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # Save training history
        self.save_training_history(all_histories)
        
        print(f"\n{'='*50}")
        print("Training complete for all subjects and folds!")
        print(f"{'='*50}")
        
        return all_histories
    
    def _weights_init(self, m):
        """Initialize model weights."""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def save_training_history(self, histories: Dict):
        """
        Save training history to disk.
        
        Args:
            histories: Training histories dictionary
        """
        history_path = self.save_dir / "training_history.pkl"
        with open(history_path, "wb") as f:
            pickle.dump(histories, f)
        
        print(f"Training history saved to {history_path}")
    
    def load_model(self, model_path: str):
        """
        Load model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"Model loaded from {model_path}")
