import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any

class DEPredictiveCodingDataset(Dataset):
    """
    Dataset class for predictive coding with DE features.
    """
    
    def __init__(self, past_dict: Dict, future_dict: Dict, indices: List[int] = None):
        """
        Initialize dataset.
        
        Args:
            past_dict: Dictionary of past segments by subject
            future_dict: Dictionary of future segments by subject
            indices: List of specific indices to use (optional)
        """
        self.samples = []
        
        if indices is not None:
            # Use specific indices (for cross-validation)
            # Assuming we're working with a single subject
            subject_id = list(past_dict.keys())[0]
            for i in indices:
                self.samples.append((past_dict[subject_id][i], future_dict[subject_id][i]))
        else:
            # Use all data from all subjects
            for subj_id in past_dict:
                for past_seg, future_seg in zip(past_dict[subj_id], future_dict[subj_id]):
                    self.samples.append((past_seg, future_seg))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_past, x_future = self.samples[idx]
        x_past = torch.tensor(x_past, dtype=torch.float32).unsqueeze(0)   # (1, 62, 5)
        x_future = torch.tensor(x_future, dtype=torch.float32).unsqueeze(0)
        return x_past, x_future

class LabeledDEDataset(Dataset):
    """
    Dataset class for labeled DE features (for downstream tasks).
    """
    
    def __init__(self, data_dict: Dict, label_dict: Dict, indices: List[int] = None):
        """
        Initialize labeled dataset.
        
        Args:
            data_dict: Dictionary of data segments by subject
            label_dict: Dictionary of labels by subject
            indices: List of specific indices to use (optional)
        """
        self.samples = []
        
        if indices is not None:
            # Use specific indices (for cross-validation)
            subject_id = list(data_dict.keys())[0]
            for i in indices:
                self.samples.append((data_dict[subject_id][i], label_dict[subject_id][i]))
        else:
            # Use all data from all subjects
            for subj_id in data_dict:
                for data_seg, label in zip(data_dict[subj_id], label_dict[subj_id]):
                    self.samples.append((data_seg, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_data, y_label = self.samples[idx]
        x_data = torch.tensor(x_data, dtype=torch.float32).unsqueeze(0)   # (1, 62, 5)
        
        # Handle label format
        if isinstance(y_label, (list, np.ndarray)):
            y_label = int(y_label[0])
        else:
            y_label = int(y_label)
            
        y_label = torch.tensor(y_label, dtype=torch.long)
        return x_data, y_label

class DataLoaderFactory:
    """
    Factory class for creating data loaders.
    """
    
    @staticmethod
    def create_predictive_coding_loaders(data_dir: str, subject_id: int, fold_idx: int,
                                       batch_size: int = 256, split_type: str = "trial") -> Tuple[DataLoader, DataLoader]:
        """
        Create data loaders for predictive coding task.
        
        Args:
            data_dir: Directory containing processed data
            subject_id: Subject ID
            fold_idx: Fold index (0-based)
            batch_size: Batch size
            split_type: Type of split ("trial" or "session")
            
        Returns:
            Tuple[DataLoader, DataLoader]: Train and validation loaders
        """
        data_dir = Path(data_dir)
        
        # Load data
        with open(data_dir / "past_by_subject_DE.pkl", "rb") as f:
            past_by_subject = pickle.load(f)
        with open(data_dir / "future_by_subject_DE.pkl", "rb") as f:
            future_by_subject = pickle.load(f)
        
        # Load folds
        fold_filename = f"folds_by_subject_{split_type}_DE.pkl"
        with open(data_dir / fold_filename, "rb") as f:
            folds_by_subject = pickle.load(f)
        
        # Get fold information
        fold = folds_by_subject[subject_id][fold_idx]
        train_indices = fold['train_indices']
        val_indices = fold['test_indices']
        
        # Create datasets
        train_dataset = DEPredictiveCodingDataset(
            {subject_id: past_by_subject[subject_id]},
            {subject_id: future_by_subject[subject_id]},
            train_indices
        )
        
        val_dataset = DEPredictiveCodingDataset(
            {subject_id: past_by_subject[subject_id]},
            {subject_id: future_by_subject[subject_id]},
            val_indices
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    @staticmethod
    def create_labeled_loaders(data_dir: str, subject_id: int, fold_idx: int,
                             batch_size: int = 256, split_type: str = "trial") -> Tuple[DataLoader, DataLoader]:
        """
        Create data loaders for labeled classification task.
        
        Args:
            data_dir: Directory containing processed data
            subject_id: Subject ID
            fold_idx: Fold index (0-based)
            batch_size: Batch size
            split_type: Type of split ("trial" or "session")
            
        Returns:
            Tuple[DataLoader, DataLoader]: Train and validation loaders
        """
        data_dir = Path(data_dir)
        
        # Load data
        with open(data_dir / "past_by_subject_DE.pkl", "rb") as f:
            past_by_subject = pickle.load(f)
        with open(data_dir / "past_labels_by_subject_DE.pkl", "rb") as f:
            labels_by_subject = pickle.load(f)
        
        # Load folds
        fold_filename = f"folds_by_subject_{split_type}_DE.pkl"
        with open(data_dir / fold_filename, "rb") as f:
            folds_by_subject = pickle.load(f)
        
        # Get fold information
        fold = folds_by_subject[subject_id][fold_idx]
        train_indices = fold['train_indices']
        val_indices = fold['test_indices']
        
        # Create datasets
        train_dataset = LabeledDEDataset(
            {subject_id: past_by_subject[subject_id]},
            {subject_id: labels_by_subject[subject_id]},
            train_indices
        )
        
        val_dataset = LabeledDEDataset(
            {subject_id: past_by_subject[subject_id]},
            {subject_id: labels_by_subject[subject_id]},
            val_indices
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
