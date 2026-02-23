import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

class CrossValidationSplitter:
    """
    Cross-validation splitter for SEED V dataset.
    Supports trial-based and session-based splitting.
    """
    
    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize splitter.
        
        Args:
            data_dir (str): Directory containing processed data
            output_dir (str): Directory to save splits
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self) -> Tuple[Dict, Dict]:
        """
        Load processed data for splitting.
        
        Returns:
            Tuple[Dict, Dict]: Past data and labels by subject
        """
        with open(self.data_dir / "past_by_subject_DE.pkl", "rb") as f:
            past_by_subject = pickle.load(f)
        with open(self.data_dir / "past_labels_by_subject_DE.pkl", "rb") as f:
            labels_by_subject = pickle.load(f)
        
        return past_by_subject, labels_by_subject
    
    def create_trial_based_splits(self, past_by_subject: Dict, labels_by_subject: Dict) -> Dict:
        """
        Create trial-based 3-fold cross-validation splits.
        
        Args:
            past_by_subject: Past segments by subject
            labels_by_subject: Labels by subject
            
        Returns:
            Dict: Fold information by subject
        """
        folds_by_subject = defaultdict(list)

        for subject_id in past_by_subject.keys():
            label_seq = labels_by_subject[subject_id]
            segs_per_trial = len(label_seq) // 15
            trial_to_indices = defaultdict(list)

            for trial_idx in range(15):
                start = trial_idx * segs_per_trial
                end = (trial_idx + 1) * segs_per_trial
                trial_to_indices[trial_idx] = list(range(start, end))

            fold_trials = [
                list(range(0, 5)),    # Trials 0-4
                list(range(5, 10)),   # Trials 5-9
                list(range(10, 15)),  # Trials 10-14
            ]

            for fold_idx, test_trials in enumerate(fold_trials):
                train_trials = [t for t in range(15) if t not in test_trials]

                train_indices = [i for t in train_trials for i in trial_to_indices[t]]
                test_indices = [i for t in test_trials for i in trial_to_indices[t]]

                folds_by_subject[subject_id].append({
                    'fold': fold_idx + 1,
                    'train_indices': train_indices,
                    'test_indices': test_indices
                })

        print("Trial-based 3-fold splits created for each subject.")
        return dict(folds_by_subject)
    
    def create_session_based_splits(self, past_by_subject: Dict, labels_by_subject: Dict) -> Dict:
        """
        Create session-based 3-fold cross-validation splits.
        
        Args:
            past_by_subject: Past segments by subject
            labels_by_subject: Labels by subject
            
        Returns:
            Dict: Fold information by subject
        """
        folds_by_subject_session = defaultdict(list)

        for subject_id in past_by_subject.keys():
            label_seq = labels_by_subject[subject_id]
            segs_per_trial = len(label_seq) // 15
            segs_per_session = segs_per_trial * 5
            session_to_indices = defaultdict(list)

            for session_idx in range(3):
                start = session_idx * segs_per_session
                end = (session_idx + 1) * segs_per_session
                session_to_indices[session_idx] = list(range(start, end))

            for fold_idx, test_session in enumerate(range(3)):
                train_sessions = [s for s in range(3) if s != test_session]

                train_indices = [i for s in train_sessions for i in session_to_indices[s]]
                test_indices = session_to_indices[test_session]

                folds_by_subject_session[subject_id].append({
                    'fold': fold_idx + 1,
                    'train_indices': train_indices,
                    'test_indices': test_indices
                })

        print("Session-based 3-fold splits created for each subject.")
        return dict(folds_by_subject_session)
    
    def save_splits(self, folds_by_subject: Dict, filename: str):
        """
        Save cross-validation splits to disk.
        
        Args:
            folds_by_subject: Fold information by subject
            filename: Output filename
        """
        filepath = self.output_dir / filename
        with open(filepath, "wb") as f:
            pickle.dump(folds_by_subject, f)
        
        print(f"Saved {filename} in: {self.output_dir}")
    
    def create_train_val_files(self, past_by_subject: Dict, labels_by_subject: Dict, 
                             folds_by_subject: Dict, split_name: str):
        """
        Create individual train/validation files for each subject and fold.
        
        Args:
            past_by_subject: Past segments by subject
            labels_by_subject: Labels by subject
            folds_by_subject: Fold information by subject
            split_name: Name of the split strategy (trial/session)
        """
        save_dir = self.output_dir / f"train_val_split_by_{split_name}"
        save_dir.mkdir(exist_ok=True)

        for subject_id in folds_by_subject:
            past = past_by_subject[subject_id]
            labels = labels_by_subject[subject_id]

            for fold_info in folds_by_subject[subject_id]:
                fold_id = fold_info['fold']
                train_idx = fold_info['train_indices']
                val_idx = fold_info['test_indices']

                X_train = [past[i] for i in train_idx]
                y_train = [labels[i] for i in train_idx]
                X_val = [past[i] for i in val_idx]
                y_val = [labels[i] for i in val_idx]

                with open(save_dir / f"train_subject{subject_id}_fold{fold_id}.pkl", "wb") as f:
                    pickle.dump((X_train, y_train), f)
                with open(save_dir / f"val_subject{subject_id}_fold{fold_id}.pkl", "wb") as f:
                    pickle.dump((X_val, y_val), f)

        print(f"All per-subject fold files saved to: {save_dir}")
    
    def create_all_splits(self):
        """
        Create all cross-validation splits.
        """
        print("Creating cross-validation splits...")
        
        # Load data
        past_by_subject, labels_by_subject = self.load_data()
        
        # Create trial-based splits
        trial_folds = self.create_trial_based_splits(past_by_subject, labels_by_subject)
        self.save_splits(trial_folds, "folds_by_subject_trial_DE.pkl")
        self.create_train_val_files(past_by_subject, labels_by_subject, trial_folds, "trial")
        
        # Create session-based splits
        session_folds = self.create_session_based_splits(past_by_subject, labels_by_subject)
        self.save_splits(session_folds, "folds_by_subject_session_DE.pkl")
        self.create_train_val_files(past_by_subject, labels_by_subject, session_folds, "session")
        
        print("Cross-validation splits completed!")
        
        return trial_folds, session_folds
