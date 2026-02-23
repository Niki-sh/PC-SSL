import os
import glob
import pickle
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

class EEGDataProcessor:
    """
    EEG Data Processor for SEED V dataset.
    Handles loading and preprocessing of DE features.
    """
    
    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize data processor.
        
        Args:
            data_dir (str): Directory containing raw DE features
            output_dir (str): Directory to save processed data
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_raw_data(self) -> Tuple[Dict[int, Any], Dict[int, Any]]:
        """
        Load raw DE features from NPZ files.
        
        Returns:
            Tuple[Dict[int, Any], Dict[int, Any]]: Raw data and labels by subject
        """
        npz_files = sorted(glob.glob(f"{self.data_dir}/*.npz"))
        
        all_subjects_data = []
        for npz_path in npz_files:
            with np.load(npz_path, allow_pickle=True) as npz:
                raw_data = pickle.loads(npz['data'].tobytes())
                raw_labels = pickle.loads(npz['label'].tobytes())
                all_subjects_data.append((raw_data, raw_labels))
        
        return all_subjects_data
    
    def create_predictive_pairs(self, all_subjects_data: List[Tuple[Dict, Dict]]) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Create past-future pairs for predictive coding.
        
        Args:
            all_subjects_data: List of (data, labels) tuples for each subject
            
        Returns:
            Tuple containing past data, future data, past labels, future labels by subject
        """
        npz_files = sorted(glob.glob(f"{self.data_dir}/*.npz"))
        
        past_by_subject = defaultdict(list)
        future_by_subject = defaultdict(list)
        past_labels_by_subject = defaultdict(list)
        future_labels_by_subject = defaultdict(list)

        for npz_path in npz_files:
            subject_id = int(os.path.basename(npz_path).split('_')[0])
            print(f"Processing subject {subject_id} ...")

            with np.load(npz_path, allow_pickle=True) as npz:
                raw_data = pickle.loads(npz['data'].tobytes())
                raw_labels = pickle.loads(npz['label'].tobytes())

                for trial_id, segments_flat in raw_data.items():
                    segments_reshaped = segments_flat.reshape(-1, 62, 5)
                    labels = raw_labels[trial_id]

                    for i in range(len(segments_reshaped) - 1):
                        past_by_subject[subject_id].append(segments_reshaped[i])
                        future_by_subject[subject_id].append(segments_reshaped[i+1])
                        past_labels_by_subject[subject_id].append(labels[i])
                        future_labels_by_subject[subject_id].append(labels[i+1])

        print("Data processed for all subjects.")
        
        return (past_by_subject, future_by_subject, 
                past_labels_by_subject, future_labels_by_subject)
    
    def save_processed_data(self, past_by_subject: Dict, future_by_subject: Dict,
                           past_labels_by_subject: Dict, future_labels_by_subject: Dict):
        """
        Save processed data to disk.
        
        Args:
            past_by_subject: Past segments by subject
            future_by_subject: Future segments by subject
            past_labels_by_subject: Past labels by subject
            future_labels_by_subject: Future labels by subject
        """
        files_to_save = [
            ("past_by_subject_DE.pkl", past_by_subject),
            ("future_by_subject_DE.pkl", future_by_subject),
            ("past_labels_by_subject_DE.pkl", past_labels_by_subject),
            ("future_labels_by_subject_DE.pkl", future_labels_by_subject)
        ]
        
        for filename, data in files_to_save:
            filepath = self.output_dir / filename
            with open(filepath, "wb") as f:
                pickle.dump(dict(data), f)
        
        print(f"Saved all files in: {self.output_dir}")
    
    def process_data(self):
        """
        Complete data processing pipeline.
        """
        print("Starting data processing...")
        
        # Load raw data
        all_subjects_data = self.load_raw_data()
        
        # Create predictive pairs
        past_data, future_data, past_labels, future_labels = self.create_predictive_pairs(all_subjects_data)
        
        # Save processed data
        self.save_processed_data(past_data, future_data, past_labels, future_labels)
        
        print("Data processing completed!")
        
        return past_data, future_data, past_labels, future_labels
