#!/usr/bin/env python3
"""
Data preprocessing script for SEED V dataset.
Processes raw DE features and creates cross-validation splits.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.preprocessing import EEGDataProcessor
from src.data.splits import CrossValidationSplitter
from src.utils.config import Config

def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Preprocess SEED V EEG data")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, 
                       help="Override data directory from config")
    parser.add_argument("--output_dir", type=str,
                       help="Override output directory from config")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Get paths
    data_dir = args.data_dir or config.get('paths.data_dir')
    output_dir = args.output_dir or config.get('paths.output_dir')
    
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process data
    processor = EEGDataProcessor(data_dir, output_dir)
    past_data, future_data, past_labels, future_labels = processor.process_data()
    
    # Create cross-validation splits
    splitter = CrossValidationSplitter(output_dir, output_dir)
    trial_folds, session_folds = splitter.create_all_splits()
    
    print("Preprocessing completed successfully!")
    print(f"Processed data saved to: {output_dir}")

if __name__ == "__main__":
    main()
