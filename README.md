# Predictive Coding for EEG-based Emotion Recognition

This repository implements subject-dependent predictive coding approaches for EEG-based emotion recognition using the SEED V dataset. The same methodology has been applied to SEED IV dataset with similar architectures.

## Overview

The project implements predictive coding models that learn to predict future EEG segments from past segments, using self-supervised learning to extract meaningful representations for emotion recognition tasks.

### Key Features

- **Predictive Coding Framework**: Models learn to predict future EEG from past segments
- **Attention Mechanisms**: Band and channel attention for improved feature learning
- **Multiple Architectures**: CNN-based models with attention mechanisms
- **Cross-validation**: Trial-based and session-based 3-fold cross-validation
- **SEED V & IV Compatible**: Designed for SEED V dataset, with similar approaches used for SEED IV

## Datasets

- **SEED V**: 16 subjects, 3 sessions per subject, 15 trials per session
- **EEG Features**: Differential Entropy (DE) features, 62 channels, 5 frequency bands
- **Emotions**: 5 classes - Disgust, Fear, Sad, Neutral, Happy
- **Data Structure**: Sliding window segmentation with past-future pairs

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py      # Data loading and preprocessing
│   │   ├── dataset.py           # Dataset classes
│   │   └── splits.py            # Cross-validation splits
│   ├── models/
│   │   ├── __init__.py
│   │   ├── attention.py         # Attention mechanisms
│   │   ├── predictive_coding.py # Main model architectures
│   │   └── base.py              # Base model classes
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Training loop
│   │   └── utils.py             # Training utilities
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # Configuration management
│       └── seed.py              # Random seed setting
├── configs/
│   └── default.yaml             # Default configuration
├── scripts/
│   ├── preprocess_data.py       # Data preprocessing script
│   ├── train_model.py           # Training script
│   └── evaluate_model.py        # Evaluation script
├── data/
│   ├── raw/                     # Raw EEG data (empty placeholder)
│   ├── processed/               # Processed DE features (empty placeholder)
│   └── splits/                  # Cross-validation splits (empty placeholder)
├── models/                      # Trained model checkpoints
├── logs/                        # Training logs
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd predictive-coding-eeg-emotion
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing

```bash
python scripts/preprocess_data.py --data_dir /path/to/EEG_DE_features --output_dir /path/to/processed
```

### Training

```bash
python scripts/train_model.py --config configs/default.yaml
```

### Evaluation

```bash
python scripts/evaluate_model.py --model_path models/best_model.pt --data_dir /path/to/data
```

## Model Architecture

The main model (`CNNPredictiveCodingDE_Attn`) consists of:

1. **Attention Mechanisms**:
   - Band Attention: Learns importance of different frequency bands
   - Channel Attention: Learns importance of different EEG channels

2. **Encoder-Decoder Architecture**:
   - Encoder: Convolutional layers with batch normalization and pooling
   - Bottleneck: High-level feature representation
   - Decoder: Transpose convolution layers for reconstruction

3. **Predictive Coding**: Model learns to predict future EEG segments from past segments

## Cross-Validation

Two cross-validation strategies are implemented:

1. **Trial-based 3-fold**: Splits trials within each session
2. **Session-based 3-fold**: Uses entire sessions for validation

## Results

The model achieves competitive performance on SEED V emotion recognition task. Detailed results can be found in the training logs.

## SEED IV Compatibility

The same methodology has been successfully applied to the SEED IV dataset with minimal modifications:
- Adjusted for 4 emotion classes instead of 5
- Compatible with SEED IV's data structure and preprocessing pipeline

## Citation

If you use this code in your research, please cite:

```
[Add your citation here]
```

## License

[Add your license information here]

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Contact

[Add your contact information here]
