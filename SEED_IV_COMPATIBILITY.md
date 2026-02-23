# SEED IV Dataset Compatibility

This repository is primarily designed for the SEED V dataset but can be easily adapted for SEED IV with minimal changes.

## Key Differences Between SEED V and SEED IV

### Dataset Specifications
- **SEED V**: 16 subjects, 3 sessions, 15 trials/session, 5 emotions (Disgust, Fear, Sad, Neutral, Happy)
- **SEED IV**: 15 subjects, 3 sessions, 24 trials/session, 4 emotions (Happy, Sad, Fear, Neutral)

### Required Modifications

#### 1. Configuration Changes
Update `configs/default.yaml`:

```yaml
# For SEED IV
data:
  emotions: 4  # Changed from 5
  subjects: 15  # Changed from 16
  trials_per_session: 24  # Changed from 15

# Update emotion mapping in data processing
```

#### 2. Data Processing
The `EEGDataProcessor` class in `src/data/preprocessing.py` needs minor updates:

```python
# In the create_predictive_pairs method, adjust for different trial structure
# SEED IV has 24 trials per session instead of 15
segs_per_trial = len(label_seq) // 24  # Changed from 15
```

#### 3. Cross-Validation Splits
Update the trial-based splits in `src/data/splits.py`:

```python
# For SEED IV with 24 trials per session
fold_trials = [
    list(range(0, 8)),    # Trials 0-7
    list(range(8, 16)),   # Trials 8-15
    list(range(16, 24)),  # Trials 16-23
]
```

#### 4. Emotion Label Mapping
Update the emotion label dictionary:

```python
# For SEED IV
label_dict = {0: 'Happy', 1: 'Sad', 2: 'Fear', 3: 'Neutral'}
```

## Migration Steps

1. **Update Configuration**: Modify `configs/default.yaml` with SEED IV specifications
2. **Update Data Processing**: Adjust trial count and emotion mapping
3. **Update Cross-Validation**: Modify trial split strategy for 24 trials
4. **Preprocess Data**: Run preprocessing with SEED IV data
5. **Train Models**: Use the same training pipeline
6. **Evaluate**: Adjust evaluation for 4-class classification

## Model Architecture Compatibility

The model architectures (`CNNPredictiveCodingDE_Attn` and `LightweightPredictiveCoding`) are fully compatible with SEED IV as they operate on the same DE feature format (62 channels, 5 frequency bands).

## Expected Performance

The predictive coding approach should work similarly well on SEED IV, with potential differences in:
- Overall accuracy due to different number of emotion classes
- Training time due to different dataset size
- Cross-validation performance patterns

## Validation

When migrating to SEED IV:
1. Verify data loading and preprocessing
2. Check cross-validation split correctness
3. Validate model input/output shapes
4. Monitor training for any convergence issues
5. Compare results with published SEED IV benchmarks

## File Structure for SEED IV

The same file structure can be used:
```
data/
├── raw/              # SEED IV raw DE features
├── processed/        # Processed SEED IV data
└── splits/          # SEED IV cross-validation splits
```

The scripts and modules remain unchanged except for the configuration and minor data processing adjustments mentioned above.
