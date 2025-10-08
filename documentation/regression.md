# Image-to-Image Translation with nnUNet Regression

This document describes the image-to-image translation functionality implemented in nnUNet for regression tasks, such as CBCT-to-CT translation.

## Overview

The regression trainer (`nnUNetTrainerRegression_mae`) extends nnUNet to perform image-to-image translation tasks using Mean Absolute Error (MAE) loss instead of the standard segmentation losses. This is particularly useful for medical image translation tasks like MR-to-CT, CBCT-to-CT, or synthetic image generation.

## Key Features

- **Multi-channel input handling**: Automatically splits input channels into source and target
- **MAE loss function**: Uses L1 loss for continuous value regression
- **Mask-guided cropping**: Leverages field-of-view masks for anatomical focus
- **Trilinear upsampling support**: Optional trilinear decoder to reduce artifacts
- **No data augmentation**: Uses validation transforms only for consistent translation
- **Disabled mirroring**: Ensures consistent translation results

## Architecture Components

### 1. Trainer: `nnUNetTrainerRegression_mae`

**Location**: `nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerRegression_mae.py`

**Key modifications**:
- Custom `train_step()` and `validation_step()` with channel splitting
- MAE loss function (`myMAE`)
- Optional trilinear decoder support via `--trilinear` flag
- Custom data loader integration for float32 targets
- Disabled deep supervision and mirroring

### 1b. Deep Supervision Trainer: `nnUNetTrainerRegression_mae_deep`

**Location**: `nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerRegression_mae_deep.py`

**Enhanced features**:
- All features of the base regression trainer
- **Enabled deep supervision** for improved gradient flow and training stability
- **Manual target downsampling** with linear interpolation (trilinear/bilinear) for regression
- **Multi-scale loss computation** with exponential weighting (1, 1/2, 1/4, ...)
- **Denormalized prediction export** - automatically saves predictions in original intensity ranges
- **Full trilinear interpolation** for continuous data (order_z=1 instead of nearest neighbor)
- **Custom validation with preprocessing reversal** - handles cropping, spacing, and orientation
- Automatic network multi-scale output handling via `DeepSupervisionWrapper`

### 2. Data Loaders: `nnUNetDataLoader2D_Regression` / `nnUNetDataLoader3D_Regression`

**Location**: `nnunetv2/training/dataloading/data_loader_regression.py`

**Key modifications**:
- Float32 target handling (instead of int16)
- Background padding with 0 (instead of -1)
- Maintains foreground-based cropping for anatomical focus

### 3. Loss Function: `myMAE`

**Location**: `nnunetv2/training/loss/mae.py`

Simple L1Loss wrapper for regression tasks.

## Data Format

### Multi-Channel Setup

The trainer expects data in multi-channel format where:
- **Channel 0**: Source modality (e.g., CBCT)
- **Channel 1**: Target modality (e.g., CT)
- **Segmentation mask**: Field-of-view or region-of-interest mask

### Dataset Structure

```
Dataset_XXX/
├── dataset.json              # Contains channel definitions
├── imagesTr/
│   ├── case_0000.nii.gz     # Channel 0: Source (CBCT)
│   ├── case_0001.nii.gz     # Channel 1: Target (CT)
│   └── ...
└── labelsTr/
    ├── case.nii.gz          # Field-of-view masks
    └── ...
```

### Dataset JSON Example

```json
{
    "labels": {
        "background": 0,
        "label_001": 1
    },
    "channel_names": {
        "0": "CBCT",
        "1": "CT_real"
    },
    "numTraining": 395,
    "file_ending": ".nii.gz"
}
```

## Usage

### Training

```bash
# Standard training with default decoder
nnUNetv2_train -d DATASET_ID -tr nnUNetTrainerRegression_mae

# Training with trilinear upsampling (reduces artifacts)
nnUNetv2_train -d DATASET_ID -tr nnUNetTrainerRegression_mae --trilinear

# Training with deep supervision for improved gradient flow
nnUNetv2_train -d DATASET_ID -tr nnUNetTrainerRegression_mae_deep

# Deep supervision + trilinear upsampling
nnUNetv2_train -d DATASET_ID -tr nnUNetTrainerRegression_mae_deep --trilinear
```

### Validation with Denormalized Predictions

```bash
# Run validation to generate denormalized predictions
nnUNetv2_train -d DATASET_ID -tr nnUNetTrainerRegression_mae_deep --val

# Predictions will be saved in: {output_folder}/validation/
# Format: Denormalized float32 NIfTI files in original intensity ranges
```

### Inference

```bash
# Standard regression trainer
nnUNetv2_predict -d DATASET_ID -i INPUT_DIR -o OUTPUT_DIR \
    -tr nnUNetTrainerRegression_mae -c 3d_fullres -f FOLD

# Deep supervision trainer (automatically uses single-scale output for inference)
nnUNetv2_predict -d DATASET_ID -i INPUT_DIR -o OUTPUT_DIR \
    -tr nnUNetTrainerRegression_mae_deep -c 3d_fullres -f FOLD
```

**Note**: Both trainers automatically bypass segmentation post-processing and export denormalized predictions as float32 NIfTI files in original intensity ranges (e.g., CT Hounsfield Units). Predictions use full trilinear interpolation (`order_z=1`) for smooth continuous data resampling.

## Implementation Details

### Channel Splitting

The trainer automatically handles multi-channel input splitting in both training and validation:

```python
# Split multi-channel input
input_data = data[:, 0:1, ...]   # Source modality (CBCT)
target_data = data[:, 1:2, ...]  # Target modality (CT)

# Forward pass
output = self.network(input_data)
loss = self.loss(output, target_data)
```

### Mask-Guided Cropping

The regression data loaders use foreground-based cropping guided by the field-of-view mask:
- Samples patches around anatomically relevant regions
- Improves training stability and focus on important structures
- Leverages standard nnUNet preprocessing pipeline

### Denormalized Prediction Export (Deep Supervision Trainer)

The deep supervision trainer (`nnUNetTrainerRegression_mae_deep`) automatically exports denormalized predictions during validation:

**Key features**:
- **Automatic denormalization**: Reverses preprocessing normalization back to original intensity ranges
- **Full trilinear interpolation**: Uses `order_z=1` for proper continuous data resampling
- **Complete preprocessing reversal**: Handles cropping, spacing, and orientation restoration
- **Original image space**: Predictions saved in original dimensions and coordinate system

**Supported normalization schemes**:
- `ZScoreNormalization`: Reverses z-score normalization using stored mean/std
- `CTNormalization`: Reverses CT-specific normalization with clipping parameters
- `NoNormalization`: No denormalization applied
- Automatic detection from dataset preprocessing parameters

**Output format**:
- Saved as continuous float32 NIfTI files (`.nii.gz`)
- Located in `{output_folder}/validation/` during training validation
- Predictions maintain original CT intensity ranges (e.g., -1000 to 3000 HU)
- Proper affine matrices for spatial coordinate mapping

### Training Configuration

#### Base Trainer (`nnUNetTrainerRegression_mae`)
- **Epochs**: 1000
- **Iterations per epoch**: 250
- **Loss function**: MAE (L1 Loss)
- **Deep supervision**: Disabled
- **Data augmentation**: Disabled (uses validation transforms)
- **Mirroring**: Disabled for consistent translation

#### Deep Supervision Trainer (`nnUNetTrainerRegression_mae_deep`)
- **Epochs**: 1000
- **Iterations per epoch**: 250
- **Loss function**: MAE (L1 Loss) with `DeepSupervisionWrapper`
- **Deep supervision**: Enabled with linear interpolation downsampling
- **Multi-scale weighting**: Exponential decay (1.0, 0.5, 0.25, 0.125, ...)
- **Target downsampling**: Manual in-training downsampling with trilinear/bilinear interpolation
- **Interpolation**: Full trilinear interpolation for continuous data (order_z=1)
- **Prediction export**: Automatic denormalization and preprocessing reversal
- **Data augmentation**: Disabled (uses validation transforms)
- **Mirroring**: Disabled for consistent translation

## Dependencies

The implementation requires a fork of `dynamic-network-architectures` for trilinear upsampling support:

```toml
# pyproject.toml
dependencies = [
    "dynamic-network-architectures @ git+https://github.com/Phyrise/dynamic-network-architectures.git@main",
    # ... other dependencies
]
```

## Performance Considerations

### Training Tips

1. **Use mask-guided cropping**: Ensure your field-of-view masks cover anatomically relevant regions
2. **Monitor MAE loss**: Lower MAE indicates better translation quality
3. **Try trilinear upsampling**: May reduce checkerboard artifacts in outputs
4. **Adjust patch size**: Consider anatomical structures when setting patch sizes
5. **Choose trainer variant**:
   - Use `nnUNetTrainerRegression_mae` for baseline training
   - Use `nnUNetTrainerRegression_mae_deep` for improved gradient flow and potentially better results
6. **Deep supervision benefits**: The deep supervision trainer may provide:
   - Better training stability
   - Improved gradient flow through all decoder layers
   - Potentially better translation quality
   - Faster convergence in some cases

### Memory Usage

- Float32 targets increase memory usage compared to segmentation
- Consider reducing batch size if encountering OOM errors
- Multi-channel data doubles input memory requirements
- **Deep supervision trainer**: Slightly higher memory usage due to:
  - Multi-scale network outputs during training
  - Multiple target downsampling operations
  - Additional loss computations for each scale

## Validation and Metrics

The regression trainers provide comprehensive validation metrics for evaluating translation quality. Both trainers automatically compute and save detailed metrics during validation.

### Online Metrics (During Training)
- **Validation MAE loss**: Monitored every epoch for training progress

### Automatic Validation Metrics 
Both regression trainers now automatically compute comprehensive metrics during validation:

- **MAE (Mean Absolute Error)**: Mean absolute difference between predicted and ground truth values
- **PSNR (Peak Signal-to-Noise Ratio)**: Image quality metric in decibels (higher is better)  
- **SSIM (Structural Similarity Index)**: Measures structural similarity between images (0-1, higher is better)
- **MSE/RMSE**: Mean Squared Error and Root Mean Squared Error for reference

**Automatic computation features**:
- Metrics computed on **denormalized predictions** in original intensity ranges (e.g., CT Hounsfield Units)
- Results saved as `validation_summary.json` in each fold's validation folder
- Per-case metrics and summary statistics included
- Multiprocessing support for efficient computation
- Compatible with both 2D and 3D data

### Validation Summary JSON Structure
```json
{
  "metric_per_case": [
    {
      "reference_file": "/path/to/gt/case_001.nii.gz",
      "prediction_file": "/path/to/pred/case_001.nii.gz", 
      "MAE": 45.23,
      "PSNR": 28.15,
      "SSIM": 0.85,
      "MSE": 2156.7,
      "RMSE": 46.44
    }
  ],
  "mean": {
    "MAE": 42.18,
    "PSNR": 29.34,
    "SSIM": 0.87,
    "MSE": 2045.3, 
    "RMSE": 45.22
  },
  "num_cases": 25,
  "valid_cases": {
    "MAE": 25,
    "PSNR": 25,
    "SSIM": 25,
    "MSE": 25,
    "RMSE": 25
  }
}
```

### Cross-Validation Results
The cross-validation accumulation system automatically detects regression trainers and:
- Computes combined metrics across all folds  
- Uses ground truth images (not segmentation labels) for comparison
- Saves comprehensive summary statistics for the entire cross-validation

### Denormalized Prediction Export
Both regression trainers automatically save denormalized predictions during validation:
- **Location**: `{output_folder}/validation/`
- **Format**: Float32 NIfTI files with original intensity ranges
- **Timing**: Generated during final validation or when using `--val` flag
- **Quality**: Ready for direct comparison with ground truth images

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all regression components are properly imported
2. **Memory issues**: Reduce batch size or patch size
3. **Channel mismatch**: Verify dataset has exactly 2 channels (source + target)
4. **Preprocessing errors**: Check that both channels have valid intensity properties
5. **Denormalization issues**: Check normalization scheme in plans.json matches expected type
6. **Prediction export failures**: Verify output folder permissions and disk space
7. **Metrics computation failures**: 
   - Ensure scikit-image is installed for PSNR/SSIM computation
   - Check that ground truth folder exists and contains matching files
   - Verify file formats are compatible (same extensions, readable by SimpleITK)
8. **Ground truth folder not found**: Check that `imagesTr` folder exists with ground truth images (channel 1 contains target data)

### Debugging

Enable verbose logging to monitor:
- Channel splitting in train/validation steps
- Loss computation
- Data loader behavior
- Memory usage
- Denormalization parameters and applied schemes
- Prediction export progress and final intensity ranges
- Preprocessing reversal steps (resampling, cropping, transpose)
- **Metrics computation**: Ground truth folder detection, file matching, and metric calculation progress
- **Validation summary creation**: JSON file generation and content verification

## Example Use Cases

### CBCT-to-CT Translation
- **Source**: Cone-beam CT (lower quality, faster acquisition)
- **Target**: High-quality CT
- **Applications**: Treatment planning, dose calculation

### MR-to-CT Synthesis
- **Source**: MR images
- **Target**: Synthetic CT
- **Applications**: MR-only treatment planning, PET attenuation correction

### Image Enhancement
- **Source**: Low-dose CT
- **Target**: Standard-dose CT
- **Applications**: Dose reduction while maintaining image quality

## References

- Longuefosse, A., et al. (2024). "Adapted nnU-Net: A Robust Baseline for Cross-Modality Synthesis and Medical Image Inpainting." SASHIMI Workshop.
- Isensee, F., et al. (2021). "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nature Methods.