import argparse
import torch
import warnings
import multiprocessing
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.functional import interpolate
from typing import Union, Tuple, List
from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np
from time import time, sleep
# Use the same multi-threading setup as base trainer
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerRegression_mae import nnUNetTrainerRegression_mae
from nnunetv2.training.loss.mae import myMAE
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.dataloading.data_loader_regression import nnUNetDataLoader2D_Regression, nnUNetDataLoader3D_Regression
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.configuration import default_num_processes
from nnunetv2.utilities.helpers import dummy_context
from torch import autocast
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from torch import nn
from torch import distributed as dist


class nnUNetTrainerRegression_mae_deep(nnUNetTrainerRegression_mae):
    """
    nnU-Net trainer for image-to-image regression tasks using MAE loss with deep supervision.
    
    Inherits all functionality from nnUNetTrainerRegression_mae and adds:
    - Deep supervision with multi-scale loss computation
    - Manual target downsampling with linear interpolation for regression
    - All shared post-processing methods from base class
    
    Key features:
    - Uses MAE (L1) loss with DeepSupervisionWrapper
    - Enables deep supervision for better gradient flow
    - Uses multi-scale loss with exponential weighting
    - Supports optional trilinear upsampling to reduce artifacts
    - Uses custom data loaders for float32 targets
    - Disables mirroring for consistent translation
    - Uses validation transforms only (no augmentation)
    """
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = True  # Enable deep supervision for regression
        
        # Re-initialize loss with deep supervision wrapper
        self.loss = self._build_loss()
        if isinstance(self.loss, torch.nn.Module):
            self.loss.to(self.device)

    def initialize(self):
        """
        Override to set trilinear interpolation for regression predictions
        """
        # Call parent initialize first
        super().initialize()
        
        # Override resampling parameters for regression: use full trilinear interpolation
        # instead of mixed linear/nearest neighbor interpolation used for segmentation
        if hasattr(self.configuration_manager, 'configuration'):
            if 'resampling_fn_probabilities_kwargs' in self.configuration_manager.configuration:
                self.configuration_manager.configuration['resampling_fn_probabilities_kwargs']['order_z'] = 1
                self.print_to_log_file('Using full trilinear interpolation for regression predictions (order_z=1)')

    def _build_loss(self):
        """Build MAE (L1) loss function with deep supervision wrapper"""
        loss = myMAE()
        
        # Add deep supervision if enabled
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # DDP compatibility: set very low weight instead of 0
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            
            # Normalize weights so they sum to 1
            weights = weights / weights.sum()
            # Wrap loss with deep supervision
            loss = DeepSupervisionWrapper(loss, weights)
            
        return loss

    def _downsample_target_for_ds(self, target: torch.Tensor) -> List[torch.Tensor]:
        """
        Manually downsample target for deep supervision using linear interpolation (for regression)
        """
        if not self.enable_deep_supervision:
            return target
            
        ds_scales = self._get_deep_supervision_scales()
        if ds_scales is None:
            return target
            
        results = []
        for s in ds_scales:
            if not isinstance(s, (tuple, list)):
                s = [s] * (target.ndim - 2)  # exclude batch and channel dims
            else:
                assert len(s) == target.ndim - 2
                
            if all([i == 1 for i in s]):
                results.append(target)
            else:
                new_shape = [round(i * j) for i, j in zip(target.shape[2:], s)]  # skip batch and channel
                # Use linear interpolation for regression (continuous values)
                mode = 'trilinear' if target.ndim == 5 else 'bilinear'  # 3D vs 2D
                downsampled = interpolate(target, new_shape, mode=mode, align_corners=False)
                results.append(downsampled)
        return results

    def predict_logits_from_preprocessed_data(self, data: torch.Tensor, 
                                            apply_denormalization: bool = False) -> torch.Tensor:
        """
        Regression-specific prediction with deep supervision disabled for single-scale inference.
        
        Args:
            data: Preprocessed input data with 2 channels [source, target]
            apply_denormalization: If True, apply denormalization to return original intensity ranges
            
        Returns:
            Single-scale prediction tensor (optionally denormalized)
        """
        # Channel splitting: use only channel 0 (source modality) for network input
        input_data = data[:, 0:1, ...]  # Keep channel dimension, remove target channel
        
        # Temporarily disable deep supervision for inference to get single-scale output
        was_deep_supervision_enabled = self.enable_deep_supervision
        self.set_deep_supervision_enabled(False)
        
        try:
            # Get network prediction using parent class method (from nnUNetTrainer)
            # This bypasses the regression trainer's predict method to avoid double channel splitting
            prediction = super(nnUNetTrainerRegression_mae, self).predict_logits_from_preprocessed_data(input_data)
            
            # Handle case where network still returns multi-scale output despite deep supervision being disabled
            # This can happen if the network was compiled with deep supervision enabled
            if isinstance(prediction, (list, tuple)):
                # Take the highest resolution prediction (first element)
                prediction = prediction[0]
            
        finally:
            # Restore original deep supervision setting
            self.set_deep_supervision_enabled(was_deep_supervision_enabled)
        
        # Optional denormalization to original intensity ranges
        if apply_denormalization:
            # Convert to numpy for denormalization
            prediction_np = prediction.cpu().numpy()
            
            # Apply denormalization using existing method (channel_idx=1 for CT target)
            prediction_denormalized = self.denormalize_prediction(prediction_np, channel_idx=1)
            
            # Convert back to tensor
            prediction = torch.from_numpy(prediction_denormalized).to(prediction.device)
        
        return prediction

    def train_step(self, batch: dict) -> dict:
        """
        Training step with multi-channel input splitting and deep supervision
        Channel 0: CBCT (input to network)
        Channel 1: CT (target for regression)
        """
        data = torch.from_numpy(batch['data'])
        target = torch.from_numpy(batch['target'])

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Split multi-channel input: Channel 0 = CBCT input, Channel 1 = CT target
        input_data = data[:, 0:1, ...]  # CBCT channel (keep channel dimension)
        target_data = data[:, 1:2, ...]  # CT channel (keep channel dimension)
        
        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(input_data)
            # del input_data
            
            # Handle deep supervision by manually downsampling target
            if self.enable_deep_supervision:
                target_scales = self._downsample_target_for_ds(target_data)
                l = self.loss(output, target_scales)
            else:
                l = self.loss(output, target_data)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        """
        Validation step with multi-channel input splitting and deep supervision
        Channel 0: CBCT (input to network)
        Channel 1: CT (target for regression)
        """
        data = torch.from_numpy(batch['data'])
        target = torch.from_numpy(batch['target'])

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Split multi-channel input: Channel 0 = CBCT input, Channel 1 = CT target
        input_data = data[:, 0:1, ...]  # CBCT channel (keep channel dimension)
        target_data = data[:, 1:2, ...]  # CT channel (keep channel dimension)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(input_data)
            del input_data

            if self.enable_deep_supervision:
                target_scales = self._downsample_target_for_ds(target_data)
                l = self.loss(output, target_scales)
            else:
                l = self.loss(output, target_data)

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': 0, 'fp_hard': 0, 'fn_hard': 0}

    def perform_actual_validation(self, save_probabilities: bool = False):
        """
        Custom validation for regression that saves denormalized predictions
        (Uses inherited shared post-processing methods)
        """
        self.set_deep_supervision_enabled(False)
        self.network.eval()
        
        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                                   "encounter crashes in validation then this is because torch.compile forgets "
                                   "to trigger a recompilation of the model with deep supervision disabled. "
                                   "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                                   "validation with --val (exactly the same as before) and then it will work. "
                                   "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                                   "forward pass (where compile is triggered) already has deep supervision disabled. "
                                   "This is exactly what we need in perform_actual_validation")

        # Use nnUNetPredictor for sliding window inference
        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)

        validation_output_folder = join(self.output_folder, 'validation')
        maybe_mkdir_p(validation_output_folder)
        
        # Get validation keys
        _, val_keys = self.do_split()
        if self.is_ddp:
            last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1
            val_keys = val_keys[self.local_rank:: dist.get_world_size()]

        dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)

        for i, k in enumerate(dataset_val.identifiers):
            self.print_to_log_file(f"predicting {k}")
            data, _, seg_prev, properties = dataset_val.load_case(k)
            
            # Convert blosc2 to numpy and split channels
            data = data[:]  
            
            # Use only channel 0 (source modality) as input to network
            input_data = data[0:1, ...]  # Keep channel dimension
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                input_tensor = torch.from_numpy(input_data)
            
            self.print_to_log_file(f'{k}, input shape {input_tensor.shape}, rank {self.local_rank}')
            
            # Get prediction using sliding window
            prediction = predictor.predict_sliding_window_return_logits(input_tensor)
            prediction = prediction.cpu()
            
            # Use shared post-processing and saving methods (inherited from base class)
            output_file_truncated = join(validation_output_folder, k)
            processed_prediction = self._postprocess_regression_prediction(
                prediction, properties, apply_denormalization=True, channel_idx=1
            )
            self._save_regression_prediction(processed_prediction, properties, output_file_truncated)
            
            # Handle DDP barriers for large datasets
            if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                dist.barrier()
        
        if self.is_ddp:
            dist.barrier()
        
        # Re-enable deep supervision
        self.set_deep_supervision_enabled(True)
        
        if self.local_rank == 0:
            self.print_to_log_file("Regression validation complete - denormalized predictions saved", also_print_to_console=True)
            self.print_to_log_file(f"Predictions saved in: {validation_output_folder}", also_print_to_console=True)
            
            # Compute regression metrics (MAE, PSNR, SSIM) - inherited from base class
            self._compute_validation_metrics(validation_output_folder)
