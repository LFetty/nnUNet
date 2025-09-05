import argparse
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Union, Tuple, List
from batchgenerators.transforms.abstract_transforms import AbstractTransform
import numpy as np
from time import time
# Use the same multi-threading setup as base trainer
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.mae import myMAE
from nnunetv2.training.dataloading.data_loader_regression import nnUNetDataLoader2D_Regression, nnUNetDataLoader3D_Regression
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.collate_outputs import collate_outputs
from torch import autocast
from batchgenerators.utilities.file_and_folder_operations import join
from torch import nn


class nnUNetTrainerRegression_mae(nnUNetTrainer):
    """
    nnU-Net trainer for image-to-image regression tasks using MAE loss.
    
    Key features:
    - Uses MAE (L1) loss instead of Dice/Cross-entropy
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
        self.enable_deep_supervision = False
        self.num_iterations_per_epoch = 250
        self.num_epochs = 1000
        
        # Initialize MAE loss
        self.loss = self._build_loss()
        
        # Parse command line arguments for trilinear support
        self.decoder_type = "standard"  # default
        self._parse_decoder_args()

    def initialize(self):
        """
        Override to use 1 input channel for network (since we split channels manually)
        """
        if not self.was_initialized:
            # DDP batch size and oversampling can differ between workers and needs adaptation
            self._set_batch_size_and_oversample()

            self.num_input_channels = 1  # Force 1 input channel for regression
            self.num_output_channels = 1  # Force 1 output channel for regression

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.num_output_channels,
                self.enable_deep_supervision
            ).to(self.device)
            
            # Skip torch.compile for regression to avoid shape mismatch issues
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)
            
            # optimizer, lr_scheduler and loss are initialized here
            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])
            
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                             "That should not happen.")

    def _parse_decoder_args(self):
        """Parse command line arguments to check for trilinear decoder option"""
        try:
            import sys
            if '--trilinear' in sys.argv:
                self.decoder_type = "trilinear"
                print(f"Using trilinear decoder for reduced upsampling artifacts")
        except:
            # If argument parsing fails, use default
            pass

    def _build_loss(self):
        """Build MAE (L1) loss function"""
        loss = myMAE()
        return loss

    def build_network_architecture(self, architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        Build network architecture with optional trilinear decoder support
        """
        # Add decoder_type to architecture kwargs if supported
        if self.decoder_type != "standard":
            arch_init_kwargs = arch_init_kwargs.copy()
            arch_init_kwargs['decoder_type'] = self.decoder_type
            print(f"Building network with {self.decoder_type} decoder")
        
        return get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision)

    @staticmethod
    def get_training_transforms(patch_size: Union[np.ndarray, Tuple[int]],
                                rotation_for_DA: dict,
                                deep_supervision_scales: Union[List, Tuple, None],
                                mirror_axes: Tuple[int, ...],
                                do_dummy_2d_data_aug: bool,
                                order_resampling_data: int = 1,
                                order_resampling_seg: int = 0,
                                border_val_seg: int = -1,
                                use_mask_for_norm: List[bool] = None,
                                is_cascaded: bool = False,
                                foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                ignore_label: int = None) -> AbstractTransform:
        """
        Use validation transforms only - no augmentation for consistent translation
        """
        return nnUNetTrainer.get_validation_transforms(deep_supervision_scales, is_cascaded, foreground_labels,
                                                       regions, ignore_label)

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """
        Disable mirroring for consistent image translation results
        """
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def get_dataloaders(self):
        """
        Get data loaders using regression-specific loaders that handle float32 targets
        """
        if self.dataset_class is None:
            from nnunetv2.utilities.dataset_name_id_conversion import find_candidate_datasets
            from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        # (
        #     rotation_for_DA,
        #     do_dummy_2d_data_aug,
        #     initial_patch_size,
        #     mirror_axes,
        # ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # # Use validation transforms for both training and validation to avoid augmentation issues
        # # but still get proper tensor conversion
        # tr_transforms = self.get_validation_transforms(deep_supervision_scales,
        #                                                is_cascaded=self.is_cascaded,
        #                                                foreground_labels=self.label_manager.foreground_labels,
        #                                                regions=self.label_manager.foreground_regions if
        #                                                self.label_manager.has_regions else None,
        #                                                ignore_label=self.label_manager.ignore_label)

        # # validation pipeline  
        # val_transforms = self.get_validation_transforms(deep_supervision_scales,
        #                                                 is_cascaded=self.is_cascaded,
        #                                                 foreground_labels=self.label_manager.foreground_labels,
        #                                                 regions=self.label_manager.foreground_regions if
        #                                                 self.label_manager.has_regions else None,
        #                                                 ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        # Determine if we need 2D or 3D based on patch size
        dim = len(patch_size)
        
        dl_tr = nnUNetDataLoader(dataset_tr, self.batch_size,
                                    self.configuration_manager.patch_size,
                                    self.configuration_manager.patch_size,
                                    self.label_manager,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    sampling_probabilities=None, pad_sides=None, #transforms=tr_transforms,
                                    probabilistic_oversampling=self.probabilistic_oversampling)
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size,
                                    self.configuration_manager.patch_size,
                                    self.configuration_manager.patch_size,
                                    self.label_manager,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    sampling_probabilities=None, pad_sides=None, #transforms=val_transforms,
                                    probabilistic_oversampling=self.probabilistic_oversampling)
        
        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        
        # Initialize the generators
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def train_step(self, batch: dict) -> dict:
        """
        Training step with multi-channel input splitting
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
            # Use MAE loss between network output and target CT
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
        Validation step with multi-channel input splitting
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
            # Use MAE loss between network output and target CT
            l = self.loss(output, target_data)

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': 0, 'fp_hard': 0, 'fn_hard': 0}

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        """
        Custom validation epoch end handling for regression metrics
        """
        outputs_collated = collate_outputs(val_outputs)
        
        loss_here = np.mean(outputs_collated['loss'])

        self.logger.log('val_losses', loss_here, self.current_epoch)

    def on_epoch_end(self):
        """
        Custom epoch end handling optimized for regression training
        """
        # Log the end time of the epoch
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        # Logging train and validation loss
        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        
        # Log the duration of the epoch
        epoch_duration = self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1]
        self.print_to_log_file(f"Epoch time: {np.round(epoch_duration, decimals=2)} s")

        # Checkpoint handling for best and periodic saves
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        best_metric = 'val_losses'  # Use validation loss as best metric for regression
        if self._best_ema is None or self.logger.my_fantastic_logging[best_metric][-1] < self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging[best_metric][-1]
            self.print_to_log_file(f"Yayy! New best EMA MAE: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        # Increment the epoch counter
        self.current_epoch += 1

    def denormalize_prediction(self, prediction: np.ndarray, channel_idx: int = 1) -> np.ndarray:
        """
        Denormalize predictions back to original intensity range
        
        Args:
            prediction: Normalized prediction array
            channel_idx: Channel index for normalization parameters (1 for CT target channel)
        
        Returns:
            Denormalized prediction in original intensity range
        """
        # Get normalization info from plans
        normalization_schemes = self.configuration_manager.normalization_schemes
        intensity_props = self.plans_manager.foreground_intensity_properties_per_channel
        
        if channel_idx >= len(normalization_schemes):
            self.print_to_log_file(f'Warning: channel_idx {channel_idx} >= number of channels, using channel 0 for denormalization')
            channel_idx = 0
            
        norm_scheme = normalization_schemes[channel_idx]
        props = intensity_props[str(channel_idx)]
        
        self.print_to_log_file(f'Denormalizing with scheme: {norm_scheme}, channel: {channel_idx}')
        
        if norm_scheme == 'ZScoreNormalization':
            # Reverse: x_orig = (x_norm * std) + mean  
            mean = props['mean']
            std = props['std']
            prediction = prediction * std + mean
            self.print_to_log_file(f'Applied ZScore denormalization: mean={mean:.2f}, std={std:.2f}')
            
        elif norm_scheme == 'CTNormalization':
            # Reverse: x_orig = (x_norm * std) + mean
            mean = props['mean']
            std = props['std']  
            prediction = prediction * std + mean
            self.print_to_log_file(f'Applied CT denormalization: mean={mean:.2f}, std={std:.2f}')
            
        elif norm_scheme == 'RescaleTo01Normalization':
            # This would need the original min/max, but those aren't stored in intensity properties
            # For now, just return as-is and warn
            self.print_to_log_file('Warning: RescaleTo01Normalization denormalization not implemented, returning normalized values')
            
        elif norm_scheme == 'NoNormalization':
            # No denormalization needed
            self.print_to_log_file('No denormalization applied (NoNormalization scheme)')
            
        else:
            self.print_to_log_file(f'Warning: Unknown normalization scheme {norm_scheme}, returning normalized values')
            
        return prediction

    def export_regression_prediction(self, prediction: Union[torch.Tensor, np.ndarray], 
                                   properties_dict: dict, output_file_truncated: str,
                                   channel_idx: int = 1) -> None:
        """
        Export regression prediction with denormalization and preprocessing reversal
        
        Args:
            prediction: Network prediction (logits/continuous values)
            properties_dict: Case properties for preprocessing reversal
            output_file_truncated: Output filename without extension
            channel_idx: Channel index for denormalization (1 for CT target)
        """
        import nibabel as nib
        from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
        
        # Convert to numpy if needed
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.cpu().numpy()
        
        # Remove batch dimension if present and keep only single channel
        if prediction.ndim == 5:  # [B, C, H, W, D]
            prediction = prediction[0, 0]  # Take first batch, first channel
        elif prediction.ndim == 4:  # [C, H, W, D] or [B, H, W, D]
            if prediction.shape[0] > 1:  # Assume first dim is channels if > 1
                prediction = prediction[0]  # Take first channel
            else:
                prediction = prediction[0]  # Remove batch dimension
        elif prediction.ndim == 3:  # Already [H, W, D]
            pass
        else:
            raise ValueError(f"Unexpected prediction shape: {prediction.shape}")
        
        self.print_to_log_file(f'Processing prediction with shape: {prediction.shape}')
        
        # Denormalize prediction
        prediction_denormalized = self.denormalize_prediction(prediction, channel_idx)
        
        # Revert preprocessing steps using existing nnUNet infrastructure
        # 1. Resample back to original spacing
        spacing_transposed = [properties_dict['spacing'][i] for i in self.plans_manager.transpose_forward]
        current_spacing = self.configuration_manager.spacing if \
            len(self.configuration_manager.spacing) == \
            len(properties_dict['shape_after_cropping_and_before_resampling']) else \
            [spacing_transposed[0], *self.configuration_manager.spacing]
        target_spacing = [properties_dict['spacing'][i] for i in self.plans_manager.transpose_forward]
        
        if 'shape_after_cropping_and_before_resampling' in properties_dict:
            # Add channel dimension for resampling function
            prediction_with_channel = prediction_denormalized[None]  # Add channel dim
            prediction_resampled = self.configuration_manager.resampling_fn_probabilities(
                prediction_with_channel,
                properties_dict['shape_after_cropping_and_before_resampling'],
                current_spacing,
                target_spacing
            )[0]  # Remove channel dimension
            self.print_to_log_file(f'Resampled from {prediction_denormalized.shape} to {prediction_resampled.shape}')
        else:
            prediction_resampled = prediction_denormalized
            self.print_to_log_file('No resampling needed')
        
        # 2. Revert cropping to original image size
        if 'bbox_used_for_cropping' in properties_dict:
            prediction_full = np.zeros(properties_dict['shape_before_cropping'], dtype=np.float32)
            prediction_full = insert_crop_into_image(prediction_full, prediction_resampled, 
                                                    properties_dict['bbox_used_for_cropping'])
            self.print_to_log_file(f'Reverted cropping to shape: {prediction_full.shape}')
        else:
            prediction_full = prediction_resampled.astype(np.float32)
            self.print_to_log_file('No cropping reversal needed')
        
        # 3. Revert transpose if needed
        if hasattr(self.plans_manager, 'transpose_backward'):
            prediction_final = prediction_full.transpose(self.plans_manager.transpose_backward)
            self.print_to_log_file(f'Reverted transpose to final shape: {prediction_final.shape}')
        else:
            prediction_final = prediction_full
            self.print_to_log_file('No transpose reversal needed')
        
        # Save using SimpleITK to preserve float32 values and correct orientation
        import SimpleITK as sitk
        
        output_filename = f'{output_file_truncated}{self.dataset_json["file_ending"]}'
        
        # Check if it's 2D (remove singleton first dimension if present)
        output_dimension = len(properties_dict['sitk_stuff']['spacing'])
        if output_dimension == 2 and prediction_final.ndim == 3:
            prediction_final = prediction_final[0]
        
        # Create SimpleITK image with float32 precision (no conversion to uint8/uint16)
        itk_image = sitk.GetImageFromArray(prediction_final.astype(np.float32, copy=False))
        itk_image.SetSpacing(properties_dict['sitk_stuff']['spacing'])
        itk_image.SetOrigin(properties_dict['sitk_stuff']['origin'])
        itk_image.SetDirection(properties_dict['sitk_stuff']['direction'])
        
        # Write image preserving float32 values
        sitk.WriteImage(itk_image, output_filename, True)
        
        self.print_to_log_file(f'Saved denormalized prediction to: {output_filename}')
        self.print_to_log_file(f'Final prediction range: [{prediction_final.min():.2f}, {prediction_final.max():.2f}]')

    def perform_actual_validation(self, save_probabilities: bool = False):
        """
        Custom validation for regression that saves predictions with channel splitting
        """
        import warnings
        import torch.distributed as dist
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
        
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
            
            # Export denormalized prediction using our custom function
            output_file_truncated = join(validation_output_folder, k)
            self.export_regression_prediction(prediction, properties, output_file_truncated, channel_idx=1)
            
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


# Add dummy context for compatibility
class dummy_context:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass