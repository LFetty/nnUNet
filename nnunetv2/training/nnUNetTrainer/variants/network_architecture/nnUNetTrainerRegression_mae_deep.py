import argparse
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.functional import interpolate
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
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.dataloading.data_loader_regression import nnUNetDataLoader2D_Regression, nnUNetDataLoader3D_Regression
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.collate_outputs import collate_outputs
from torch import autocast
from batchgenerators.utilities.file_and_folder_operations import join
from torch import nn


class nnUNetTrainerRegression_mae_deep(nnUNetTrainer):
    """
    nnU-Net trainer for image-to-image regression tasks using MAE loss with deep supervision.
    
    Key features:
    - Uses MAE (L1) loss instead of Dice/Cross-entropy
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


# Add dummy context for compatibility
class dummy_context:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass