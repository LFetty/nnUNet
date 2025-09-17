from typing import Union, List

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from batchgenerators.utilities.file_and_folder_operations import load_json, save_pickle

from nnunetv2.configuration import default_num_processes
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


def convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits: Union[torch.Tensor, np.ndarray],
                                                                plans_manager: PlansManager,
                                                                configuration_manager: ConfigurationManager,
                                                                label_manager: LabelManager,
                                                                properties_dict: dict,
                                                                return_probabilities: bool = False,
                                                                num_threads_torch: int = default_num_processes):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    # resample to original shape
    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    predicted_logits = configuration_manager.resampling_fn_probabilities(predicted_logits,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            [properties_dict['spacing'][i] for i in plans_manager.transpose_forward])
    # return value of resampling_fn_probabilities can be ndarray or Tensor but that does not matter because
    # apply_inference_nonlin will convert to torch
    if not return_probabilities:
        # this has a faster computation path because we can skip the softmax in regular (not region based) training
        segmentation = label_manager.convert_logits_to_segmentation(predicted_logits)
    else:
        predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
        segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)
    del predicted_logits

    # put segmentation in bbox (revert cropping)
    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'],
                                              dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16)
    segmentation_reverted_cropping = insert_crop_into_image(segmentation_reverted_cropping, segmentation, properties_dict['bbox_used_for_cropping'])
    del segmentation

    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation_reverted_cropping, torch.Tensor):
        segmentation_reverted_cropping = segmentation_reverted_cropping.cpu().numpy()

    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)
    if return_probabilities:
        # revert cropping
        predicted_probabilities = label_manager.revert_cropping_on_probabilities(predicted_probabilities,
                                                                                 properties_dict[
                                                                                     'bbox_used_for_cropping'],
                                                                                 properties_dict[
                                                                                     'shape_before_cropping'])
        predicted_probabilities = predicted_probabilities.cpu().numpy()
        # revert transpose
        predicted_probabilities = predicted_probabilities.transpose([0] + [i + 1 for i in
                                                                           plans_manager.transpose_backward])
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping, predicted_probabilities
    else:
        torch.set_num_threads(old_threads)
        return segmentation_reverted_cropping


def export_prediction_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor], properties_dict: dict,
                                  configuration_manager: ConfigurationManager,
                                  plans_manager: PlansManager,
                                  dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
                                  save_probabilities: bool = False,
                                  num_threads_torch: int = default_num_processes,
                                  is_regression: bool = False):
    # Use regression export for regression trainers
    if is_regression:
        export_regression_prediction_from_logits(
            predicted_array_or_file, properties_dict, configuration_manager, plans_manager,
            dataset_json_dict_or_file, output_file_truncated, num_threads_torch
        )
        return

    # Original segmentation export logic
    # if isinstance(predicted_array_or_file, str):
    #     tmp = deepcopy(predicted_array_or_file)
    #     if predicted_array_or_file.endswith('.npy'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)
    #     elif predicted_array_or_file.endswith('.npz'):
    #         predicted_array_or_file = np.load(predicted_array_or_file)['softmax']
    #     os.remove(tmp)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    ret = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file, plans_manager, configuration_manager, label_manager, properties_dict,
        return_probabilities=save_probabilities, num_threads_torch=num_threads_torch
    )
    del predicted_array_or_file

    # save
    if save_probabilities:
        segmentation_final, probabilities_final = ret
        np.savez_compressed(output_file_truncated + '.npz', probabilities=probabilities_final)
        save_pickle(properties_dict, output_file_truncated + '.pkl')
        del probabilities_final, ret
    else:
        segmentation_final = ret
        del ret

    rw = plans_manager.image_reader_writer_class()
    rw.write_seg(segmentation_final, output_file_truncated + dataset_json_dict_or_file['file_ending'],
                 properties_dict)


def resample_and_save(predicted: Union[torch.Tensor, np.ndarray], target_shape: List[int], output_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager, properties_dict: dict,
                      dataset_json_dict_or_file: Union[dict, str], num_threads_torch: int = default_num_processes,
                      dataset_class=None) \
        -> None:

    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    # resample to original shape
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    target_spacing = configuration_manager.spacing if len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]
    predicted_array_or_file = configuration_manager.resampling_fn_probabilities(predicted,
                                                                                target_shape,
                                                                                current_spacing,
                                                                                target_spacing)

    # create segmentation (argmax, regions, etc)
    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    segmentation = label_manager.convert_logits_to_segmentation(predicted_array_or_file)
    # segmentation may be torch.Tensor but we continue with numpy
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    if dataset_class is None or dataset_class == nnUNetDatasetBlosc2:
        block_size, chunk_size = nnUNetDatasetBlosc2.comp_blosc2_params(
            (1, *segmentation.shape),
            tuple(configuration_manager.patch_size),
            bytes_per_pixel=1 if len(label_manager.foreground_labels) < 255 else 2
        )
        block_size = [int(i) for i in block_size[1:]]
        chunk_size = [int(i) for i in chunk_size[1:]]
        nnUNetDatasetBlosc2.save_seg(
            segmentation.astype(dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16),
            output_file,
            chunks_seg=chunk_size,
            blocks_seg=block_size)
    else:
        dataset_class.save_seg(segmentation.astype(dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16), output_file)
    torch.set_num_threads(old_threads)


def export_regression_prediction_from_logits(predicted_array_or_file: Union[np.ndarray, torch.Tensor], 
                                           properties_dict: dict,
                                           configuration_manager: ConfigurationManager,
                                           plans_manager: PlansManager,
                                           dataset_json_dict_or_file: Union[dict, str], 
                                           output_file_truncated: str,
                                           num_threads_torch: int = default_num_processes):
    """
    Export regression predictions (bypasses segmentation post-processing).
    This function handles continuous regression values and saves them as float32 NIfTI files.
    
    Args:
        predicted_array_or_file: Network prediction (continuous values, not logits)
        properties_dict: Case properties for preprocessing reversal
        configuration_manager: Configuration manager
        plans_manager: Plans manager
        dataset_json_dict_or_file: Dataset JSON or path to it
        output_file_truncated: Output filename without extension
        num_threads_torch: Number of threads for torch operations
    """
    import SimpleITK as sitk
    
    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)
    
    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    try:
        # Convert to numpy if needed and normalize shape
        if isinstance(predicted_array_or_file, torch.Tensor):
            prediction = predicted_array_or_file.cpu().numpy()
        else:
            prediction = predicted_array_or_file
        
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
        
        # Apply denormalization to original intensity ranges (for CT data)
        prediction = _denormalize_regression_prediction(prediction, configuration_manager, plans_manager, channel_idx=1)
        
        # Revert preprocessing steps using existing nnUNet infrastructure
        # 1. Resample back to original spacing
        spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
        current_spacing = configuration_manager.spacing if \
            len(configuration_manager.spacing) == \
            len(properties_dict['shape_after_cropping_and_before_resampling']) else \
            [spacing_transposed[0], *configuration_manager.spacing]
        target_spacing = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
        
        if 'shape_after_cropping_and_before_resampling' in properties_dict:
            # Add channel dimension for resampling function
            prediction_with_channel = prediction[None]  # Add channel dim
            
            # For regression, force trilinear interpolation (order_z=1) for continuous data
            # This ensures smooth interpolation in all directions, unlike segmentation which uses mixed interpolation
            from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
            
            prediction_resampled = resample_data_or_seg_to_shape(
                prediction_with_channel,
                properties_dict['shape_after_cropping_and_before_resampling'],
                current_spacing,
                target_spacing,
                is_seg=False,  # Continuous data, not segmentation
                order=1,  # Linear interpolation in x,y
                order_z=1,  # Linear interpolation in z (not nearest neighbor like segmentation)
                force_separate_z=None,
                separate_z_anisotropy_threshold=3.0
            )[0]  # Remove channel dimension
        else:
            prediction_resampled = prediction
        
        # 2. Revert cropping to original image size
        if 'bbox_used_for_cropping' in properties_dict:
            prediction_full = np.zeros(properties_dict['shape_before_cropping'], dtype=np.float32)
            prediction_full = insert_crop_into_image(prediction_full, prediction_resampled, 
                                                    properties_dict['bbox_used_for_cropping'])
        else:
            prediction_full = prediction_resampled.astype(np.float32)
        
        # 3. Revert transpose if needed
        if hasattr(plans_manager, 'transpose_backward'):
            prediction_final = prediction_full.transpose(plans_manager.transpose_backward)
        else:
            prediction_final = prediction_full
        
        # Save using SimpleITK to preserve float32 values and correct orientation
        output_filename = f'{output_file_truncated}{dataset_json_dict_or_file["file_ending"]}'
        
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
        
    finally:
        torch.set_num_threads(old_threads)


def _denormalize_regression_prediction(prediction: np.ndarray, configuration_manager: ConfigurationManager, 
                                     plans_manager: PlansManager, channel_idx: int = 1) -> np.ndarray:
    """
    Denormalize regression predictions back to original intensity range
    
    Args:
        prediction: Normalized prediction array
        configuration_manager: Configuration manager
        plans_manager: Plans manager  
        channel_idx: Channel index for normalization parameters (1 for CT target channel)
    
    Returns:
        Denormalized prediction in original intensity range
    """
    # Get normalization info from plans
    normalization_schemes = configuration_manager.normalization_schemes
    intensity_props = plans_manager.foreground_intensity_properties_per_channel
    
    if channel_idx >= len(normalization_schemes):
        print(f'Warning: channel_idx {channel_idx} >= number of channels, using channel 0 for denormalization')
        channel_idx = 0
        
    norm_scheme = normalization_schemes[channel_idx]
    props = intensity_props[str(channel_idx)]
    
    print(f'Denormalizing with scheme: {norm_scheme}, channel: {channel_idx}')
    
    if norm_scheme == 'ZScoreNormalization':
        # Reverse: x_orig = (x_norm * std) + mean  
        mean = props['mean']
        std = props['std']
        prediction = prediction * std + mean
        print(f'Applied ZScore denormalization: mean={mean:.2f}, std={std:.2f}')
        
    elif norm_scheme == 'GlobalNormalization':
        # Reverse: x_orig = (x_norm * std) + mean  
        mean = props['mean']
        std = props['std']
        prediction = prediction * std + mean
        print(f'Applied Global denormalization: mean={mean:.2f}, std={std:.2f}')
        
    elif norm_scheme == 'CTNormalization':
        # Reverse: x_orig = (x_norm * std) + mean
        mean = props['mean']
        std = props['std']  
        prediction = prediction * std + mean
        print(f'Applied CT denormalization: mean={mean:.2f}, std={std:.2f}')
        
    elif norm_scheme == 'RescaleTo01Normalization':
        # This would need the original min/max, but those aren't stored in intensity properties
        # For now, just return as-is and warn
        print('Warning: RescaleTo01Normalization denormalization not implemented, returning normalized values')
        
    elif norm_scheme == 'NoNormalization':
        # No denormalization needed
        print('No denormalization applied (NoNormalization scheme)')
        
    else:
        print(f'Warning: Unknown normalization scheme {norm_scheme}, returning normalized values')
        
    return prediction
