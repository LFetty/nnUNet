import multiprocessing
import os
from copy import deepcopy
from multiprocessing import Pool
from typing import Tuple, List, Union, Optional

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json, load_json, \
    isfile
from nnunetv2.configuration import default_num_processes
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json, \
    determine_reader_writer_from_file_ending
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
except ImportError:
    raise ImportError("scikit-image is required for PSNR and SSIM computation. Install with: pip install scikit-image")


def save_regression_summary_json(results: dict, output_file: str):
    """
    Save regression metrics results to JSON file
    """
    results_converted = deepcopy(results)
    # Convert NaN and infinity values for JSON compatibility
    recursive_fix_for_json_export(results_converted)
    save_json(results_converted, output_file, sort_keys=True)


def load_regression_summary_json(filename: str):
    """
    Load regression metrics results from JSON file
    """
    return load_json(filename)


def compute_regression_metrics(reference_file: str, prediction_file: str, image_reader_writer: BaseReaderWriter) -> dict:
    """
    Compute regression metrics (MAE, PSNR, SSIM) between reference and prediction images
    
    Args:
        reference_file: Path to ground truth image
        prediction_file: Path to predicted image  
        image_reader_writer: Reader/writer for loading images
    
    Returns:
        Dictionary with computed metrics
    """
    # Load images - use read_images for continuous data (not read_seg)
    try:
        # Try reading as images first (for continuous data)
        ref_data, ref_dict = image_reader_writer.read_images([reference_file])
        pred_data, pred_dict = image_reader_writer.read_images([prediction_file])
        ref_image = ref_data[0]  # Extract single image from list
        pred_image = pred_data[0]
    except:
        # Fallback to segmentation reader if image reader fails
        ref_image, ref_dict = image_reader_writer.read_seg(reference_file)
        pred_image, pred_dict = image_reader_writer.read_seg(prediction_file)
    
    # Ensure images have the same shape
    if ref_image.shape != pred_image.shape:
        raise ValueError(f"Shape mismatch: reference {ref_image.shape} vs prediction {pred_image.shape}")
    
    # Convert to float32 for computation
    ref_image = ref_image.astype(np.float32)
    pred_image = pred_image.astype(np.float32)
    
    # Compute metrics
    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    
    # Mean Absolute Error
    mae = np.mean(np.abs(ref_image - pred_image))
    results['MAE'] = float(mae)
    
    # Mean Squared Error (for reference)
    mse = np.mean((ref_image - pred_image) ** 2)
    results['MSE'] = float(mse)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    results['RMSE'] = float(rmse)
    
    # Peak Signal-to-Noise Ratio
    # For medical images, use the full data range
    data_range = ref_image.max() - ref_image.min()
    if data_range > 0:
        try:
            psnr = peak_signal_noise_ratio(ref_image, pred_image, data_range=data_range)
            results['PSNR'] = float(psnr)
        except:
            # Fallback if PSNR computation fails
            results['PSNR'] = np.nan
    else:
        results['PSNR'] = np.nan
    
    # Structural Similarity Index Measure
    try:
        # For 3D data, we need to specify which axes to compute SSIM over
        if ref_image.ndim == 3:
            # Compute SSIM slice by slice and take the mean
            ssim_scores = []
            for i in range(ref_image.shape[0]):
                slice_ref = ref_image[i]
                slice_pred = pred_image[i]
                if slice_ref.std() > 0 and slice_pred.std() > 0:  # Only compute if there's variation
                    ssim_slice = structural_similarity(
                        slice_ref, slice_pred, 
                        data_range=data_range,
                        gaussian_weights=True,
                        use_sample_covariance=False
                    )
                    ssim_scores.append(ssim_slice)
            
            if ssim_scores:
                ssim = np.mean(ssim_scores)
            else:
                ssim = np.nan
        else:
            # 2D case
            if ref_image.std() > 0 and pred_image.std() > 0:
                ssim = structural_similarity(
                    ref_image, pred_image,
                    data_range=data_range,
                    gaussian_weights=True,
                    use_sample_covariance=False
                )
            else:
                ssim = np.nan
        
        results['SSIM'] = float(ssim)
    except:
        # Fallback if SSIM computation fails
        results['SSIM'] = np.nan
    
    return results


def compute_regression_metrics_on_folder(folder_ref: str, folder_pred: str, output_file: str,
                                        image_reader_writer: BaseReaderWriter,
                                        file_ending: str,
                                        num_processes: int = default_num_processes,
                                        chill: bool = True) -> dict:
    """
    Compute regression metrics on all files in folders
    
    Args:
        folder_ref: Folder with reference/ground truth images
        folder_pred: Folder with predicted images
        output_file: Output JSON file path (must end with .json, can be None)
        image_reader_writer: Reader/writer for loading images
        file_ending: File extension to process
        num_processes: Number of parallel processes
        chill: If False, require all ref files to exist in pred folder
        
    Returns:
        Dictionary with per-case and summary metrics
    """
    if output_file is not None:
        assert output_file.endswith('.json'), 'output_file should end with .json'
    
    files_pred = subfiles(folder_pred, suffix=file_ending, join=False)
    files_ref = subfiles(folder_ref, suffix=file_ending, join=False)

    if not chill:
        present = [isfile(join(folder_pred, i)) for i in files_ref]
        assert all(present), "Not all files in folder_ref exist in folder_pred"

    # Handle regression case where GT is the highest numbered channel
    files_ref_full = []
    files_pred_full = []

    for pred_file in files_pred:
        pred_file_full = join(folder_pred, pred_file)
        files_pred_full.append(pred_file_full)

        # Find the highest numbered channel file for this case as ground truth
        case_id = pred_file.replace(file_ending, '')

        # Find all files for this case ID with channel suffixes
        case_files = [f for f in files_ref if f.startswith(case_id + '_') and f.endswith(file_ending)]

        if not case_files:
            if not chill:
                raise FileNotFoundError(f"Could not find any ground truth files for case {case_id} in {folder_ref}")
            else:
                print(f"Warning: Could not find ground truth files for case {case_id}")
                continue

        # Extract channel numbers and find the highest one
        channel_numbers = []
        for f in case_files:
            try:
                # Extract channel number from filename like "case_0001.nii.gz"
                channel_part = f.replace(case_id + '_', '').replace(file_ending, '')
                channel_num = int(channel_part)
                channel_numbers.append(channel_num)
            except ValueError:
                continue

        if not channel_numbers:
            if not chill:
                raise FileNotFoundError(f"Could not parse channel numbers for case {case_id}")
            else:
                print(f"Warning: Could not parse channel numbers for case {case_id}")
                continue

        # Use the highest numbered channel as ground truth
        highest_channel = max(channel_numbers)
        gt_file = join(folder_ref, f"{case_id}_{highest_channel:04d}{file_ending}")

        if not isfile(gt_file):
            if not chill:
                raise FileNotFoundError(f"Ground truth file does not exist: {gt_file}")
            else:
                print(f"Warning: Ground truth file does not exist: {gt_file}")
                continue

        files_ref_full.append(gt_file)
    
    # Compute metrics for each case
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        results = pool.starmap(
            compute_regression_metrics,
            list(zip(files_ref_full, files_pred_full, [image_reader_writer] * len(files_pred)))
        )
    
    # Compute summary statistics
    if len(results) == 0:
        raise RuntimeError("No predictions found to evaluate!")
    
    metric_names = ['MAE', 'MSE', 'RMSE', 'PSNR', 'SSIM']
    
    # Mean metrics across all cases
    means = {}
    for metric in metric_names:
        values = [r[metric] for r in results if not np.isnan(r[metric])]
        if values:
            means[metric] = np.mean(values)
        else:
            means[metric] = np.nan
    
    # For regression, we don't have "foreground" vs "background" like segmentation
    # So we just use the overall means as the summary
    
    recursive_fix_for_json_export(results)
    recursive_fix_for_json_export(means)
    
    result = {
        'metric_per_case': results, 
        'mean': means,
        'num_cases': len(results),
        'valid_cases': {
            metric: len([r[metric] for r in results if not np.isnan(r[metric])]) 
            for metric in metric_names
        }
    }
    
    if output_file is not None:
        save_regression_summary_json(result, output_file)
    
    return result


def compute_regression_metrics_on_folder_with_dataset_json(folder_ref: str, folder_pred: str, 
                                                          dataset_json_file: str, plans_file: str,
                                                          output_file: str = None,
                                                          num_processes: int = default_num_processes,
                                                          chill: bool = False):
    """
    Compute regression metrics using dataset.json and plans.json for configuration
    """
    dataset_json = load_json(dataset_json_file)
    # get file ending
    file_ending = dataset_json['file_ending']

    # get reader writer class
    example_file = subfiles(folder_ref, suffix=file_ending, join=True)[0]
    rw = determine_reader_writer_from_dataset_json(dataset_json, example_file)()

    # maybe auto set output file
    if output_file is None:
        output_file = join(folder_pred, 'validation_summary.json')

    compute_regression_metrics_on_folder(folder_ref, folder_pred, output_file, rw, file_ending,
                                        num_processes, chill=chill)


def compute_regression_metrics_on_folder_simple(folder_ref: str, folder_pred: str,
                                               output_file: str = None,
                                               num_processes: int = default_num_processes,
                                               chill: bool = False):
    """
    Simple regression metrics computation without dataset.json
    """
    example_file = subfiles(folder_ref, join=True)[0]
    file_ending = os.path.splitext(example_file)[-1]
    rw = determine_reader_writer_from_file_ending(file_ending, example_file, allow_nonmatching_filename=True,
                                                  verbose=False)()
    # maybe auto set output file
    if output_file is None:
        output_file = join(folder_pred, 'validation_summary.json')
    
    compute_regression_metrics_on_folder(folder_ref, folder_pred, output_file, rw, file_ending,
                                        num_processes=num_processes, chill=chill)


def evaluate_regression_folder_entry_point():
    """Entry point for command line evaluation of regression predictions"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str, help='folder with gt images')
    parser.add_argument('pred_folder', type=str, help='folder with predicted images')
    parser.add_argument('-djfile', type=str, required=True,
                        help='dataset.json file')
    parser.add_argument('-pfile', type=str, required=True,
                        help='plans.json file')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: pred_folder/validation_summary.json')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'number of processes used. Optional. Default: {default_num_processes}')
    parser.add_argument('--chill', action='store_true', 
                        help='dont crash if folder_pred does not have all files that are present in folder_gt')
    args = parser.parse_args()
    compute_regression_metrics_on_folder_with_dataset_json(
        args.gt_folder, args.pred_folder, args.djfile, args.pfile, args.o, args.np, chill=args.chill
    )


def evaluate_regression_simple_entry_point():
    """Entry point for simple command line evaluation of regression predictions"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_folder', type=str, help='folder with gt images')
    parser.add_argument('pred_folder', type=str, help='folder with predicted images')
    parser.add_argument('-o', type=str, required=False, default=None,
                        help='Output file. Optional. Default: pred_folder/validation_summary.json')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'number of processes used. Optional. Default: {default_num_processes}')
    parser.add_argument('--chill', action='store_true', 
                        help='dont crash if folder_pred does not have all files that are present in folder_gt')

    args = parser.parse_args()
    compute_regression_metrics_on_folder_simple(args.gt_folder, args.pred_folder, args.o, args.np, chill=args.chill)


if __name__ == '__main__':
    # Test example (similar to evaluate_predictions.py)
    folder_ref = '/path/to/reference/images'
    folder_pred = '/path/to/predicted/images'
    output_file = '/path/to/validation_summary.json'
    image_reader_writer = SimpleITKIO()
    file_ending = '.nii.gz'
    num_processes = 12
    
    compute_regression_metrics_on_folder(folder_ref, folder_pred, output_file, image_reader_writer, 
                                        file_ending, num_processes)