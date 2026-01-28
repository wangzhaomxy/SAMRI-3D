# -*- coding: utf-8 -*-
"""
Generate SAM2-compatible embeddings from NIfTI images.

This script pre-computes SAM2 image embeddings including high-resolution features
required by the SAM2 mask decoder. The embeddings can be used to accelerate
training by skipping the image encoder forward pass.

Output NPZ format:
- image_embed: (256, 64, 64) - main image embedding
- high_res_feat_0: (32, 256, 256) - high-res feature level 0
- high_res_feat_1: (64, 128, 128) - high-res feature level 1
- mask: (1, H, W) - ground truth mask
- ori_size: (H, W) - original image size

Pickle output format (embedding_list.pkl):
- 'all_entries': List of all entries with full metadata for debugging/modification
    [{'id': unique_id, 'emb_path': path, 'img_path': orig_img, 'mask_path': orig_mask,
      'dataset': name, 'labels': label, 'dice': score}, ...]
- 'train_40_down': Filtered list for training (0.2 < dice <= 0.4)
- 'train_40_up': Filtered list for training (dice > 0.4)
- 'metadata': Generation metadata (config, thresholds, timestamps, etc.)

Usage:
  1. Generate SAM2 embeddings (default settings):
python generate_sam2_embeddings.py \
--sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
--ckpt-path /scratch/project/samri/pretrained_ckpt/sam2.1_hiera_small.pt \
--data-paths /scratch/project/samri/Datasets/SAMRI_train_test/*/training/ \
--output-dir /scratch/project/samri/SAM2_embeddings/ \
--save-pickle-list

  1b. Fast generation (no compression, skip dice eval, more workers):
python generate_sam2_embeddings.py \
--sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
--ckpt-path /scratch/project/samri/pretrained_ckpt/sam2.1_hiera_small.pt \
--data-paths /scratch/project/samri/Datasets/SAMRI_train_test/*/training/ \
--output-dir /scratch/project/samri/SAM2_embeddings/ \
--save-pickle-list \
--num-workers 8 --prefetch-size 16 --no-compress --skip-dice-eval

  2. Train with SAM2 embeddings:
python train_stage_one_slicewise.py \
--sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \
--ckpt-path /scratch/project/samri/pretrained_ckpt/sam2.1_hiera_small.pt \
--use-embeddings \
--sam2-embeddings \
--embedding-file /scratch/project/samri/SAM2_embeddings/embedding_list.pkl \
--save-path /scratch/project/samri/Model_save/samri2_stage1/

  3. Modify training list (remove failed entries, re-filter):
python generate_sam2_embeddings.py --modify-list embedding_list.pkl \
--remove-ids "dataset_file_1,dataset_file_2" \
--output modified_list.pkl
"""

import os
import argparse
from datetime import datetime
from tqdm import tqdm
import numpy as np
import glob
import nibabel as nib
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# Dice score thresholds for pre-filtering (for backward compatibility)
# Note: These are used for pre-generating train_40_up/train_40_down lists
# The training script can also filter dynamically using all_entries with custom thresholds
DICE_THRESHOLD_LOW = 0.2   # Minimum dice to include
DICE_THRESHOLD_HIGH = 0.4  # Threshold to split train_40_down vs train_40_up


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SAM2 embeddings from NIfTI images")

    # Model configuration (required for generation mode)
    parser.add_argument("--sam2-config", type=str, default=None,
                        help="Path to SAM2 config file")
    parser.add_argument("--ckpt-path", type=str, default=None,
                        help="Path to SAM2 checkpoint")

    # Data paths (required for generation mode)
    parser.add_argument("--data-paths", type=str, nargs="+", default=None,
                        help="List of data paths containing NIfTI images")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for embeddings")

    # Processing options
    parser.add_argument("--image-key", type=str, default="*_img_*",
                        help="Glob pattern for image files")
    parser.add_argument("--mask-key", type=str, default="*_seg_*",
                        help="Glob pattern for mask files")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for processing (currently only 1 supported)")

    # Performance options
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers for parallel file I/O (default: 4)")
    parser.add_argument("--no-compress", action="store_true",
                        help="Disable NPZ compression for faster saving (larger files)")
    parser.add_argument("--prefetch-size", type=int, default=8,
                        help="Number of images to prefetch in background (default: 8)")
    parser.add_argument("--skip-dice-eval", action="store_true",
                        help="Skip dice evaluation for faster generation (dice will be set to 1.0)")

    # Output options
    parser.add_argument("--save-pickle-list", action="store_true",
                        help="Also save a pickle file listing all embeddings for BalancedEmbDataset")
    parser.add_argument("--pickle-output", type=str, default=None,
                        help="Path for pickle list output (default: output_dir/embedding_list.pkl)")

    # List modification options (alternative mode)
    parser.add_argument("--modify-list", type=str, default=None,
                        help="Path to existing pickle list to modify (enables modification mode)")
    parser.add_argument("--remove-ids", type=str, default=None,
                        help="Comma-separated list of entry IDs to remove")
    parser.add_argument("--remove-by-emb-path", type=str, default=None,
                        help="Remove entries matching this embedding path pattern")
    parser.add_argument("--remove-by-dataset", type=str, default=None,
                        help="Remove all entries from this dataset")
    parser.add_argument("--refilter", action="store_true",
                        help="Re-filter the list using current thresholds (use with --modify-list)")
    parser.add_argument("--new-threshold-low", type=float, default=None,
                        help="New low threshold for re-filtering")
    parser.add_argument("--new-threshold-high", type=float, default=None,
                        help="New high threshold for re-filtering")
    parser.add_argument("--list-datasets", action="store_true",
                        help="List all datasets in the pickle file (use with --modify-list)")
    parser.add_argument("--show-stats", action="store_true",
                        help="Show statistics of the pickle file (use with --modify-list)")

    return parser.parse_args()


def load_nii(nii_file):
    """Load NIfTI file and return numpy array."""
    return nib.load(nii_file).get_fdata()


def dice_score(y_true, y_pred, smooth=1e-10):
    """
    Calculate dice similarity coefficient between two binary masks.

    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
        smooth: Small constant to avoid division by zero

    Returns:
        float: Dice score in range [0, 1]
    """
    intersection = np.sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)


def get_bbox_from_mask(mask):
    """
    Get bounding box from binary mask in XYXY format.

    Args:
        mask: Binary mask array (H, W)

    Returns:
        np.ndarray: Bounding box [x1, y1, x2, y2] or None if mask is empty
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return np.array([x1, y1, x2, y2])


def preprocess_image(np_image):
    """
    Preprocess image to SAM2 input format.

    Args:
        np_image: Input image (1, H, W) or (H, W)

    Returns:
        np.ndarray: Image in HWC uint8 format (H, W, 3)
    """
    if np_image.ndim == 3:
        np_image = np_image[0]  # (1, H, W) -> (H, W)

    # Normalize to [0, 1]
    np_image = (np_image - np_image.min()) / (np_image.max() - np_image.min() + 1e-8)

    # Convert to uint8 [0, 255]
    np_image = (np_image * 255).astype(np.uint8)

    # Expand to 3 channels (H, W, 3)
    np_image = np.stack([np_image, np_image, np_image], axis=-1)

    return np_image


def extract_dataset_name(file_path, data_paths):
    """
    Extract dataset name from file path based on data_paths.

    Args:
        file_path: Full path to the file
        data_paths: List of data directories

    Returns:
        str: Dataset name extracted from path
    """
    for dp in data_paths:
        if file_path.startswith(dp):
            # Try to extract dataset name from parent directories
            # Common patterns: .../DatasetName/training/file.nii.gz
            parts = dp.rstrip('/').split('/')
            for i, part in enumerate(parts):
                if part in ['training', 'testing', 'validation']:
                    if i > 0:
                        return parts[i - 1]
            # Fallback: use the last non-empty part
            return parts[-1] if parts[-1] else parts[-2]
    # Fallback: use parent directory name
    return os.path.basename(os.path.dirname(file_path))


def generate_entry_id(dataset, filename, label):
    """
    Generate a unique ID for an embedding entry.

    Args:
        dataset: Dataset name
        filename: Base filename (without extension)
        label: Label value

    Returns:
        str: Unique identifier
    """
    # Clean filename to remove extensions
    clean_name = filename.replace('.nii.gz', '').replace('.nii', '')
    clean_name = clean_name.replace('.mha', '').replace('.mhd', '').replace('.nrrd', '')
    return f"{dataset}__{clean_name}__label{label}"


def load_image_mask_pair(args_tuple):
    """
    Load and preprocess an image-mask pair (for parallel loading).

    Args:
        args_tuple: Tuple of (img_path, mask_path)

    Returns:
        dict: Loaded and preprocessed data
    """
    img_path, mask_path = args_tuple
    img_nii = load_nii(img_path)
    mask_nii = load_nii(mask_path)
    img = preprocess_image(img_nii)

    # Prepare mask (ensure it's in the right format)
    if mask_nii.ndim == 2:
        mask_2d = mask_nii
        mask_nii = mask_nii[np.newaxis, ...]  # (H, W) -> (1, H, W)
    elif mask_nii.ndim == 3 and mask_nii.shape[0] != 1:
        # Assume it's (H, W, 1) or similar
        mask_nii = mask_nii.transpose(2, 0, 1)
        mask_2d = mask_nii[0]
    else:
        mask_2d = mask_nii[0] if mask_nii.ndim == 3 else mask_nii

    return {
        'img': img,
        'mask_nii': mask_nii,
        'mask_2d': mask_2d,
        'img_path': img_path,
        'mask_path': mask_path,
    }


def save_embedding_async(save_args):
    """
    Save embedding to NPZ file (for async saving).

    Args:
        save_args: Dict with output_path, features, mask_nii, compress flag
    """
    output_path = save_args['output_path']
    features = save_args['features']
    mask_nii = save_args['mask_nii']
    compress = save_args['compress']

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    save_fn = np.savez_compressed if compress else np.savez
    save_fn(
        output_path,
        image_embed=features['image_embed'],
        high_res_feat_0=features['high_res_feat_0'],
        high_res_feat_1=features['high_res_feat_1'],
        mask=mask_nii.astype(np.int16),
        ori_size=np.array(features['orig_hw']),
    )
    return output_path


def prefetch_data_generator(file_pairs, num_workers, prefetch_size):
    """
    Generator that prefetches data using a thread pool.

    Args:
        file_pairs: List of (img_path, mask_path) tuples
        num_workers: Number of worker threads
        prefetch_size: Number of items to prefetch

    Yields:
        Preprocessed data dictionaries
    """
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit initial batch
        futures = {}
        for i, pair in enumerate(file_pairs[:prefetch_size]):
            future = executor.submit(load_image_mask_pair, pair)
            futures[future] = i

        next_idx = prefetch_size
        results = {}
        expected_idx = 0

        while futures or expected_idx < len(file_pairs):
            # Wait for any future to complete
            if futures:
                done_futures = []
                for future in list(futures.keys()):
                    if future.done():
                        done_futures.append(future)

                if not done_futures:
                    # Wait for at least one to complete
                    import concurrent.futures
                    done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
                    done_futures = list(done)

                for future in done_futures:
                    idx = futures.pop(future)
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        print(f"Error loading file pair {idx}: {e}")
                        results[idx] = None

                    # Submit next item if available
                    if next_idx < len(file_pairs):
                        new_future = executor.submit(load_image_mask_pair, file_pairs[next_idx])
                        futures[new_future] = next_idx
                        next_idx += 1

            # Yield results in order
            while expected_idx in results:
                result = results.pop(expected_idx)
                expected_idx += 1
                if result is not None:
                    yield result


def process_dataset(predictor, data_paths, image_key, mask_key, output_dir,
                    num_workers=4, compress=True, prefetch_size=8, skip_dice_eval=False):
    """
    Process all images in the dataset, generate embeddings, and evaluate dice scores.

    Uses parallel file I/O and async saving for better GPU utilization.

    Args:
        predictor: SAM2ImagePredictor for mask prediction and feature extraction
        data_paths: List of data directories
        image_key: Glob pattern for images
        mask_key: Glob pattern for masks
        output_dir: Output directory
        num_workers: Number of workers for parallel I/O (default: 4)
        compress: Whether to use compression for NPZ files (default: True)
        prefetch_size: Number of images to prefetch (default: 8)
        skip_dice_eval: Skip dice evaluation for faster generation (default: False)

    Returns:
        list: List of dictionaries with embedding info, dice scores, and metadata
    """
    # Collect all image and mask files
    img_files = []
    mask_files = []
    for path in data_paths:
        img_files += sorted(glob.glob(os.path.join(path, image_key)))
        mask_files += sorted(glob.glob(os.path.join(path, mask_key)))

    if len(img_files) != len(mask_files):
        print(f"Warning: Number of images ({len(img_files)}) != number of masks ({len(mask_files)})")
        print("Attempting to match by name...")
        # Try to match by base name
        img_dict = {os.path.basename(f).replace('_img_', '_'): f for f in img_files}
        mask_dict = {os.path.basename(f).replace('_seg_', '_').replace('_gt_', '_'): f for f in mask_files}
        common_keys = set(img_dict.keys()) & set(mask_dict.keys())
        img_files = [img_dict[k] for k in sorted(common_keys)]
        mask_files = [mask_dict[k] for k in sorted(common_keys)]
        print(f"Matched {len(img_files)} image-mask pairs")

    print(f"Processing {len(img_files)} images...")
    print(f"  Parallel workers: {num_workers}")
    print(f"  Prefetch size: {prefetch_size}")
    print(f"  Compression: {'enabled' if compress else 'disabled'}")
    print(f"  Dice evaluation: {'disabled' if skip_dice_eval else 'enabled'}")

    embedding_list = []
    file_pairs = list(zip(img_files, mask_files))

    # Create thread pool for async saving
    save_executor = ThreadPoolExecutor(max_workers=num_workers)
    save_futures = []

    # Process with prefetching
    pbar = tqdm(total=len(file_pairs))

    for data in prefetch_data_generator(file_pairs, num_workers, prefetch_size):
        img = data['img']
        mask_nii = data['mask_nii']
        mask_2d = data['mask_2d']
        img_path = data['img_path']
        mask_path = data['mask_path']

        # Set image for predictor (this also extracts features internally)
        predictor.set_image(img)

        # Extract features for saving (reuse from predictor's internal state)
        features = {
            'image_embed': predictor._features['image_embed'].cpu().numpy().squeeze(0),
            'high_res_feat_0': predictor._features['high_res_feats'][0].cpu().numpy().squeeze(0),
            'high_res_feat_1': predictor._features['high_res_feats'][1].cpu().numpy().squeeze(0),
            'orig_hw': predictor._orig_hw[0],
        }

        # Create output filename
        base_name = os.path.basename(img_path)
        base_name = os.path.splitext(base_name)[0]
        if base_name.endswith('.nii'):
            base_name = base_name[:-4]
        output_name = f"{base_name}_sam2emb.npz"

        # Determine output subdirectory (preserve dataset structure)
        rel_path = None
        for dp in data_paths:
            if img_path.startswith(dp):
                rel_path = os.path.relpath(os.path.dirname(img_path), dp)
                break

        if rel_path and rel_path != '.':
            out_subdir = os.path.join(output_dir, rel_path)
        else:
            out_subdir = output_dir

        output_path = os.path.join(out_subdir, output_name)

        # Submit async save
        save_future = save_executor.submit(save_embedding_async, {
            'output_path': output_path,
            'features': features,
            'mask_nii': mask_nii,
            'compress': compress,
        })
        save_futures.append(save_future)

        # Extract dataset name for metadata
        dataset_name = extract_dataset_name(img_path, data_paths)

        # Collect unique labels and evaluate dice scores
        unique_labels = np.unique(mask_2d)
        unique_labels = unique_labels[unique_labels != 0]  # Remove background

        for label in unique_labels:
            # Create binary mask for this label
            gt_mask = (mask_2d == label).astype(np.float32)

            if skip_dice_eval:
                # Skip dice evaluation for faster generation
                dsc = 1.0
                iou_pred = 1.0
            else:
                # Get bounding box from ground truth mask
                bbox = get_bbox_from_mask(gt_mask)
                if bbox is None:
                    continue

                # Predict mask using box prompt
                masks, iou_preds, _ = predictor.predict(
                    box=bbox,
                    multimask_output=False,
                )

                # Calculate dice score
                pred_mask = masks[0] > 0  # Binary mask
                dsc = dice_score(gt_mask, pred_mask.astype(np.float32))
                iou_pred = float(iou_preds[0])

            # Generate unique ID for this entry
            entry_id = generate_entry_id(dataset_name, base_name, int(label))

            embedding_list.append({
                'id': entry_id,
                'emb_path': output_path,
                'img_path': img_path,
                'mask_path': mask_path,
                'dataset': dataset_name,
                'labels': int(label),
                'dice': float(dsc),
                'iou_pred': iou_pred,
            })

        pbar.update(1)

    pbar.close()

    # Wait for all saves to complete
    print("Waiting for file saves to complete...")
    for future in tqdm(as_completed(save_futures), total=len(save_futures), desc="Saving"):
        try:
            future.result()  # This will raise any exceptions
        except Exception as e:
            print(f"Error saving file: {e}")

    save_executor.shutdown(wait=True)

    return embedding_list


def filter_entries(all_entries, threshold_low, threshold_high):
    """
    Filter entries by dice score thresholds.

    Args:
        all_entries: List of all entries with full metadata
        threshold_low: Minimum dice to include
        threshold_high: Threshold to split down vs up

    Returns:
        tuple: (train_40_down, train_40_up, skipped_count)
    """
    train_40_down = []
    train_40_up = []
    skipped_low = 0

    for entry in all_entries:
        dice = entry.get('dice', 0)
        # Format for BalancedEmbDataset: {'emb_path': path, 'labels': label}
        sample = {
            'emb_path': entry['emb_path'],
            'labels': entry['labels'],
        }
        if dice > threshold_high:
            train_40_up.append(sample)
        elif dice > threshold_low:
            train_40_down.append(sample)
        else:
            skipped_low += 1

    return train_40_down, train_40_up, skipped_low


def show_pickle_stats(pickle_data):
    """Display statistics of a pickle file."""
    all_entries = pickle_data.get('all_entries', [])
    metadata = pickle_data.get('metadata', {})

    print("\n=== Pickle File Statistics ===")
    print(f"Generated at: {metadata.get('generated_at', 'Unknown')}")
    print(f"SAM2 config: {metadata.get('sam2_config', 'Unknown')}")
    print(f"Checkpoint: {metadata.get('ckpt_path', 'Unknown')}")
    print(f"\nThresholds:")
    print(f"  Low: {metadata.get('dice_threshold_low', 'Unknown')}")
    print(f"  High: {metadata.get('dice_threshold_high', 'Unknown')}")

    print(f"\nTotal entries: {len(all_entries)}")
    print(f"  train_40_up: {len(pickle_data.get('train_40_up', []))}")
    print(f"  train_40_down: {len(pickle_data.get('train_40_down', []))}")
    print(f"  Skipped: {metadata.get('skipped_low', 'Unknown')}")

    # Dataset breakdown
    if all_entries:
        datasets = {}
        for entry in all_entries:
            ds = entry.get('dataset', 'Unknown')
            datasets[ds] = datasets.get(ds, 0) + 1

        print(f"\nDatasets ({len(datasets)} total):")
        for ds, count in sorted(datasets.items()):
            print(f"  {ds}: {count} entries")

        # Dice score distribution
        dices = [e.get('dice', 0) for e in all_entries]
        print(f"\nDice score distribution:")
        print(f"  Min: {min(dices):.4f}")
        print(f"  Max: {max(dices):.4f}")
        print(f"  Mean: {np.mean(dices):.4f}")
        print(f"  Median: {np.median(dices):.4f}")


def list_datasets(pickle_data):
    """List all datasets in the pickle file."""
    all_entries = pickle_data.get('all_entries', [])
    datasets = {}
    for entry in all_entries:
        ds = entry.get('dataset', 'Unknown')
        datasets[ds] = datasets.get(ds, 0) + 1

    print("\n=== Datasets ===")
    for ds, count in sorted(datasets.items()):
        print(f"{ds}: {count} entries")


def modify_pickle_list(args):
    """Handle pickle list modification operations."""
    print(f"Loading pickle list from {args.modify_list}...")
    with open(args.modify_list, 'rb') as f:
        pickle_data = pickle.load(f)

    all_entries = pickle_data.get('all_entries', [])
    metadata = pickle_data.get('metadata', {})

    # Show stats if requested
    if args.show_stats:
        show_pickle_stats(pickle_data)
        return

    # List datasets if requested
    if args.list_datasets:
        list_datasets(pickle_data)
        return

    modified = False
    original_count = len(all_entries)

    # Remove by IDs
    if args.remove_ids:
        ids_to_remove = set(args.remove_ids.split(','))
        all_entries = [e for e in all_entries if e.get('id') not in ids_to_remove]
        removed = original_count - len(all_entries)
        print(f"Removed {removed} entries by ID")
        modified = True

    # Remove by embedding path pattern
    if args.remove_by_emb_path:
        pattern = args.remove_by_emb_path
        before = len(all_entries)
        all_entries = [e for e in all_entries if pattern not in e.get('emb_path', '')]
        removed = before - len(all_entries)
        print(f"Removed {removed} entries matching emb_path pattern '{pattern}'")
        modified = True

    # Remove by dataset
    if args.remove_by_dataset:
        dataset = args.remove_by_dataset
        before = len(all_entries)
        all_entries = [e for e in all_entries if e.get('dataset') != dataset]
        removed = before - len(all_entries)
        print(f"Removed {removed} entries from dataset '{dataset}'")
        modified = True

    # Re-filter with new thresholds
    threshold_low = args.new_threshold_low if args.new_threshold_low is not None else metadata.get('dice_threshold_low', DICE_THRESHOLD_LOW)
    threshold_high = args.new_threshold_high if args.new_threshold_high is not None else metadata.get('dice_threshold_high', DICE_THRESHOLD_HIGH)

    if args.refilter or modified or args.new_threshold_low is not None or args.new_threshold_high is not None:
        train_40_down, train_40_up, skipped = filter_entries(all_entries, threshold_low, threshold_high)

        # Update metadata
        metadata['dice_threshold_low'] = threshold_low
        metadata['dice_threshold_high'] = threshold_high
        metadata['skipped_low'] = skipped
        metadata['last_modified'] = datetime.now().isoformat()
        metadata['total_entries'] = len(all_entries)

        # Build new pickle data
        new_pickle_data = {
            'all_entries': all_entries,
            'train_40_down': train_40_down,
            'train_40_up': train_40_up,
            'metadata': metadata,
        }

        # Save to output
        output_path = args.pickle_output or args.modify_list
        with open(output_path, 'wb') as f:
            pickle.dump(new_pickle_data, f)

        print(f"\nSaved modified list to {output_path}")
        print(f"  Total entries: {len(all_entries)}")
        print(f"  train_40_up (dice > {threshold_high}): {len(train_40_up)} samples")
        print(f"  train_40_down ({threshold_low} < dice <= {threshold_high}): {len(train_40_down)} samples")
        print(f"  Skipped (dice <= {threshold_low}): {skipped} samples")
    else:
        print("No modifications made. Use --show-stats to view current state.")


def main():
    args = parse_args()

    # Check if we're in modification mode
    if args.modify_list:
        modify_pickle_list(args)
        return

    # Validate required arguments for generation mode
    if not all([args.sam2_config, args.ckpt_path, args.data_paths, args.output_dir]):
        print("Error: Generation mode requires --sam2-config, --ckpt-path, --data-paths, and --output-dir")
        print("Use --modify-list for modification mode")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading SAM2 model from {args.ckpt_path}...")

    # Build SAM2 model
    model = build_sam2(
        config_file=args.sam2_config,
        ckpt_path=args.ckpt_path,
        device=args.device,
        mode="eval",
    )
    model.eval()

    # Create image predictor for mask prediction and dice evaluation
    predictor = SAM2ImagePredictor(model)

    print(f"Model loaded. Image size: {model.image_size}")
    print(f"Processing data from: {args.data_paths}")
    print(f"Output directory: {args.output_dir}")

    # Process dataset with parallel I/O
    embedding_list = process_dataset(
        predictor=predictor,
        data_paths=args.data_paths,
        image_key=args.image_key,
        mask_key=args.mask_key,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        compress=not args.no_compress,
        prefetch_size=args.prefetch_size,
        skip_dice_eval=args.skip_dice_eval,
    )

    print(f"Generated {len(embedding_list)} embedding entries")

    # Save pickle list if requested
    if args.save_pickle_list:
        pickle_path = args.pickle_output or os.path.join(args.output_dir, "embedding_list.pkl")

        # Filter entries by dice score
        train_40_down, train_40_up, skipped_low = filter_entries(
            embedding_list, DICE_THRESHOLD_LOW, DICE_THRESHOLD_HIGH
        )

        # Build pickle data with full metadata
        pickle_data = {
            # Raw data with all metadata (for re-filtering and debugging)
            'all_entries': embedding_list,

            # Filtered lists for training (compatible with BalancedEmbDataset)
            'train_40_down': train_40_down,
            'train_40_up': train_40_up,

            # Generation metadata
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'sam2_config': args.sam2_config,
                'ckpt_path': args.ckpt_path,
                'data_paths': args.data_paths,
                'image_key': args.image_key,
                'mask_key': args.mask_key,
                'dice_threshold_low': DICE_THRESHOLD_LOW,
                'dice_threshold_high': DICE_THRESHOLD_HIGH,
                'total_entries': len(embedding_list),
                'skipped_low': skipped_low,
            },
        }

        with open(pickle_path, 'wb') as f:
            pickle.dump(pickle_data, f)

        print(f"\nSaved embedding list to {pickle_path}")
        print(f"  Total entries (all_entries): {len(embedding_list)}")
        print(f"  train_40_up (dice > {DICE_THRESHOLD_HIGH}): {len(train_40_up)} samples")
        print(f"  train_40_down ({DICE_THRESHOLD_LOW} < dice <= {DICE_THRESHOLD_HIGH}): {len(train_40_down)} samples")
        print(f"  Skipped (dice <= {DICE_THRESHOLD_LOW}): {skipped_low} samples")

    print("\nDone!")


if __name__ == "__main__":
    main()
