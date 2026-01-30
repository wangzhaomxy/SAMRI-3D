# -*- coding: utf-8 -*-
"""
Generate SAM2-compatible embeddings from NIfTI images.

This script pre-computes SAM2 image embeddings including high-resolution features
required by the SAM2 mask decoder. Only embeddings with dice score > threshold are
saved to conserve disk space.

Output NPZ format:
- image_embed: (256, 64, 64) - main image embedding
- high_res_feat_0: (32, 256, 256) - high-res feature level 0
- high_res_feat_1: (64, 128, 128) - high-res feature level 1
- mask: (1, H, W) - ground truth mask
- ori_size: (H, W) - original image size

Alternative: For training without pre-computing embeddings, use gen_training_list.py
to generate a training list with dice scores, then train with FilteredNiiDataset.
See train_stage_one_slicewise.py for workflow options.

=== Usage ===

1. Generate SAM2 embeddings:
   python generate_sam2_embeddings.py \\
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
       --ckpt-path /scratch/project/samri/pretrained_ckpt/sam2.1_hiera_small.pt \\
       --data-paths /scratch/project/samri/Datasets/SAMRI_train_test/*/training/ \\
       --output-dir /scratch/project/samri/SAM2_embeddings/ \\
       --num-workers 12 --prefetch-size 24

2. Resume after crash (automatically skips existing embeddings):
   python generate_sam2_embeddings.py \\
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
       --ckpt-path /scratch/project/samri/pretrained_ckpt/sam2.1_hiera_small.pt \\
       --data-paths /scratch/project/samri/Datasets/SAMRI_train_test/*/training/ \\
       --output-dir /scratch/project/samri/SAM2_embeddings/ \\
       --num-workers 8 --prefetch-size 16

3. Custom dice threshold (default is 0.4):
   python generate_sam2_embeddings.py \\
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
       --ckpt-path /scratch/project/samri/pretrained_ckpt/sam2.1_hiera_small.pt \\
       --data-paths /scratch/project/samri/Datasets/SAMRI_train_test/*/training/ \\
       --output-dir /scratch/project/samri/SAM2_embeddings/ \\
       --dice-threshold 0.5 \\
       --num-workers 8 --prefetch-size 16

4. Force reprocessing all images:
   python generate_sam2_embeddings.py \\
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
       --ckpt-path /scratch/project/samri/pretrained_ckpt/sam2.1_hiera_small.pt \\
       --data-paths /scratch/project/samri/Datasets/SAMRI_train_test/*/training/ \\
       --output-dir /scratch/project/samri/SAM2_embeddings/ \\
       --no-skip-existing

5. Train with SAM2 embeddings:
   python train_stage_one_slicewise.py \\
       --sam2-config configs/sam2.1/sam2.1_hiera_s.yaml \\
       --ckpt-path /scratch/project/samri/pretrained_ckpt/sam2.1_hiera_small.pt \\
       --use-embeddings --sam2-embeddings \\
       --embedding-dir /scratch/project/samri/SAM2_embeddings/ \\
       --save-path /scratch/project/samri/Model_save/samri2_stage1/

=== Options ===

Performance:
  --num-workers N      Workers for parallel file I/O (default: 4)
  --prefetch-size N    Images to prefetch in background (default: 8)
  --no-compress        Disable NPZ compression (faster saves, larger files)
  --dice-threshold     Minimum dice score to save embedding (default: 0.4)

Resume:
  --skip-existing      Skip images with existing embeddings (default: enabled)
  --no-skip-existing   Force reprocessing all images
"""

import os
import argparse
from tqdm import tqdm
import numpy as np
import glob
import nibabel as nib
from concurrent.futures import ThreadPoolExecutor, as_completed

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# Minimum dice score threshold - embeddings below this are discarded to save space
DICE_THRESHOLD = 0.4


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
    parser.add_argument("--dice-threshold", type=float, default=DICE_THRESHOLD,
                        help=f"Minimum dice score to save embedding (default: {DICE_THRESHOLD})")

    # Resume options
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip images that already have embeddings (default: enabled)")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Force reprocessing of all images, even if embeddings exist")

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


def get_existing_embeddings(output_dir):
    """
    Scan output directory for existing embedding files.

    Args:
        output_dir: Root output directory containing dataset subdirectories

    Returns:
        set: Set of (dataset_name, base_name) tuples for existing embeddings
    """
    existing = set()
    if not os.path.exists(output_dir):
        return existing

    # Recursively find all .npz files
    npz_files = glob.glob(os.path.join(output_dir, "**", "*_sam2emb.npz"), recursive=True)

    for npz_path in npz_files:
        # Extract dataset name from parent directory
        dataset_name = os.path.basename(os.path.dirname(npz_path))

        # Extract base name from filename (remove _sam2emb.npz suffix)
        filename = os.path.basename(npz_path)
        base_name = filename.replace("_sam2emb.npz", "")

        existing.add((dataset_name, base_name))

    return existing


def get_base_name_from_image_path(img_path):
    """
    Extract base name from image path for matching with existing embeddings.

    Args:
        img_path: Full path to the image file

    Returns:
        str: Base name without extension
    """
    base_name = os.path.basename(img_path)
    # Remove common extensions
    base_name = os.path.splitext(base_name)[0]
    if base_name.endswith('.nii'):
        base_name = base_name[:-4]
    return base_name


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
                    num_workers=4, compress=True, prefetch_size=8, dice_threshold=0.4,
                    skip_existing=True):
    """
    Process all images in the dataset, generate embeddings, and evaluate dice scores.

    Only saves embeddings with dice score > dice_threshold to conserve disk space.
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
        dice_threshold: Minimum dice score to save embedding (default: 0.4)
        skip_existing: If True, skip images that already have embeddings (default: True)

    Returns:
        tuple: (saved_count, skipped_count, already_exist_count)
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

    print(f"Found {len(img_files)} images...")

    # Check for existing embeddings if skip_existing is enabled
    already_exist_count = 0
    if skip_existing:
        print("Scanning for existing embeddings...")
        existing_embeddings = get_existing_embeddings(output_dir)
        print(f"  Found {len(existing_embeddings)} existing embeddings")

        # Filter out already-processed images
        filtered_img_files = []
        filtered_mask_files = []
        for img_path, mask_path in zip(img_files, mask_files):
            dataset_name = extract_dataset_name(img_path, data_paths)
            base_name = get_base_name_from_image_path(img_path)
            if (dataset_name, base_name) not in existing_embeddings:
                filtered_img_files.append(img_path)
                filtered_mask_files.append(mask_path)
            else:
                already_exist_count += 1

        img_files = filtered_img_files
        mask_files = filtered_mask_files
        print(f"  Skipping {already_exist_count} already-processed images")

    print(f"Processing {len(img_files)} images...")
    print(f"  Parallel workers: {num_workers}")
    print(f"  Prefetch size: {prefetch_size}")
    print(f"  Compression: {'enabled' if compress else 'disabled'}")
    print(f"  Dice threshold: {dice_threshold} (only save if dice > {dice_threshold})")

    saved_count = 0
    skipped_count = 0
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

        # Collect unique labels and evaluate dice scores FIRST
        unique_labels = np.unique(mask_2d)
        unique_labels = unique_labels[unique_labels != 0]  # Remove background

        # Check if any label passes the dice threshold
        should_save = False
        for label in unique_labels:
            gt_mask = (mask_2d == label).astype(np.float32)
            bbox = get_bbox_from_mask(gt_mask)
            if bbox is None:
                continue

            # Predict mask using box prompt
            masks, iou_preds, _ = predictor.predict(
                box=bbox,
                multimask_output=False,
            )

            # Calculate dice score
            pred_mask = masks[0] > 0
            dsc = dice_score(gt_mask, pred_mask.astype(np.float32))

            if dsc > dice_threshold:
                should_save = True
                break

        if not should_save:
            skipped_count += 1
            pbar.update(1)
            continue

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
        dataset_name = extract_dataset_name(img_path, data_paths)
        out_subdir = os.path.join(output_dir, dataset_name)
        output_path = os.path.join(out_subdir, output_name)

        # Submit async save
        save_future = save_executor.submit(save_embedding_async, {
            'output_path': output_path,
            'features': features,
            'mask_nii': mask_nii,
            'compress': compress,
        })
        save_futures.append(save_future)
        saved_count += 1

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

    return saved_count, skipped_count, already_exist_count


def main():
    args = parse_args()

    # Validate required arguments
    if not all([args.sam2_config, args.ckpt_path, args.data_paths, args.output_dir]):
        print("Error: Required arguments: --sam2-config, --ckpt-path, --data-paths, and --output-dir")
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

    # Determine skip_existing flag (--no-skip-existing overrides default)
    skip_existing = args.skip_existing and not args.no_skip_existing

    # Process dataset with parallel I/O
    saved_count, skipped_count, already_exist_count = process_dataset(
        predictor=predictor,
        data_paths=args.data_paths,
        image_key=args.image_key,
        mask_key=args.mask_key,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        compress=not args.no_compress,
        prefetch_size=args.prefetch_size,
        dice_threshold=args.dice_threshold,
        skip_existing=skip_existing,
    )

    print(f"\n=== Summary ===")
    print(f"  Saved: {saved_count} embeddings (dice > {args.dice_threshold})")
    print(f"  Skipped (low dice): {skipped_count} embeddings (dice <= {args.dice_threshold})")
    if skip_existing:
        print(f"  Skipped (existing): {already_exist_count} embeddings already processed")
    print(f"  Output directory: {args.output_dir}")
    print("\nDone!")


if __name__ == "__main__":
    main()
