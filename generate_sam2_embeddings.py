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
"""

import os
import argparse
from tqdm import tqdm
import torch
import numpy as np
import glob
import nibabel as nib
import pickle

from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SAM2 embeddings from NIfTI images")

    # Model configuration
    parser.add_argument("--sam2-config", type=str, required=True,
                        help="Path to SAM2 config file")
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="Path to SAM2 checkpoint")

    # Data paths
    parser.add_argument("--data-paths", type=str, nargs="+", required=True,
                        help="List of data paths containing NIfTI images")
    parser.add_argument("--output-dir", type=str, required=True,
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

    # Output options
    parser.add_argument("--save-pickle-list", action="store_true",
                        help="Also save a pickle file listing all embeddings for BalancedEmbDataset")
    parser.add_argument("--pickle-output", type=str, default=None,
                        help="Path for pickle list output (default: output_dir/embedding_list.pkl)")

    return parser.parse_args()


def load_nii(nii_file):
    """Load NIfTI file and return numpy array."""
    return nib.load(nii_file).get_fdata()


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


def extract_sam2_features(model, image, transforms, device):
    """
    Extract SAM2 features including high-resolution features.

    Args:
        model: SAM2 model
        image: Input image in HWC uint8 format
        transforms: SAM2Transforms
        device: Target device

    Returns:
        dict: Dictionary containing image_embed, high_res_feats, and orig_hw
    """
    orig_hw = image.shape[:2]

    # Transform image
    image_tensor = transforms(image)
    image_tensor = image_tensor[None, ...].to(device)  # (1, 3, H, W)

    with torch.no_grad():
        # Get backbone features
        backbone_out = model.forward_image(image_tensor)
        _, vision_feats, _, _ = model._prepare_backbone_features(backbone_out)

        # Add no_mem_embed for image-only mode
        if model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + model.no_mem_embed

        # Prepare features (same as in forward_sam2_image)
        bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
        ][::-1]

        # image_embed is the last (lowest resolution) feature
        image_embed = feats[-1]  # (1, 256, 64, 64)

        # high_res_feats are the higher resolution features
        high_res_feats = feats[:-1]  # List of [(1, C, H, W), ...]

    return {
        'image_embed': image_embed.cpu().numpy().squeeze(0),  # (256, 64, 64)
        'high_res_feat_0': high_res_feats[0].cpu().numpy().squeeze(0),  # (32, 256, 256)
        'high_res_feat_1': high_res_feats[1].cpu().numpy().squeeze(0),  # (64, 128, 128)
        'orig_hw': orig_hw,
    }


def process_dataset(model, transforms, data_paths, image_key, mask_key, output_dir, device):
    """
    Process all images in the dataset and generate embeddings.

    Args:
        model: SAM2 model
        transforms: SAM2Transforms
        data_paths: List of data directories
        image_key: Glob pattern for images
        mask_key: Glob pattern for masks
        output_dir: Output directory
        device: Target device

    Returns:
        list: List of dictionaries with embedding info for pickle output
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

    embedding_list = []

    for img_path, mask_path in tqdm(zip(img_files, mask_files), total=len(img_files)):
        # Load image and mask
        img_nii = load_nii(img_path)
        mask_nii = load_nii(mask_path)

        # Preprocess image
        img = preprocess_image(img_nii)

        # Extract features
        features = extract_sam2_features(model, img, transforms, device)

        # Prepare mask (ensure it's in the right format)
        if mask_nii.ndim == 2:
            mask_nii = mask_nii[np.newaxis, ...]  # (H, W) -> (1, H, W)
        elif mask_nii.ndim == 3 and mask_nii.shape[0] != 1:
            # Assume it's (H, W, 1) or similar
            mask_nii = mask_nii.transpose(2, 0, 1)

        # Create output filename
        base_name = os.path.basename(img_path)
        base_name = os.path.splitext(base_name)[0]
        if base_name.endswith('.nii'):
            base_name = base_name[:-4]
        output_name = f"{base_name}_sam2emb.npz"

        # Determine output subdirectory (preserve dataset structure)
        # Find which data_path this file came from
        rel_path = None
        for dp in data_paths:
            if img_path.startswith(dp):
                rel_path = os.path.relpath(os.path.dirname(img_path), dp)
                break

        if rel_path and rel_path != '.':
            out_subdir = os.path.join(output_dir, rel_path)
        else:
            out_subdir = output_dir

        os.makedirs(out_subdir, exist_ok=True)
        output_path = os.path.join(out_subdir, output_name)

        # Save embedding
        np.savez_compressed(
            output_path,
            image_embed=features['image_embed'],
            high_res_feat_0=features['high_res_feat_0'],
            high_res_feat_1=features['high_res_feat_1'],
            mask=mask_nii.astype(np.int16),
            ori_size=np.array(features['orig_hw']),
        )

        # Collect unique labels for the pickle list
        unique_labels = np.unique(mask_nii)
        unique_labels = unique_labels[unique_labels != 0]  # Remove background

        for label in unique_labels:
            embedding_list.append({
                'emb_path': output_path,
                'labels': int(label),
            })

    return embedding_list


def main():
    args = parse_args()

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

    # Setup transforms
    transforms = SAM2Transforms(
        resolution=model.image_size,
        mask_threshold=0.0,
    )

    print(f"Model loaded. Image size: {model.image_size}")
    print(f"Processing data from: {args.data_paths}")
    print(f"Output directory: {args.output_dir}")

    # Process dataset
    embedding_list = process_dataset(
        model=model,
        transforms=transforms,
        data_paths=args.data_paths,
        image_key=args.image_key,
        mask_key=args.mask_key,
        output_dir=args.output_dir,
        device=args.device,
    )

    print(f"Generated {len(embedding_list)} embedding entries")

    # Save pickle list if requested
    if args.save_pickle_list:
        pickle_path = args.pickle_output or os.path.join(args.output_dir, "embedding_list.pkl")

        # Create the format expected by BalancedEmbDataset
        # Split into categories based on some criteria (here we just use "all")
        pickle_data = {
            'train_60_up': embedding_list,  # All samples in one category
            'train_60_down': [],  # Empty for now
        }

        with open(pickle_path, 'wb') as f:
            pickle.dump(pickle_data, f)

        print(f"Saved embedding list to {pickle_path}")

    print("Done!")


if __name__ == "__main__":
    main()
