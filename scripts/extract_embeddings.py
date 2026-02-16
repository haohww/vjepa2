#!/usr/bin/env python3
"""
Extract V-JEPA2 video embeddings for success/failure video pairs.

For each subfolder containing gt.mp4 (success) and fail.mp4 (failure),
this script encodes both videos through V-JEPA2 ViT-g and saves the
resulting embedding pairs.

Usage:
    python -m scripts.extract_embeddings [--model_size giant] [--batch_size 4] [--output_dir outputs/embeddings]

Output:
    A .pt file per video directory containing a dict:
    {
        "folder_name": {
            "success_emb": tensor of shape [embed_dim],
            "failure_emb": tensor of shape [embed_dim],
        },
        ...
    }
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader

# Add project root to path so we can import src.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.hub.backbones import vjepa2_vit_giant, vjepa2_vit_large


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# All video directories to process
VIDEO_DIRS = [
    "dataset/failurev2/datasetv2/datasetv1/videos",
    "dataset/failurev2/datasetv2/gripper/gripper_noise_delay_close/videos",
    "dataset/failurev2/datasetv2/gripper/gripper_noise_force_open/videos",
    "dataset/failurev2/datasetv2/gripper/gripper_noise_weak_close/videos",
]

# Model configurations: maps model name -> (loader function, image size)
MODEL_CONFIGS = {
    "large": (vjepa2_vit_large, 256),
    "giant": (vjepa2_vit_giant, 256),
}

NUM_FRAMES = 64  # V-JEPA2 expects 64 frames; our videos are exactly 64 frames


# ──────────────────────────────────────────────────────────────────────
# Pooling strategies
# ──────────────────────────────────────────────────────────────────────

def mean_pool(patch_features: torch.Tensor) -> torch.Tensor:
    """
    Mean pool over all spatiotemporal patch tokens.

    Args:
        patch_features: [B, num_patches, embed_dim]

    Returns:
        Pooled features: [B, embed_dim]
    """
    return patch_features.mean(dim=1)


def max_pool(patch_features: torch.Tensor) -> torch.Tensor:
    """
    Max pool over all spatiotemporal patch tokens.

    Args:
        patch_features: [B, num_patches, embed_dim]

    Returns:
        Pooled features: [B, embed_dim]
    """
    return patch_features.max(dim=1).values


# TODO: Add attentive pooling strategy here once a pretrained AttentivePooler
# is available for the target domain. The AttentivePooler uses a learned query
# token that cross-attends to all patch features, producing a weighted summary.
#
# Example usage (requires trained pooler weights):
#
#   from src.models.attentive_pooler import AttentivePooler
#
#   def attentive_pool(patch_features, pooler):
#       """
#       Args:
#           patch_features: [B, num_patches, embed_dim]
#           pooler: pretrained AttentivePooler module
#       Returns:
#           Pooled features: [B, embed_dim]
#       """
#       return pooler(patch_features).squeeze(1)  # [B, 1, embed_dim] -> [B, embed_dim]

POOLING_FNS = {
    "mean": mean_pool,
    "max": max_pool,
}


# ──────────────────────────────────────────────────────────────────────
# Video loading and preprocessing
# ──────────────────────────────────────────────────────────────────────

def build_video_transform(img_size: int):
    """
    Build the standard V-JEPA2 eval transform: resize, center crop, normalize.

    Args:
        img_size: target spatial resolution (e.g., 256)

    Returns:
        A composable transform that takes [T, C, H, W] and returns [C, T, H, W].
    """
    short_side_size = int(256.0 / 224 * img_size)
    return video_transforms.Compose([
        video_transforms.Resize(short_side_size, interpolation="bilinear"),
        video_transforms.CenterCrop(size=(img_size, img_size)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])


def load_video(video_path: str, num_frames: int = NUM_FRAMES) -> np.ndarray:
    """
    Load a video file and sample exactly `num_frames` frames.

    If the video has more frames than needed, uniformly subsample.
    If fewer, take all available frames (the model can handle shorter clips,
    but our dataset is already exactly 64 frames).

    Args:
        video_path: path to the .mp4 file
        num_frames: target number of frames

    Returns:
        video: numpy array of shape [T, H, W, C] (uint8)
    """
    vr = VideoReader(video_path)
    total_frames = len(vr)

    if total_frames >= num_frames:
        # Uniformly sample num_frames indices from the full video
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        # Use all available frames
        indices = np.arange(total_frames)

    video = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
    return video


def preprocess_video(video: np.ndarray, transform) -> torch.Tensor:
    """
    Apply V-JEPA2 preprocessing to a raw video.

    Args:
        video: numpy array [T, H, W, C]
        transform: the video transform pipeline

    Returns:
        tensor: [C, T, H, W] ready for batching
    """
    # decord gives [T, H, W, C], we need [T, C, H, W] for the transform
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)  # [T, C, H, W]
    video_tensor = transform(video_tensor)  # [C, T, H, W]
    return video_tensor


# ──────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────

def load_model(model_size: str = "giant", device: str = "cuda"):
    """
    Load the V-JEPA2 encoder.

    The checkpoint is auto-downloaded from Meta's servers on first use
    and cached in ~/.cache/torch/hub/checkpoints/.

    Args:
        model_size: "large" (ViT-L, 300M) or "giant" (ViT-g, 1B)
        device: target device

    Returns:
        encoder: the V-JEPA2 encoder (nn.Module)
        img_size: spatial resolution for preprocessing
        embed_dim: output embedding dimension
    """
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size '{model_size}'. Choose from: {list(MODEL_CONFIGS.keys())}")

    loader_fn, img_size = MODEL_CONFIGS[model_size]

    print(f"Loading V-JEPA2 {model_size} (img_size={img_size})...")
    encoder, _predictor = loader_fn(pretrained=True)
    encoder = encoder.to(device).eval()

    # Read embed_dim from the model itself (e.g., 1024 for ViT-L, 1408 for ViT-g)
    embed_dim = encoder.embed_dim
    num_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    print(f"Encoder loaded: {num_params:.0f}M params, embed_dim={embed_dim}")

    return encoder, img_size, embed_dim


# ──────────────────────────────────────────────────────────────────────
# Batch embedding extraction
# ──────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def extract_embeddings_batch(
    encoder,
    video_paths: list,
    transform,
    pool_fn,
    device: str = "cuda",
) -> list:
    """
    Extract pooled embeddings for a batch of videos.

    Args:
        encoder: V-JEPA2 encoder
        video_paths: list of paths to video files
        transform: video preprocessing transform
        pool_fn: pooling function (mean_pool, max_pool, etc.)
        device: target device

    Returns:
        list of embedding tensors, each [embed_dim] on CPU
    """
    tensors = []
    for path in video_paths:
        video = load_video(path)
        tensor = preprocess_video(video, transform)
        tensors.append(tensor)

    # Stack into batch: [B, C, T, H, W]
    batch = torch.stack(tensors, dim=0).to(device)

    # Forward pass through encoder -> [B, num_patches, embed_dim]
    patch_features = encoder(batch)

    # Pool to [B, embed_dim]
    embeddings = pool_fn(patch_features)

    # Return as list of CPU tensors
    return [emb.cpu() for emb in embeddings]


def process_video_directory(
    encoder,
    video_dir: str,
    transform,
    pool_fn,
    batch_size: int = 4,
    limit: int = None,
    device: str = "cuda",
) -> dict:
    """
    Process all subfolders in a video directory, extracting embedding pairs.

    Each subfolder should contain gt.mp4 (success) and fail.mp4 (failure).

    Args:
        encoder: V-JEPA2 encoder
        video_dir: path to the videos/ directory
        transform: video preprocessing transform
        pool_fn: pooling function
        batch_size: number of videos to process at once
        device: target device

    Returns:
        dict mapping folder_name -> {"success_emb": tensor, "failure_emb": tensor}
    """
    subfolders = sorted([
        f for f in os.listdir(video_dir)
        if os.path.isdir(os.path.join(video_dir, f))
    ])

    results = {}
    # Collect all valid (gt, fail) video pairs
    pairs = []
    for folder_name in subfolders:
        gt_path = os.path.join(video_dir, folder_name, "gt.mp4")
        fail_path = os.path.join(video_dir, folder_name, "fail.mp4")
        if os.path.exists(gt_path) and os.path.exists(fail_path):
            pairs.append((folder_name, gt_path, fail_path))
        else:
            # Some folders might only have gt.mp4 (no failure)
            if not os.path.exists(gt_path):
                print(f"  [WARN] Missing gt.mp4 in {folder_name}, skipping")
            if not os.path.exists(fail_path):
                print(f"  [WARN] Missing fail.mp4 in {folder_name}, skipping")

    print(f"  Found {len(pairs)} valid video pairs out of {len(subfolders)} subfolders")

    if limit is not None and limit > 0:
        print(f"  Limiting to first {limit} pairs")
        pairs = pairs[:limit]

    # Process in batches: we batch gt and fail videos together
    # Each "pair" contributes 2 videos, so effective batch = batch_size pairs = 2*batch_size videos
    for start_idx in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start_idx : start_idx + batch_size]

        # Collect all video paths for this batch (interleaved: gt, fail, gt, fail, ...)
        all_paths = []
        for _, gt_path, fail_path in batch_pairs:
            all_paths.append(gt_path)
            all_paths.append(fail_path)

        # Extract embeddings for the whole batch
        embeddings = extract_embeddings_batch(encoder, all_paths, transform, pool_fn, device)

        # Unpack: even indices are success, odd indices are failure
        for i, (folder_name, _, _) in enumerate(batch_pairs):
            results[folder_name] = {
                "success_emb": embeddings[2 * i],
                "failure_emb": embeddings[2 * i + 1],
            }

        processed = min(start_idx + batch_size, len(pairs))
        print(f"  Processed {processed}/{len(pairs)} pairs", end="\r")

    print()  # newline after progress
    return results


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract V-JEPA2 embeddings for success/failure video pairs"
    )
    parser.add_argument(
        "--model_size", type=str, default="giant",
        choices=list(MODEL_CONFIGS.keys()),
        help="V-JEPA2 model variant (default: giant)"
    )
    parser.add_argument(
        "--pooling", type=str, default="mean",
        choices=list(POOLING_FNS.keys()),
        help="Pooling strategy for patch features (default: mean)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Number of video PAIRS per batch (actual videos = 2x this). "
             "Reduce if OOM. (default: 4, i.e. 8 videos at a time)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of video pairs to process per directory (for testing)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/embeddings",
        help="Directory to save embedding files (default: outputs/embeddings)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on (default: cuda)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup output directory
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model (checkpoint auto-downloaded on first run)
    encoder, img_size, embed_dim = load_model(args.model_size, args.device)
    print(f"Using pooling strategy: {args.pooling}")
    print(f"Output directory: {output_dir}")
    print()

    # Build preprocessing transform
    transform = build_video_transform(img_size)

    # Select pooling function
    pool_fn = POOLING_FNS[args.pooling]

    # Process each video directory
    all_embeddings = {}
    for video_dir_rel in VIDEO_DIRS:
        video_dir = str(PROJECT_ROOT / video_dir_rel)
        # Derive a short label for the directory (e.g., "datasetv1" or "gripper_noise_delay_close")
        # videos/ parent is e.g. "datasetv1" or "gripper_noise_delay_close"
        dir_label = Path(video_dir).parent.name

        print(f"Processing: {video_dir_rel}")
        print(f"  Label: {dir_label}")

        start_time = time.time()
        results = process_video_directory(
            encoder, video_dir, transform, pool_fn,
            batch_size=args.batch_size, limit=args.limit, device=args.device,
        )
        elapsed = time.time() - start_time

        print(f"  Done: {len(results)} pairs in {elapsed:.1f}s "
              f"({elapsed / max(len(results), 1):.2f}s per pair)")

        # Save per-directory results
        save_path = output_dir / f"{dir_label}_embeddings.pt"
        torch.save({
            "model_size": args.model_size,
            "pooling": args.pooling,
            "embed_dim": embed_dim,
            "num_pairs": len(results),
            "embeddings": results,
        }, save_path)
        print(f"  Saved to: {save_path}")
        print()

        all_embeddings[dir_label] = results

    # Also save a combined file with all embeddings
    combined_path = output_dir / "all_embeddings.pt"
    torch.save({
        "model_size": args.model_size,
        "pooling": args.pooling,
        "embed_dim": embed_dim,
        "total_pairs": sum(len(v) for v in all_embeddings.values()),
        "embeddings": all_embeddings,
    }, combined_path)
    print(f"Combined embeddings saved to: {combined_path}")
    print(f"Total pairs: {sum(len(v) for v in all_embeddings.values())}")


if __name__ == "__main__":
    main()
