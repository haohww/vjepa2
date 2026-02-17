# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

V-JEPA 2 is a self-supervised video model that learns to understand, predict, and plan from video data. The project includes:
- **V-JEPA 2**: Video encoder pre-trained via masked latent feature prediction
- **V-JEPA 2-AC**: Action-conditioned world model for robot manipulation (post-trained from V-JEPA 2)

## Development Setup

```bash
# Activate the project environment
conda activate jepa
pip install -e .  # Development mode (if not already installed)
```

**macOS Note**: This project uses `decord` for video loading, which doesn't support macOS. Users may need alternative implementations like `eva-decord` or `decord2`.

## Common Commands

### Code Quality
```bash
# Format code
black .
isort .

# Lint
flake8 .

# Run tests
pytest tests/
```

**Code Style**:
- Line length: 119 characters
- Use black profile for isort
- See `.flake8` and `pyproject.toml` for detailed rules

### Training

#### Local Training (V-JEPA 2)
```bash
# Pretraining
python -m app.main --fname configs/train/vitl16/pretrain-256px-16f.yaml \
  --devices cuda:0

# Action-conditioned post-training
python -m app.main --fname configs/train/vitg16/droid-256px-8f.yaml \
  --devices cuda:0
```

#### Distributed Training (SLURM)
```bash
python -m app.main_distributed \
  --fname configs/train/vitl16/pretrain-256px-16f.yaml \
  --time 6000 \
  --account my_account --qos=my_qos
```

**Important**: Update config file paths (`folder`, `datasets`, `checkpoint`) to match your filesystem before training.

### Evaluation

#### Training Attentive Probes
```bash
# Local
python -m evals.main --fname configs/eval/vitl/ssv2.yaml \
  --devices cuda:0 cuda:1

# Distributed
python -m evals.main_distributed \
  --fname configs/eval/vitl/ssv2.yaml \
  --time 8600 \
  --account my_account --qos=my_qos
```

#### Inference from Pretrained Probes
1. Download checkpoint and rename to `latest.pt`
2. Place in folder matching config: `[folder]/[eval_name]/[tag]/latest.pt`
3. Run with inference config: `python -m evals.main --fname configs/inference/vitl/ssv2.yaml`

### Model Loading

#### PyTorch Hub
```python
import torch

# Preprocessor
processor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_preprocessor')

# Models
vjepa2_vit_large = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')
vjepa2_vit_huge = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_huge')
vjepa2_vit_giant = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_giant')

# Action-conditioned
encoder, ac_predictor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_ac_vit_giant')
```

#### HuggingFace
```python
from transformers import AutoVideoProcessor, AutoModel

model = AutoModel.from_pretrained("facebook/vjepa2-vitg-fpc64-256")
processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitg-fpc64-256")
```

### Custom Scripts

#### Extract Video Embeddings
```bash
python -m scripts.extract_embeddings \
  --model_size giant \
  --batch_size 4 \
  --output_dir outputs/embeddings
```

## Architecture

### High-Level Structure

**Training Pipeline**:
1. `app/main.py` or `app/main_distributed.py` - Entry points
2. `app/scaffold.py` - Imports and launches the appropriate training module
3. `app/vjepa/train.py` or `app/vjepa_droid/train.py` - Main training loops
4. Config files (YAML) specify all hyperparameters and paths

**Two Training Modes**:
- `app: vjepa` - Standard V-JEPA 2 pretraining and cooldown
- `app: vjepa_droid` - Action-conditioned post-training on robot data

### Core Components

**Models** (`src/models/`):
- `vision_transformer.py` - ViT encoder (supports video via 3D patch embedding)
- `predictor.py` - Predictor for masked feature prediction
- `ac_predictor.py` - Action-conditioned predictor for robotics
- `attentive_pooler.py` - Attentive probe for downstream classification

**Data** (`src/datasets/`):
- `video_dataset.py` - Primary video dataset with temporal sampling
- `imagenet1k.py` - Image classification dataset
- `data_manager.py` - Factory for creating data loaders
- Supports webdataset format and CSV-based video paths

**Masking** (`src/masks/`):
- `multiseq_multiblock3d.py` - Spatiotemporal block masking for V-JEPA
- Configurable aspect ratios, spatial/temporal scales, and number of blocks

**Evaluations** (`evals/`):
- `video_classification_frozen/` - Video understanding (SSv2, Diving48)
- `action_anticipation_frozen/` - Action anticipation (EPIC-KITCHENS-100)
- `image_classification_frozen/` - Image understanding
- All use frozen encoder + trained attentive probe

### Config File Structure

YAML configs control all training/evaluation parameters:
```yaml
app: vjepa                    # Training mode: vjepa or vjepa_droid
meta:                         # Training meta-params
  dtype: bfloat16            # Use mixed precision
  use_sdpa: true             # Scaled dot-product attention
model:
  model_name: vit_large      # Encoder size
  pred_depth: 12             # Predictor depth
  use_rope: true             # Rotary position embeddings
data:
  dataset_type: VideoDataset
  batch_size: 24
  crop_size: 256
  patch_size: 16
  fps: 4                     # Target frame rate
mask:                        # List of mask configurations
  - num_blocks: 8
    spatial_scale: [0.15, 0.15]
optimization:
  lr: 0.000525
  epochs: 10
```

### Model Variants

Three encoder sizes available:
- **ViT-L/16**: 300M parameters
- **ViT-H/16**: 600M parameters
- **ViT-g/16**: 1B parameters (also available at 384px resolution)

All use:
- 3D patch embedding (spatiotemporal tokenization)
- Sinusoidal position embeddings or RoPE
- Optional SiLU activations
- Activation checkpointing for memory efficiency

### Distributed Training

- Uses PyTorch DDP for multi-GPU training
- Local: Spawns processes per GPU via multiprocessing
- Distributed: Launches SLURM jobs via submitit
- Each process sees only one CUDA device (via `CUDA_VISIBLE_DEVICES`)

### Action-Conditioned Model (V-JEPA 2-AC)

Post-training process:
1. Start from pretrained V-JEPA 2 encoder (typically ViT-g)
2. Train action-conditioned predictor on robot trajectory data (DROID dataset)
3. Learn to predict future video states given actions
4. Use energy landscape for planning (see `notebooks/energy_landscape_example.ipynb`)

## Key Files to Modify

When adding features or fixing bugs:
- Training loops: `app/vjepa/train.py`, `app/vjepa_droid/train.py`
- Model architectures: `src/models/vision_transformer.py`, `src/models/predictor.py`
- Data loading: `src/datasets/video_dataset.py`, `src/datasets/data_manager.py`
- Transforms: `app/vjepa/transforms.py`, `app/vjepa_droid/transforms.py`
- Model utilities: `src/models/utils/` (attention, patch embedding, position embeddings)

## Testing

Unit tests in `tests/` cover:
- `tests/models/` - Model components (ViT, predictor)
- `tests/datasets/` - Data loading and transforms

Run specific test:
```bash
pytest tests/models/test_vision_transformer.py -v
```
