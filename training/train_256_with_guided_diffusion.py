#!/usr/bin/env python3
"""
256x256 Ultrasound Diffusion Model Training with Guided-Diffusion
High-Quality Training with Advanced Techniques:
- EMA (Exponential Moving Average)
- Advanced Data Augmentation
- Cosine Annealing with Warmup
- Gradient Clipping
- Mixed Precision Training
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths  # registers code subdirs on sys.path


import argparse
import math
import os
import sys
from pathlib import Path
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
from tqdm import tqdm
import numpy as np
import logging
import copy
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

print("Importing guided-diffusion modules...", flush=True)
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
)
print("guided_diffusion.script_util imported successfully", flush=True)

# TrainLoop is not used in this script
# from guided_diffusion.train_util import TrainLoop

print("All imports completed", flush=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)


def str2bool(value):
    """Parse argparse booleans robustly; argparse type=bool treats 'False' as True."""
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")

# Print immediately to verify script is running
print("=" * 80, flush=True)
print("SCRIPT STARTED - Ultrasound Diffusion Training", flush=True)
print("=" * 80, flush=True)

# ==============================================================================
# EMA (Exponential Moving Average)
# ==============================================================================

class EMA:
    """Exponential Moving Average for model parameters"""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# ==============================================================================
# ADVANCED DATA AUGMENTATION
# ==============================================================================

DEFAULT_SAMPLE_INPUT_PATHS = []


class UltrasoundAugmentation:
    """Advanced augmentation for ultrasound images"""

    def __init__(self, image_size=256, strength='medium'):
        self.image_size = image_size
        self.strength = strength

        if strength == 'light':
            rotation = 5
            translate = 0.05
            scale = (0.95, 1.05)
        elif strength == 'medium':
            rotation = 10
            translate = 0.1
            scale = (0.9, 1.1)
        else:  # strong
            rotation = 15
            translate = 0.15
            scale = (0.85, 1.15)

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=rotation),
            transforms.RandomAffine(
                degrees=0,
                translate=(translate, translate),
                scale=scale,
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
        ])

    def __call__(self, image):
        return self.transform(image)

# ==============================================================================
# DATASET
# ==============================================================================

class UltrasoundDataset(Dataset):
    """Ultrasound dataset for existing clean 256x256 images.

    DRUS/PICMUS prior images are already 256x256. Keep them at their native
    size and only resize a file if it is unexpectedly not the requested size.
    """

    def __init__(self, data_dir, image_size=256, augment=True, augment_strength='medium'):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.image_paths = sorted(self.data_dir.glob("**/*.png"))
        self.augment = augment
        self.augment_strength = augment_strength

        log.info(f"Found {len(self.image_paths)} images in {data_dir}")

        self.target_size = (image_size, image_size)  # PIL size is (W, H)

        # Base transform without resizing; DRUS images are expected to be 256x256.
        self.to_tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # [0, 1] -> [-1, 1]
        ])

        # Advanced augmentation (applied BEFORE to_tensor)
        if augment:
            self.augment_transform = UltrasoundAugmentation(
                image_size=image_size,
                strength=augment_strength
            )
        else:
            self.augment_transform = None

        log.info(f"Augmentation: {'enabled (' + augment_strength + ')' if augment else 'disabled'}")

    def __len__(self):
        return len(self.image_paths)

    def _resize_if_needed(self, image):
        if image.size == self.target_size:
            return image
        return image.resize(self.target_size, Image.Resampling.LANCZOS)

    def __getitem__(self, idx):
        try:
            # Load image as grayscale
            image = Image.open(self.image_paths[idx]).convert('L')

            # Apply augmentation FIRST (on PIL Image)
            if self.augment and self.augment_transform is not None:
                image = self.augment_transform(image)

            # DRUS stays native 256x256; non-256 files are resized as a fallback.
            image = self._resize_if_needed(image)
            tensor = self.to_tensor_transform(image)

            # Return format expected by training loop
            return tensor, {}
        except Exception as e:
            log.warning(f"Error loading {self.image_paths[idx]}: {e}")
            # Return dummy data on error
            return torch.zeros(1, self.image_size, self.image_size), {}


def build_training_dataset(args):
    """Build the DRUS prior dataset and return a source summary."""
    drus_dataset = UltrasoundDataset(
        args.data_dir,
        image_size=args.image_size,
        augment=args.augment,
        augment_strength=args.augment_strength,
    )
    summary = {
        "drus_dir": str(Path(args.data_dir).resolve()),
        "drus_count": len(drus_dataset),
        "drus_sampling": "native 256x256 PNG; resize only if an input file is not image_size",
        "drus_first_paths": [str(p) for p in drus_dataset.image_paths[:10]],
        "total_effective_count": len(drus_dataset),
    }
    return drus_dataset, summary


def write_source_summary(output_dir, summary):
    out_path = os.path.join(output_dir, "training_sources.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Training source summary saved: {out_path}")


def save_training_batch_preview(batch, output_dir, filename="training_batch_check.png"):
    preview = (batch.detach().cpu() + 1.0) / 2.0
    preview = preview.clamp(0, 1)
    nrow = min(4, preview.shape[0])
    out_path = os.path.join(output_dir, "samples", filename)
    save_image(preview, out_path, nrow=nrow, normalize=False)
    log.info(f"Training batch preview saved: {out_path}")


def load_sample_inputs(paths, image_size):
    """Load fixed sample inputs for input-seeded sampling."""
    to_tensor = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
    ])
    tensors = []
    names = []
    for path in paths:
        if not os.path.exists(path):
            log.warning(f"Sample input not found, skipping: {path}")
            continue
        try:
            image = Image.open(path).convert('L')
            tensor = to_tensor(image) * 2 - 1  # [0, 1] -> [-1, 1]
            tensors.append(tensor)
            names.append(Path(path).stem)
        except Exception as e:
            log.warning(f"Failed to load sample input {path}: {e}")
    if not tensors:
        return None, []
    return torch.stack(tensors, dim=0), names


def save_input_seeded_samples(model, diffusion, input_batch, output_dir, global_step, tag=""):
    """Generate samples by noising fixed inputs to x_T and denoising."""
    if input_batch is None:
        return
    device = next(model.parameters()).device
    inputs = input_batch.to(device)
    t_batch = torch.full((inputs.shape[0],), diffusion.num_timesteps - 1, device=device, dtype=torch.long)
    x_t = diffusion.q_sample(x_start=inputs, t=t_batch)

    samples = diffusion.p_sample_loop(
        model,
        inputs.shape,
        noise=x_t,
        clip_denoised=True,
        progress=False,
    )

    inputs_vis = (inputs + 1) / 2
    samples_vis = (samples + 1) / 2
    nrow = inputs.shape[0]
    grid_inputs = make_grid(inputs_vis, nrow=nrow, padding=2, normalize=False)
    grid_outputs = make_grid(samples_vis, nrow=nrow, padding=2, normalize=False)
    grid = torch.cat([grid_inputs, grid_outputs], dim=1)
    out_path = os.path.join(
        output_dir, 'samples', f'samples_input_step_{global_step:07d}{tag}.png'
    )
    save_image(grid, out_path)
    log.info(f"✓ Input-seeded samples saved: {out_path}")


def create_grayscale_from_rgb_weights(state_dict):
    """
    Convert RGB pretrained model to grayscale
    """
    log.info("Converting RGB pretrained weights to grayscale...")

    # Convert input layer: (out, 3, k, k) -> (out, 1, k, k)
    if 'input_blocks.0.0.weight' in state_dict:
        rgb_weight = state_dict['input_blocks.0.0.weight']
        if rgb_weight.shape[1] == 3:
            # Average across RGB channels
            state_dict['input_blocks.0.0.weight'] = rgb_weight.mean(dim=1, keepdim=True)
            log.info(f"✓ Converted input layer: {rgb_weight.shape} -> {state_dict['input_blocks.0.0.weight'].shape}")

    # Convert output layer: (6, out, k, k) -> (2, out, k, k)  [if learn_sigma=True]
    if 'out.2.weight' in state_dict:
        out_weight = state_dict['out.2.weight']
        if out_weight.shape[0] == 6:  # RGB with variance (3*2)
            # Reshape to (2, 3, ...) and average RGB
            state_dict['out.2.weight'] = out_weight.view(2, 3, *out_weight.shape[1:]).mean(dim=1)
            log.info(f"✓ Converted output layer: {out_weight.shape} -> {state_dict['out.2.weight'].shape}")

        if 'out.2.bias' in state_dict:
            out_bias = state_dict['out.2.bias']
            if out_bias.shape[0] == 6:
                state_dict['out.2.bias'] = out_bias.view(2, 3).mean(dim=1)

    return state_dict


# ==============================================================================
# LEARNING RATE SCHEDULER
# ==============================================================================

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-7):
    """Cosine annealing with warmup"""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine annealing
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(min_lr / optimizer.defaults['lr'], cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_p2_weights(t, diffusion, p2_k, p2_gamma):
    """Compute P2 loss weights for diffusion timesteps."""
    alphas_cumprod = torch.from_numpy(diffusion.alphas_cumprod).to(t.device).float()
    alpha_t = alphas_cumprod[t]
    snr = alpha_t / (1.0 - alpha_t)
    return (p2_k + snr) ** (-p2_gamma)


def compute_weighted_loss(losses, t, diffusion, args):
    if "mse" in losses:
        if args.use_p2_weighting:
            weights = compute_p2_weights(t, diffusion, args.p2_k, args.p2_gamma)
            mse_term = (losses["mse"] * weights).mean()
        else:
            mse_term = losses["mse"].mean()
        loss = mse_term
        if "vb" in losses:
            loss = loss + 0.001 * losses["vb"].mean()
        return loss
    return losses["loss"].mean()


def accumulation_window(batch_index, batches_per_epoch, grad_accum_steps):
    """Return the divisor and boundary flag for one accumulation micro-step."""
    group_start = (batch_index // grad_accum_steps) * grad_accum_steps
    count = min(grad_accum_steps, batches_per_epoch - group_start)
    boundary = (
        (batch_index + 1) % grad_accum_steps == 0
        or (batch_index + 1) == batches_per_epoch
    )
    return count, boundary


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def main():
    print("\n" + "="*80, flush=True)
    print("MAIN FUNCTION STARTED", flush=True)
    print("="*80 + "\n", flush=True)

    parser = argparse.ArgumentParser(description="256x256 Guided-Diffusion Training")

    # Dataset arguments
    parser.add_argument("--data_dir", type=str,
                        default=str(paths.DRUS_TRAIN_DIR),
                        help="Path to training data")
    parser.add_argument("--output_dir", type=str,
                        default=str(paths.GD_OUTPUT_DAS),
                        help="Output directory for checkpoints and samples")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Per-step micro-batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Micro-batches accumulated per optimizer step")
    parser.add_argument("--allow_batch_override", action="store_true",
                        help="Allow an effective batch size other than the paper value")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--augment", type=str2bool, default=True,
                        help="Use data augmentation")
    parser.add_argument("--augment_strength", type=str, default='medium',
                        choices=['light', 'medium', 'strong'],
                        help="Augmentation strength")
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--save_epoch_checkpoints", type=str2bool, default=True,
                        help="Save model/EMA checkpoint at the end of each epoch")
    parser.add_argument("--save_final_checkpoint", type=str2bool, default=True,
                        help="Save model/EMA final checkpoint after training")
    parser.add_argument("--enable_input_sampling", type=str2bool, default=True,
                        help="Enable input-seeded sampling at checkpoint time")
    parser.add_argument("--sample_input_paths", type=str, nargs='*',
                        default=DEFAULT_SAMPLE_INPUT_PATHS,
                        help="Fixed input images for input-seeded sampling")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Stop after this many optimizer steps (0 = full epochs)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker count")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for dataloader shuffle")
    parser.add_argument("--verify_data_only", action="store_true",
                        help="Build the dataset, save one batch preview, then exit before model creation")

    # Advanced training techniques
    parser.add_argument("--use_ema", type=str2bool, default=True,
                        help="Use Exponential Moving Average")
    parser.add_argument("--ema_decay", type=float, default=0.9999,
                        help="EMA decay rate")
    parser.add_argument("--use_warmup", type=str2bool, default=True,
                        help="Use learning rate warmup")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Number of warmup steps")
    parser.add_argument("--gradient_clip", type=float, default=1.0,
                        help="Gradient clipping value (0 to disable)")
    parser.add_argument("--use_amp", type=str2bool, default=False,
                        help="Use automatic mixed precision")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW")
    parser.add_argument("--use_p2_weighting", type=str2bool, default=True,
                        help="Enable P2 loss weighting for MSE term")
    parser.add_argument("--p2_gamma", type=float, default=1.0,
                        help="P2 loss weighting gamma (Nichol & Dhariwal)")
    parser.add_argument("--p2_k", type=float, default=1.0,
                        help="P2 loss weighting k (Nichol & Dhariwal)")

    # Pretrained model
    parser.add_argument("--pretrained_path", type=str,
                        default=str(paths.BASE_DIFFUSION_CKPT),
                        help="Path to pretrained checkpoint")
    parser.add_argument("--use_pretrained", type=str2bool, default=True,
                        help="Use pretrained weights")
    parser.add_argument("--convert_to_grayscale", type=str2bool, default=True,
                        help="Convert RGB pretrained weights to grayscale")

    # Add guided-diffusion defaults
    defaults = model_and_diffusion_defaults()
    defaults.update({
        'in_channels': 1,
        'image_size': 256,
        'num_channels': 256,
        'num_res_blocks': 2,
        'num_heads': 4,
        'num_heads_upsample': -1,
        'num_head_channels': 64,
        'attention_resolutions': '32,16,8',
        'channel_mult': '',
        'dropout': 0.0,
        'class_cond': False,
        'use_checkpoint': False,
        'use_scale_shift_norm': True,
        'resblock_updown': True,
        'use_fp16': False,
        'use_new_attention_order': False,
        'learn_sigma': True,
        'diffusion_steps': 1000,
        'noise_schedule': 'linear',
        'timestep_respacing': '',
        'use_kl': False,
        'predict_xstart': False,
        'rescale_timesteps': False,
        'rescale_learned_sigmas': False,
    })

    add_dict_to_argparser(parser, defaults)

    print("Parsing arguments...", flush=True)
    args = parser.parse_args()
    print(f"Arguments parsed successfully", flush=True)

    if args.batch_size < 1 or args.grad_accum_steps < 1:
        parser.error("--batch_size and --grad_accum_steps must both be positive")
    effective_batch = args.batch_size * args.grad_accum_steps
    if effective_batch != 8 and not args.allow_batch_override:
        parser.error(
            f"effective batch is {effective_batch}, but the paper configuration is 8; "
            "pass --allow_batch_override to train with a different effective batch"
        )

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'tensorboard'), exist_ok=True)

    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Initialize TensorBoard writer
    tensorboard_dir = os.path.join(args.output_dir, 'tensorboard')
    writer = SummaryWriter(tensorboard_dir)
    log.info(f"TensorBoard logs will be saved to: {tensorboard_dir}")
    log.info(f"To view, run: tensorboard --logdir={tensorboard_dir}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    # Create dataset
    log.info("Creating dataset...")
    dataset, source_summary = build_training_dataset(args)
    write_source_summary(args.output_dir, source_summary)
    log.info(f"DRUS clean images: {source_summary['drus_count']}")
    log.info(f"Total effective training samples: {source_summary['total_effective_count']}")

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    persistent_workers = args.num_workers > 0 and torch.cuda.is_available()
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda',
        drop_last=True,
        persistent_workers=persistent_workers,
        generator=generator,
    )

    log.info(f"Training dataset: {len(dataset)} images")
    log.info(f"Batches per epoch: {len(dataloader)}")

    # Verify a batch before model creation so source mistakes are caught cheaply.
    log.info("Checking one training batch...")
    sample_batch_cpu = next(iter(dataloader))[0][: min(args.batch_size, 16)]
    log.info(
        "Batch check: "
        f"shape={tuple(sample_batch_cpu.shape)}, "
        f"min={sample_batch_cpu.min().item():.3f}, "
        f"max={sample_batch_cpu.max().item():.3f}, "
        f"mean={sample_batch_cpu.mean().item():.3f}"
    )
    writer.add_images('Training_Data/Sample_Batch', (sample_batch_cpu[:4] + 1) / 2, 0)
    save_training_batch_preview(sample_batch_cpu, args.output_dir)
    if args.verify_data_only:
        log.info("verify_data_only set; exiting before model creation.")
        writer.close()
        return

    # Load fixed input images for sampling
    sample_input_batch = None
    sample_input_names = []
    if args.enable_input_sampling:
        sample_input_batch, sample_input_names = load_sample_inputs(
            args.sample_input_paths,
            args.image_size
        )
        if sample_input_batch is not None:
            log.info(f"Loaded {len(sample_input_names)} input images for sampling")
        else:
            log.info("No valid input images for sampling; input-seeded sampling disabled")
    else:
        log.info("Input-seeded sampling disabled by flag")

    # Create model and diffusion
    log.info("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        in_channels=args.in_channels,
        image_size=args.image_size,
        class_cond=args.class_cond,
        learn_sigma=args.learn_sigma,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        channel_mult=args.channel_mult,
        num_heads=args.num_heads,
        num_head_channels=args.num_head_channels,
        num_heads_upsample=args.num_heads_upsample,
        attention_resolutions=args.attention_resolutions,
        dropout=args.dropout,
        diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        timestep_respacing=args.timestep_respacing,
        use_kl=args.use_kl,
        predict_xstart=args.predict_xstart,
        rescale_timesteps=args.rescale_timesteps,
        rescale_learned_sigmas=args.rescale_learned_sigmas,
        use_checkpoint=args.use_checkpoint,
        use_scale_shift_norm=args.use_scale_shift_norm,
        resblock_updown=args.resblock_updown,
        use_fp16=args.use_fp16,
        use_new_attention_order=args.use_new_attention_order,
    )

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {total_params:,}")

    # Load pretrained weights
    if args.use_pretrained and os.path.exists(args.pretrained_path):
        log.info(f"\n{'='*80}")
        log.info(f"Loading pretrained weights: {args.pretrained_path}")
        log.info(f"{'='*80}")

        try:
            state_dict = torch.load(args.pretrained_path, map_location=device)

            # Convert RGB to grayscale if needed
            if args.convert_to_grayscale:
                state_dict = create_grayscale_from_rgb_weights(state_dict)

            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            log.info(f"✓ Pretrained weights loaded successfully")
            if len(missing) > 0:
                log.info(f"  Missing keys: {len(missing)}")
                log.debug(f"  {missing}")
            if len(unexpected) > 0:
                log.info(f"  Unexpected keys: {len(unexpected)}")
                log.debug(f"  {unexpected}")

        except Exception as e:
            log.error(f"Failed to load pretrained weights: {e}")
            log.info("Training from scratch...")
    else:
        log.info("Training from scratch (no pretrained weights)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    optimizer_steps_per_epoch = math.ceil(len(dataloader) / args.grad_accum_steps)
    num_training_steps = (
        args.max_steps
        if args.max_steps > 0
        else optimizer_steps_per_epoch * args.num_epochs
    )
    scheduler = None
    if args.use_warmup:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=num_training_steps
        )
        log.info(f"Using cosine annealing with {args.warmup_steps} warmup steps")

    # EMA
    ema = None
    if args.use_ema:
        ema = EMA(model, decay=args.ema_decay)
        log.info(f"Using EMA with decay={args.ema_decay}")

    # Mixed precision training
    scaler = None
    if args.use_amp and device.type == 'cuda':
        scaler = GradScaler()
        log.info("Using automatic mixed precision (AMP)")

    # Training Loop
    log.info("=" * 80)
    log.info("Starting Training...")
    log.info(f"Total epochs: {args.num_epochs}")
    log.info(f"Micro-batch size: {args.batch_size}")
    log.info(f"Gradient accumulation steps: {args.grad_accum_steps}")
    log.info(f"Effective batch size: {effective_batch}")
    log.info(f"Learning rate: {args.lr}")
    if args.max_steps > 0:
        log.info(f"Max optimizer steps: {args.max_steps}")
    log.info(f"Training samples: {len(dataset)}")
    log.info(f"Batches per epoch: {len(dataloader)}")
    log.info("=" * 80)
    log.info("")

    global_step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Log model graph and sample training images to TensorBoard
    log.info("Sample training images were logged during dataset check")

    stop_training = False
    for epoch in range(args.num_epochs):
        log.info(f"\n{'='*80}")
        log.info(f"Epoch {epoch+1}/{args.num_epochs}")
        log.info(f"{'='*80}")

        epoch_loss = 0.0
        epoch_batches = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for batch_idx, (batch, _) in enumerate(progress_bar):
            epoch_batches = batch_idx + 1
            batch = batch.to(device)

            # Sample random timesteps
            t = torch.randint(0, args.diffusion_steps, (batch.shape[0],), device=device).long()

            # Forward pass with optional AMP
            if scaler is not None:
                with autocast():
                    losses = diffusion.training_losses(model, batch, t)
                    loss = compute_weighted_loss(losses, t, diffusion, args)
            else:
                losses = diffusion.training_losses(model, batch, t)
                loss = compute_weighted_loss(losses, t, diffusion, args)

            accumulation_count, accumulation_boundary = accumulation_window(
                batch_idx, len(dataloader), args.grad_accum_steps
            )
            scaled_loss = loss / accumulation_count

            grad_norm = 0.0
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # Metrics that are naturally counted per micro-batch.
            epoch_loss += loss.item()

            if not accumulation_boundary:
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'optimizer_step': global_step,
                    'accum': f'{(batch_idx % args.grad_accum_steps) + 1}/{accumulation_count}',
                })
                continue

            if scaler is not None:
                if args.gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.gradient_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                else:
                    # Calculate grad norm even if not clipping
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    grad_norm = total_norm ** 0.5
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            # Update EMA
            if ema is not None:
                ema.update(model)

            # Update learning rate
            if scheduler is not None:
                scheduler.step()

            # global_step counts optimizer updates, not micro-batches.
            global_step += 1

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # Log to TensorBoard
            writer.add_scalar('Training/Loss', loss.item(), global_step)
            writer.add_scalar('Training/Loss_Epoch_Avg', epoch_loss/epoch_batches, global_step)
            writer.add_scalar('Training/Learning_Rate', current_lr, global_step)
            writer.add_scalar('Training/Gradient_Norm', grad_norm, global_step)

            # Log additional diffusion losses if available
            if 'mse' in losses:
                writer.add_scalar('Training/MSE', losses['mse'].mean().item(), global_step)
            if 'vb' in losses:
                writer.add_scalar('Training/VLB', losses['vb'].mean().item(), global_step)

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss/epoch_batches:.4f}',
                'lr': f'{current_lr:.2e}',
                'step': global_step
            })

            # Save checkpoint (with EMA if available)
            if global_step % args.save_interval == 0:
                # Save regular model
                checkpoint_path = os.path.join(
                    args.output_dir, 'checkpoints',
                    f'model_step_{global_step:07d}.pt'
                )
                torch.save(model.state_dict(), checkpoint_path)
                log.info(f"\n✓ Checkpoint saved: {checkpoint_path}")

                # Save EMA model
                if ema is not None:
                    ema_checkpoint_path = os.path.join(
                        args.output_dir, 'checkpoints',
                        f'model_step_{global_step:07d}_ema.pt'
                    )
                    ema.apply_shadow(model)
                    torch.save(model.state_dict(), ema_checkpoint_path)
                    ema.restore(model)
                    log.info(f"✓ EMA checkpoint saved: {ema_checkpoint_path}")

                # Input-seeded sampling at checkpoint time
                if args.enable_input_sampling and sample_input_batch is not None:
                    log.info(f"\n{'='*40}")
                    log.info(f"Input-seeded sampling at step {global_step}...")
                    log.info(f"{'='*40}")
                    model.eval()
                    with torch.no_grad():
                        save_input_seeded_samples(
                            model,
                            diffusion,
                            sample_input_batch,
                            args.output_dir,
                            global_step,
                            tag="",
                        )
                        if ema is not None:
                            ema.apply_shadow(model)
                            save_input_seeded_samples(
                                model,
                                diffusion,
                                sample_input_batch,
                                args.output_dir,
                                global_step,
                                tag="_ema",
                            )
                            ema.restore(model)
                    model.train()

            if args.max_steps > 0 and global_step >= args.max_steps:
                log.info(f"Reached max_steps={args.max_steps}; stopping training loop.")
                stop_training = True
                break

        # Epoch summary
        avg_epoch_loss = epoch_loss / max(1, epoch_batches)
        log.info(f"\n{'='*80}")
        log.info(f"Epoch {epoch+1} Summary:")
        log.info(f"  Average Loss: {avg_epoch_loss:.4f}")
        log.info(f"  Global Step: {global_step}")
        log.info(f"{'='*80}")

        # Log epoch summary to TensorBoard
        writer.add_scalar('Epoch/Average_Loss', avg_epoch_loss, epoch+1)
        writer.add_scalar('Epoch/Steps', global_step, epoch+1)

        if args.save_epoch_checkpoints:
            # Save epoch checkpoint
            checkpoint_path = os.path.join(
                args.output_dir, 'checkpoints',
                f'model_epoch_{epoch+1:03d}.pt'
            )
            torch.save(model.state_dict(), checkpoint_path)
            log.info(f"✓ Epoch checkpoint saved: {checkpoint_path}")

            # Save EMA epoch checkpoint
            if ema is not None:
                ema_epoch_path = os.path.join(
                    args.output_dir, 'checkpoints',
                    f'model_epoch_{epoch+1:03d}_ema.pt'
                )
                ema.apply_shadow(model)
                torch.save(model.state_dict(), ema_epoch_path)
                ema.restore(model)
                log.info(f"✓ EMA epoch checkpoint saved: {ema_epoch_path}")

        if stop_training:
            break

    log.info(f"\n{'='*80}")
    log.info("✅ TRAINING COMPLETE!")
    if args.save_final_checkpoint:
        final_path = os.path.join(args.output_dir, 'checkpoints', 'model_final.pt')
        torch.save(model.state_dict(), final_path)
        log.info(f"Final model saved: {final_path}")

        if ema is not None:
            ema_final_path = os.path.join(args.output_dir, 'checkpoints', 'model_final_ema.pt')
            ema.apply_shadow(model)
            torch.save(model.state_dict(), ema_final_path)
            ema.restore(model)
            log.info(f"Final EMA model saved: {ema_final_path}")
    else:
        log.info("Final checkpoint saving disabled by flag")

    # Close TensorBoard writer
    writer.close()
    if device.type == 'cuda':
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        log.info(f"Peak CUDA memory allocated: {peak_gib:.3f} GiB")
    log.info(f"TensorBoard logs saved to: {tensorboard_dir}")
    log.info(f"{'='*80}")


if __name__ == '__main__':
    main()
