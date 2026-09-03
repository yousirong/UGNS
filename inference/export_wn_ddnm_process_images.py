#!/usr/bin/env python3
"""
Export WN-DDNM process images for academic paper figure assembly.

This script saves intermediate images that match the WN-DDNM pipeline:
- Input/normalize/range-null decomposition
- Linear weight map construction steps
- Single diffusion step internals (x_t, eps_pred, x0_hat, null, range_null, x0_guided, x_{t-1})
- Reverse diffusion progression snapshots
- Ensemble samples and statistics
- Final output with comparison

Directory structure:
    output_dir/{image_name}/
        01_input_preprocessing/
        02_weight_map_construction/
        03_single_diffusion_step/
        04_diffusion_progression/
        05_ensemble_samples/
        06_ensemble_statistics/
        07_final_output/
        guidance_schedule.png
        process_overview.png
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths  # registers code subdirs on sys.path


import argparse
import importlib
import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from wn_ddnm_speckle_denoising_best_fusion import PRESET_FILES, load_preset_defaults


def load_denoiser_class(denoiser_file, denoiser_class):
    if denoiser_file:
        denoiser_path = Path(denoiser_file).expanduser().resolve()
        if not denoiser_path.is_file():
            raise FileNotFoundError(f"denoiser_file not found: {denoiser_path}")
        module_dir = str(denoiser_path.parent)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        module_name = denoiser_path.stem
        spec = importlib.util.spec_from_file_location(module_name, str(denoiser_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module("wn_ddnm_speckle_denoising_best_fusion")
    if not hasattr(module, denoiser_class):
        raise AttributeError(f"{denoiser_class} not found in {module.__name__}")
    return getattr(module, denoiser_class)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def restore_to_input_shape(denoiser, arr):
    """Restore model-sized arrays to original input shape when available."""
    restore_fn = getattr(denoiser, "restore_original_shape", None)
    if callable(restore_fn):
        return restore_fn(arr)
    return arr


def save_gray(img, path):
    """Save grayscale image to file."""
    img = np.clip(img, 0.0, 1.0)
    img_u8 = (img * 255).astype(np.uint8)
    Image.fromarray(img_u8).save(path)


def save_heatmap(img, path, cmap="viridis", vmin=0.0, vmax=1.0, add_colorbar=True, title=None):
    """Save heatmap with colorbar (colorbar enabled by default)."""
    fig, ax = plt.subplots(figsize=(3.5, 3), dpi=150)
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9)
    if add_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_range_null_profile(y_01, y_range, y_null, path, row=None):
    """Save 1D profile comparison of input, range, and null components."""
    if row is None:
        row = y_01.shape[0] // 2
    x = np.arange(y_01.shape[1])
    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=150)
    ax.plot(x, y_01[row], label="Input y", color="#3A6EA5", linewidth=1.2)
    ax.plot(x, y_range[row], label="Range $A^\\dagger A y$", color="#2E6F46", linewidth=1.2)
    ax.plot(x, y_null[row], label="Null $|y - A^\\dagger A y|$", color="#D87A3C", linewidth=1.2)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0, len(x) - 1)
    ax.set_xlabel("Pixel", fontsize=8)
    ax.set_ylabel("Intensity", fontsize=8)
    ax.legend(fontsize=7, frameon=True, fancybox=True, loc="upper right")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_guidance_schedule(denoiser, path):
    """Save guidance schedule lambda(t) visualization."""
    num_points = 100
    timesteps = np.linspace(0, denoiser.num_timesteps, num_points)
    lambdas = [denoiser.compute_guidance_strength(t) for t in timesteps]

    fig, ax = plt.subplots(figsize=(5, 3), dpi=150)
    ax.plot(timesteps / denoiser.num_timesteps, lambdas, color="#2E86AB", linewidth=2)
    ax.fill_between(timesteps / denoiser.num_timesteps, 0, lambdas, alpha=0.2, color="#2E86AB")
    ax.set_xlabel("Normalized timestep $t/T$", fontsize=9)
    ax.set_ylabel("Guidance strength $\\lambda(t)$", fontsize=9)
    ax.set_title(f"Guidance Schedule ({denoiser.guidance_schedule})", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(lambdas) * 1.1)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=8)

    # Add annotations
    ax.annotate(f"$\\lambda_{{start}}={denoiser.guidance_start:.2f}$",
                xy=(0, denoiser.guidance_start), xytext=(0.1, denoiser.guidance_start + 0.05),
                fontsize=8, color="#2E86AB")
    ax.annotate(f"$\\lambda_{{end}}={denoiser.guidance_end:.2f}$",
                xy=(1, denoiser.guidance_end), xytext=(0.8, denoiser.guidance_end + 0.05),
                fontsize=8, color="#2E86AB")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_comparison_figure(items, path, cols=4, title=None):
    """Save multi-panel comparison figure."""
    if not items:
        return
    rows = int(math.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5), dpi=150)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    axes = np.array(axes).reshape(rows, cols)

    for idx, item in enumerate(items):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        img = item["img"]
        cmap = item.get("cmap", "gray")
        vmin = item.get("vmin", 0.0)
        vmax = item.get("vmax", 1.0)
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(item["title"], fontsize=8)
        ax.axis("off")
        if item.get("colorbar", False):
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=6)

    for idx in range(len(items), rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    if title:
        fig.suptitle(title, fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def make_grid(images, cols, pad=4, bg=1.0):
    """Create a grid of images."""
    if not images:
        return None
    rows = int(math.ceil(len(images) / cols))
    h, w = images[0].shape
    canvas_h = rows * h + pad * (rows - 1)
    canvas_w = cols * w + pad * (cols - 1)
    canvas = np.ones((canvas_h, canvas_w), dtype=np.float32) * bg
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        y0 = r * (h + pad)
        x0 = c * (w + pad)
        canvas[y0:y0 + h, x0:x0 + w] = img
    return canvas


def list_image_files(input_dir):
    """List all image files in a directory."""
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(exts)]
    return sorted(files)


def _otsu_threshold(x, bins=256):
    x = np.clip(x, 0.0, 1.0)
    hist, bin_edges = np.histogram(x.ravel(), bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    mean1 = np.cumsum(hist * bin_mids) / np.maximum(weight1, 1e-12)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / np.maximum(weight2, 1e-12))[::-1]
    between = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    if between.size == 0:
        return 0.5
    idx = int(np.argmax(between))
    return float(bin_mids[idx])


def compute_otsu_debug(denoiser, bmode_01_np, linear_01=None):
    if linear_01 is None:
        linear_01 = denoiser.bmode_to_linear(bmode_01_np)

    range_input = linear_01
    if denoiser.use_srad:
        range_input = denoiser.lpf.srad_filter(range_input)

    if denoiser.use_wavelet:
        linear_smoothed = denoiser.lpf.range_space_projection(range_input)
    else:
        linear_smoothed = range_input

    mask_min, mask_max = linear_smoothed.min(), linear_smoothed.max()
    if mask_max > mask_min:
        linear_norm_raw = (linear_smoothed - mask_min) / (mask_max - mask_min)
    else:
        linear_norm_raw = np.ones_like(linear_smoothed)

    denom = max(1e-10, 1.0 - denoiser.linear_mask_center)
    linear_norm = np.clip((linear_norm_raw - denoiser.linear_mask_center) / denom, 0.0, 1.0)

    gamma = max(denoiser.linear_mask_steepness, 1e-3)
    intensity_mask_pre = linear_norm ** gamma

    debug = {
        "linear_smoothed": linear_smoothed,
        "linear_norm_raw": linear_norm_raw,
        "linear_norm": linear_norm,
        "intensity_mask_pre": intensity_mask_pre,
    }

    if denoiser.use_otsu_gate:
        otsu_t = _otsu_threshold(linear_norm)
        p70 = float(np.percentile(linear_norm, 70))
        p30 = float(np.percentile(linear_norm, 30))
        if otsu_t < p30:
            otsu_t = p30
        if otsu_t > p70:
            otsu_t = 0.5 * (p70 + p30)
        iqr = float(np.percentile(linear_norm, 75) - np.percentile(linear_norm, 25))
        tau = max(0.05, 0.5 * iqr)
        bg_gate = 1.0 / (1.0 + np.exp(-(linear_norm - otsu_t) / (tau + 1e-6)))
        intensity_mask_post = intensity_mask_pre * bg_gate
        debug.update({
            "otsu_t": otsu_t,
            "otsu_tau": tau,
            "bg_gate": bg_gate,
            "intensity_mask_post": intensity_mask_post,
        })

    return debug


def save_histogram_with_threshold(data, path, otsu_t=None, bins=256):
    fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=150)
    ax.hist(data.ravel(), bins=bins, range=(0.0, 1.0), color="#4C72B0", alpha=0.85)
    if otsu_t is not None:
        ax.axvline(otsu_t, color="#C44E52", linewidth=1.5, label=f"Otsu={otsu_t:.3f}")
        ax.legend(fontsize=7, frameon=True)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Value", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def compute_linear_weight_steps(denoiser, bmode_01_np, debug=False):
    """Compute intermediate steps for linear mask construction."""
    linear_01 = denoiser.bmode_to_linear(bmode_01_np)
    weight = denoiser.compute_linear_intensity_mask(bmode_01_np)

    steps = {
        "bmode_input": bmode_01_np,
        "linear_01": linear_01,
        "weight": weight,
    }

    if debug:
        steps.update(compute_otsu_debug(denoiser, bmode_01_np, linear_01))

    return steps



def sample_ddim_snapshots_with_timesteps(denoiser, y_noisy, num_snapshots=5, seed=42):
    """Sample DDIM snapshots; captured arrays are detached to CPU immediately."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    y_obs_range_batch, y_noisy_batch, h, w = denoiser._prepare_range_null(y_noisy, 1)
    x = torch.randn(1, 1, h, w, device=denoiser.device)

    timesteps = denoiser.ddim_timesteps
    if len(timesteps) < 2:
        return [], []

    snapshot_indices = np.linspace(0, len(timesteps) - 1, num_snapshots).astype(int)
    snapshot_set = set(snapshot_indices.tolist())

    snapshots = []
    snapshot_timesteps = []

    for i, t_curr in enumerate(timesteps):
        t_next = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        lambda_t = denoiser.compute_guidance_strength(t_curr)

        with torch.no_grad():
            t_tensor = torch.tensor([t_curr], device=denoiser.device, dtype=torch.long)
            model_output = denoiser.model(x, t_tensor)
            eps_pred = model_output[:, 0:1, :, :] if model_output.shape[1] == 2 else model_output

            x_0_pred = denoiser._predict_x0_from_noise(x, eps_pred, t_curr)
            x_0_bar = denoiser._apply_range_null_decomposition(x_0_pred, y_obs_range_batch, 1)
            x_0_hat = (1 - lambda_t * 0.3) * x_0_bar + lambda_t * 0.3 * y_noisy_batch

            if t_next >= 0:
                alpha_bar_curr = denoiser.alphas_cumprod[t_curr]
                alpha_bar_next = denoiser.alphas_cumprod[t_next]
                if denoiser.eta > 0:
                    sigma_t = denoiser.eta * torch.sqrt(
                        (1 - alpha_bar_next) / (1 - alpha_bar_curr)
                    ) * torch.sqrt(1 - alpha_bar_curr / alpha_bar_next)
                    noise_coeff = torch.sqrt(1 - alpha_bar_next - sigma_t ** 2)
                    noise = torch.randn_like(x)
                    x = torch.sqrt(alpha_bar_next) * x_0_hat + noise_coeff * eps_pred + sigma_t * noise
                else:
                    x = torch.sqrt(alpha_bar_next) * x_0_hat + torch.sqrt(1 - alpha_bar_next) * eps_pred
            else:
                x = x_0_hat

        if i in snapshot_set:
            x_01 = (x + 1) / 2
            x_01 = torch.clamp(x_01, 0, 1)
            snapshots.append(x_01.squeeze().detach().cpu().numpy())
            snapshot_timesteps.append((i, t_curr))

    return snapshots, snapshot_timesteps


def capture_single_step(denoiser, y_noisy, step_index, seed=42):
    """Capture all intermediate values for a single diffusion step."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    y_obs_range_batch, y_noisy_batch, h, w = denoiser._prepare_range_null(y_noisy, 1)
    x = torch.randn(1, 1, h, w, device=denoiser.device)

    timesteps = denoiser.ddim_timesteps
    step_index = max(0, min(step_index, len(timesteps) - 1))

    for i, t_curr in enumerate(timesteps):
        t_next = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        lambda_t = denoiser.compute_guidance_strength(t_curr)

        with torch.no_grad():
            t_tensor = torch.tensor([t_curr], device=denoiser.device, dtype=torch.long)
            model_output = denoiser.model(x, t_tensor)
            eps_pred_raw = model_output[:, 0:1, :, :] if model_output.shape[1] == 2 else model_output

            x_0_pred = denoiser._predict_x0_from_noise(x, eps_pred_raw, t_curr)

            x_0_pred_01 = (x_0_pred + 1) / 2
            x_0_pred_01 = torch.clamp(x_0_pred_01, 0, 1)
            x_0_pred_01_np = x_0_pred_01.squeeze(1).cpu().numpy()

            x_0_range = denoiser._range_projection_np(x_0_pred_01_np[0])
            x_0_null = np.stack([
                x_0_pred_01_np[0] - x_0_range
            ])
            x_0_null = np.nan_to_num(x_0_null, nan=0.0)
            x_0_null_tensor = torch.from_numpy(x_0_null).unsqueeze(1).float().to(denoiser.device)

            total_weight = denoiser.range_weight + denoiser.null_weight
            if total_weight > 0:
                range_null_combined = (denoiser.range_weight * y_obs_range_batch +
                                       denoiser.null_weight * x_0_null_tensor) / total_weight
            else:
                range_null_combined = y_obs_range_batch

            if denoiser.use_linear_range and hasattr(denoiser, "_intensity_weight"):
                weight_tensor = torch.from_numpy(denoiser._intensity_weight).unsqueeze(0).unsqueeze(0).float().to(denoiser.device)
                range_null = weight_tensor * range_null_combined + (1 - weight_tensor) * x_0_pred_01
                range_null = torch.clamp(range_null, 0, 1)
            else:
                range_null = torch.clamp(range_null_combined, 0, 1)

            x_0_bar = range_null * 2 - 1
            x_0_hat = (1 - lambda_t * 0.3) * x_0_bar + lambda_t * 0.3 * y_noisy_batch

            # x_0_guided is x_0_hat converted to [0,1]
            x_0_guided = (x_0_hat + 1) / 2
            x_0_guided = torch.clamp(x_0_guided, 0, 1)

            if t_next >= 0:
                alpha_bar_curr = denoiser.alphas_cumprod[t_curr]
                alpha_bar_next = denoiser.alphas_cumprod[t_next]
                if denoiser.eta > 0:
                    sigma_t = denoiser.eta * torch.sqrt(
                        (1 - alpha_bar_next) / (1 - alpha_bar_curr)
                    ) * torch.sqrt(1 - alpha_bar_curr / alpha_bar_next)
                    noise_coeff = torch.sqrt(1 - alpha_bar_next - sigma_t ** 2)
                    noise = torch.randn_like(x)
                    x_next = torch.sqrt(alpha_bar_next) * x_0_hat + noise_coeff * eps_pred_raw + sigma_t * noise
                else:
                    x_next = torch.sqrt(alpha_bar_next) * x_0_hat + torch.sqrt(1 - alpha_bar_next) * eps_pred_raw
            else:
                x_next = x_0_hat

        if i == step_index:
            # Normalize eps_pred to [0,1] for visualization
            eps_np = eps_pred_raw.squeeze().cpu().numpy()
            eps_min, eps_max = eps_np.min(), eps_np.max()
            if eps_max > eps_min:
                eps_normalized = (eps_np - eps_min) / (eps_max - eps_min)
            else:
                eps_normalized = np.zeros_like(eps_np)

            return {
                "t_index": i,
                "t_value": t_curr,
                "lambda_t": lambda_t,
                "x_t": torch.clamp((x + 1) / 2, 0, 1).squeeze().cpu().numpy(),
                "eps_pred": eps_normalized,
                "eps_pred_raw": eps_np,
                "x_0_pred": torch.clamp((x_0_pred + 1) / 2, 0, 1).squeeze().cpu().numpy(),
                "x_0_null": np.clip(x_0_null.squeeze(), 0, 1),
                "range_projection": torch.clamp(y_obs_range_batch, 0, 1).squeeze().cpu().numpy(),
                "range_null_combined": np.clip(range_null.squeeze().cpu().numpy(), 0, 1),
                "x_0_guided": x_0_guided.squeeze().cpu().numpy(),
                "x_t_minus_1": torch.clamp((x_next + 1) / 2, 0, 1).squeeze().cpu().numpy(),
            }

        x = x_next

    return None


def process_single_image(denoiser, input_path, output_dir, args):
    """Process a single image and save all intermediate results."""
    basename = os.path.splitext(os.path.basename(input_path))[0]
    output_root = os.path.join(output_dir, basename)
    ensure_dir(output_root)

    # Create organized directory structure
    dir_01_input = os.path.join(output_root, "01_input_preprocessing")
    dir_02_weight = os.path.join(output_root, "02_weight_map_construction")
    dir_03_step = os.path.join(output_root, "03_single_diffusion_step")
    dir_04_progress = os.path.join(output_root, "04_diffusion_progression")
    dir_05_samples = os.path.join(output_root, "05_ensemble_samples")
    dir_06_stats = os.path.join(output_root, "06_ensemble_statistics")
    dir_07_output = os.path.join(output_root, "07_final_output")

    for d in [dir_01_input, dir_02_weight, dir_03_step, dir_04_progress,
              dir_05_samples, dir_06_stats, dir_07_output]:
        ensure_dir(d)

    # =========================================
    # 01_input_preprocessing
    # =========================================
    print(f"  [1/7] Input preprocessing...")
    y_noisy, original = denoiser.preprocess_image(input_path)
    y_01_model = np.clip(original, 0.0, 1.0)
    y_01 = restore_to_input_shape(denoiser, y_01_model)

    save_gray(y_01, os.path.join(dir_01_input, "01_input_bmode.png"))
    save_gray(y_01, os.path.join(dir_01_input, "02_normalized.png"))

    # Range projection
    y_obs_range_model = denoiser._range_projection_np(y_01_model)
    y_obs_range = restore_to_input_shape(denoiser, y_obs_range_model)
    save_gray(y_obs_range, os.path.join(dir_01_input, "03_range_projection.png"))

    # Null (high-frequency) from observation
    y_obs_null = np.abs(y_01 - y_obs_range)
    y_obs_null = np.clip(y_obs_null, 0.0, 1.0)
    save_gray(y_obs_null, os.path.join(dir_01_input, "04_null_highfreq.png"))

    # Profile comparison
    save_range_null_profile(
        y_01, y_obs_range, y_obs_null,
        os.path.join(dir_01_input, "05_range_null_profile.png"),
    )

    # =========================================
    # 02_weight_map_construction (save only images used in linear-domain weight map)
    # =========================================
    print(f"  [2/7] Weight map construction...")
    linear_steps = compute_linear_weight_steps(denoiser, y_01_model, debug=True)
    for key, value in list(linear_steps.items()):
        if isinstance(value, np.ndarray):
            linear_steps[key] = restore_to_input_shape(denoiser, value)

    save_gray(linear_steps["linear_01"], os.path.join(dir_02_weight, "02_linear_converted.png"))

    if "linear_smoothed" in linear_steps:
        save_gray(linear_steps["linear_smoothed"], os.path.join(dir_02_weight, "02b_linear_smoothed.png"))
    if "linear_norm_raw" in linear_steps:
        save_gray(linear_steps["linear_norm_raw"], os.path.join(dir_02_weight, "02c_linear_norm_raw.png"))
    if "linear_norm" in linear_steps:
        save_gray(linear_steps["linear_norm"], os.path.join(dir_02_weight, "02d_linear_norm_centered.png"))
    if "intensity_mask_post" in linear_steps:
        save_gray(linear_steps["intensity_mask_post"], os.path.join(dir_02_weight, "02e_mask_post_otsu.png"))
    elif "intensity_mask_pre" in linear_steps:
        save_gray(linear_steps["intensity_mask_pre"], os.path.join(dir_02_weight, "02e_mask_pre_otsu.png"))

    save_gray(linear_steps["weight"], os.path.join(dir_02_weight, "03_weight_map.png"))

    # =========================================
    # 03_single_diffusion_step
    # =========================================
    print(f"  [3/7] Single diffusion step capture...")
    step_index = int(round(args.capture_step_ratio * (len(denoiser.ddim_timesteps) - 1)))
    step_data = capture_single_step(denoiser, y_noisy, step_index, seed=args.snapshot_seed)

    if step_data:
        for key, value in list(step_data.items()):
            if isinstance(value, np.ndarray):
                step_data[key] = restore_to_input_shape(denoiser, value)
        save_gray(step_data["x_t"], os.path.join(dir_03_step, "01_x_t.png"))
        save_gray(step_data["eps_pred"], os.path.join(dir_03_step, "02_eps_pred.png"))
        save_heatmap(
            step_data["eps_pred"],
            os.path.join(dir_03_step, "02_eps_pred_color.png"),
            cmap="coolwarm", vmin=0.0, vmax=1.0,
            title=f"Noise prediction $\\hat{{\\epsilon}}$ (t={step_data['t_value']})"
        )
        save_gray(step_data["x_0_pred"], os.path.join(dir_03_step, "03_x0_predicted.png"))
        save_gray(step_data["x_0_null"], os.path.join(dir_03_step, "04_null_from_x0.png"))
        save_gray(step_data["range_projection"], os.path.join(dir_03_step, "05_range_from_y.png"))
        save_gray(step_data["range_null_combined"], os.path.join(dir_03_step, "06_range_null_combined.png"))
        save_gray(step_data["x_0_guided"], os.path.join(dir_03_step, "07_x0_guided.png"))
        save_gray(step_data["x_t_minus_1"], os.path.join(dir_03_step, "08_x_t_minus_1.png"))

        if hasattr(denoiser, "_intensity_weight"):
            intensity_weight = restore_to_input_shape(denoiser, denoiser._intensity_weight)
            save_gray(intensity_weight, os.path.join(dir_03_step, "weight_map.png"))
            save_heatmap(
                intensity_weight,
                os.path.join(dir_03_step, "weight_map_color.png"),
                cmap="RdYlGn_r", vmin=0.0, vmax=1.0,
                title="Intensity Weight Map"
            )


    # =========================================
    # 04_diffusion_progression
    # =========================================
    print(f"  [4/7] Diffusion progression snapshots...")
    snapshots, snapshot_timesteps = sample_ddim_snapshots_with_timesteps(
        denoiser, y_noisy, args.snapshot_count, args.snapshot_seed
    )

    snapshots_vis = []
    if snapshots:
        for idx, (snap, (step_i, t_val)) in enumerate(zip(snapshots, snapshot_timesteps)):
            progress_pct = int(100 * step_i / max(1, len(denoiser.ddim_timesteps) - 1))
            filename = f"step_{progress_pct:03d}_t{t_val:03d}.png"
            snap_vis = restore_to_input_shape(denoiser, snap)
            snapshots_vis.append(snap_vis)
            save_gray(snap_vis, os.path.join(dir_04_progress, filename))

        # Create strip image
        strip = make_grid(snapshots_vis, cols=len(snapshots_vis), pad=4, bg=1.0)
        if strip is not None:
            save_gray(strip, os.path.join(dir_04_progress, "progression_strip.png"))

    # =========================================
    # 05_ensemble_samples
    # =========================================
    print(f"  [5/7] Ensemble generation...")
    samples = denoiser.generate_ensemble(y_noisy)
    samples_vis = []

    # Save individual samples
    for i, sample in enumerate(samples):
        sample_vis = restore_to_input_shape(denoiser, sample)
        samples_vis.append(sample_vis)
        save_gray(sample_vis, os.path.join(dir_05_samples, f"sample_{i+1:02d}.png"))

    # Create grid of samples
    grid_cols = min(5, len(samples_vis))
    samples_grid = make_grid(samples_vis, cols=grid_cols, pad=4, bg=1.0)
    if samples_grid is not None:
        save_gray(samples_grid, os.path.join(dir_05_samples, "samples_grid.png"))

    # =========================================
    # 06_ensemble_statistics
    # =========================================
    print(f"  [6/7] Computing ensemble statistics...")
    mean_img, variance, uncertainty = denoiser.compute_statistics(samples)
    fusion_info = None
    if denoiser.use_best_sample_fusion:
        final_img, fusion_info = denoiser.best_sample_aware_fusion(
            samples,
            y_01_model,
            mean_img,
            uncertainty,
            fusion_strategy=denoiser.fusion_strategy,
        )
        fusion_weight = fusion_info["weight_map"]
        texture_base = fusion_info["best_sample"]
        cv_map_from_model = None
    else:
        final_img, fusion_weight = denoiser.adaptive_fusion(samples, mean_img, uncertainty)
        texture_base = np.median(samples, axis=0)
        cv_map_from_model = None

    mean_img_vis = restore_to_input_shape(denoiser, mean_img)
    variance_vis = restore_to_input_shape(denoiser, variance)
    uncertainty_vis = restore_to_input_shape(denoiser, uncertainty)
    final_img_vis = restore_to_input_shape(denoiser, final_img)
    fusion_weight_vis = restore_to_input_shape(denoiser, fusion_weight)

    save_gray(mean_img_vis, os.path.join(dir_06_stats, "01_ensemble_mean.png"))

    # Variance
    var_normalized = variance / (variance.max() + 1e-10)
    var_normalized_vis = restore_to_input_shape(denoiser, var_normalized)
    save_gray(var_normalized_vis, os.path.join(dir_06_stats, "02_variance.png"))
    save_heatmap(
        var_normalized_vis,
        os.path.join(dir_06_stats, "02_variance_color.png"),
        cmap="viridis", vmin=0.0, vmax=1.0,
        title="Variance (normalized)"
    )

    # Uncertainty
    save_gray(uncertainty_vis, os.path.join(dir_06_stats, "03_uncertainty.png"))
    save_heatmap(
        uncertainty_vis,
        os.path.join(dir_06_stats, "03_uncertainty_color.png"),
        cmap="jet", vmin=0.0, vmax=1.0,
        title="Uncertainty"
    )

    # =========================================
    # Uncertainty-based Fusion Visualization
    # =========================================
    # Compute CV map (std / mean) - same as in adaptive_fusion
    std_map = np.sqrt(((samples - mean_img)**2).mean(axis=0) + 1e-10)
    if cv_map_from_model is None:
        cv_map = std_map / (np.abs(mean_img) + 1e-6)
    else:
        cv_map = cv_map_from_model
    cv_norm = (cv_map - cv_map.min()) / (cv_map.max() - cv_map.min() + 1e-10)
    cv_norm_vis = restore_to_input_shape(denoiser, cv_norm)

    # Dark factor (intensity-based)
    intensity_norm = (mean_img - mean_img.min()) / (mean_img.max() - mean_img.min() + 1e-10)
    dark_factor = 1 - intensity_norm
    dark_factor_vis = restore_to_input_shape(denoiser, dark_factor)

    # Save CV map
    save_gray(cv_norm_vis, os.path.join(dir_06_stats, "04_cv_map.png"))
    save_heatmap(
        cv_norm_vis,
        os.path.join(dir_06_stats, "04_cv_map_color.png"),
        cmap="plasma", vmin=0.0, vmax=1.0,
        title="CV Map (std/mean, normalized)"
    )

    # Save dark factor
    save_gray(dark_factor_vis, os.path.join(dir_06_stats, "05_dark_factor.png"))
    save_heatmap(
        dark_factor_vis,
        os.path.join(dir_06_stats, "05_dark_factor_color.png"),
        cmap="Blues", vmin=0.0, vmax=1.0,
        title="Dark Factor (1 - intensity)"
    )

    # Fusion weight (uncertainty-based)
    save_gray(fusion_weight_vis, os.path.join(dir_06_stats, "06_fusion_weight.png"))
    save_heatmap(
        fusion_weight_vis,
        os.path.join(dir_06_stats, "06_fusion_weight_color.png"),
        cmap="RdYlGn_r", vmin=0.7, vmax=1.0,
        title="Fusion Weight (Uncertainty)"
    )

    # Representative sample (or median when best-sample fusion is disabled)
    texture_base_vis = restore_to_input_shape(denoiser, texture_base)
    save_gray(texture_base_vis, os.path.join(dir_06_stats, "07_representative_sample.png"))
    std_map_vis = restore_to_input_shape(denoiser, std_map)

    # Fusion overview figure
    cv_fusion_items = [
        {"title": "1. Ensemble Mean", "img": mean_img_vis},
        {"title": "2. Std Map", "img": std_map_vis / (std_map_vis.max() + 1e-10)},
        {"title": "3. Uncertainty", "img": uncertainty_vis, "cmap": "plasma", "colorbar": True},
        {"title": "4. Dark Factor", "img": dark_factor_vis, "cmap": "Blues", "colorbar": True},
        {"title": "5. Fusion Weight", "img": fusion_weight_vis, "cmap": "RdYlGn_r", "vmin": 0.7, "vmax": 1.0, "colorbar": True},
        {"title": "6. Representative Sample", "img": texture_base_vis},
        {"title": "7. Final Output", "img": final_img_vis},
        {"title": "8. Input", "img": y_01},
    ]
    save_comparison_figure(
        cv_fusion_items,
        os.path.join(dir_06_stats, "fusion_overview.png"),
        cols=4,
        title=f"Uncertainty-Guided Fusion ({denoiser.fusion_strategy})"
    )

    # =========================================
    # 07_final_output
    # =========================================
    print(f"  [7/7] Saving final output...")
    save_gray(y_01, os.path.join(dir_07_output, "input.png"))
    save_gray(final_img_vis, os.path.join(dir_07_output, "output.png"))

    # Difference
    diff = np.abs(y_01 - final_img_vis)
    save_gray(diff, os.path.join(dir_07_output, "difference.png"))
    save_heatmap(
        diff,
        os.path.join(dir_07_output, "difference_color.png"),
        cmap="hot", vmin=0.0, vmax=0.3,
        title="Absolute Difference"
    )

    # Side-by-side comparison
    comparison_items = [
        {"title": "Input (Speckle)", "img": y_01},
        {"title": "Output (Denoised)", "img": final_img_vis},
        {"title": "Difference", "img": diff, "cmap": "hot", "vmin": 0.0, "vmax": 0.3, "colorbar": True},
    ]
    save_comparison_figure(
        comparison_items,
        os.path.join(dir_07_output, "comparison.png"),
        cols=3
    )

    # =========================================
    # Root level outputs
    # =========================================
    # Guidance schedule
    save_guidance_schedule(denoiser, os.path.join(output_root, "guidance_schedule.png"))

    # Process overview figure
    overview_range_img = linear_steps.get("linear_norm_raw", y_obs_range)
    overview_range_title = "2. Linear norm (raw)" if "linear_norm_raw" in linear_steps else "2. Range $A^\\dagger Ay$"
    overview_items = [
        {"title": "1. Input y", "img": y_01},
        {"title": overview_range_title, "img": overview_range_img},
        {"title": "3. Null (high-freq)", "img": y_obs_null},
        {"title": "4. Weight Map w(x)", "img": linear_steps["weight"]},
    ]

    if snapshots_vis:
        mid_idx = len(snapshots_vis) // 2
        overview_items.extend([
            {"title": "5. Diffusion (early)", "img": snapshots_vis[0]},
            {"title": "6. Diffusion (mid)", "img": snapshots_vis[mid_idx]},
            {"title": "7. Diffusion (late)", "img": snapshots_vis[-1]},
        ])

    overview_items.extend([
        {"title": "8. Uncertainty", "img": uncertainty_vis, "cmap": "plasma", "colorbar": True},
        {"title": "9. Dark Factor", "img": dark_factor_vis, "cmap": "Blues", "colorbar": True},
        {"title": "10. Fusion Weight", "img": fusion_weight_vis, "cmap": "RdYlGn_r", "vmin": 0.7, "vmax": 1.0, "colorbar": True},
        {"title": "11. Output", "img": final_img_vis},
    ])

    save_comparison_figure(
        overview_items,
        os.path.join(output_root, "process_overview.png"),
        cols=4,
        title="UGNS Pipeline with Uncertainty-Guided Fusion"
    )

    print(f"  Saved all process images to: {output_root}")


def main():
    preset_parser = argparse.ArgumentParser(add_help=False)
    preset_parser.add_argument("--preset", choices=sorted(PRESET_FILES), default="paper")
    selected_preset, _ = preset_parser.parse_known_args()

    parser = argparse.ArgumentParser(description="Export WN-DDNM process images for paper figures")
    parser.add_argument("--preset", choices=sorted(PRESET_FILES), default=selected_preset.preset,
                        help="Named YAML preset; explicit CLI options override it")
    parser.add_argument("--input", required=True, help="Input image or directory")
    parser.add_argument("--output_dir", default="./wn_ddnm_process_images", help="Output directory")
    parser.add_argument("--sample_index", type=int, default=0, help="Index if input is a directory")
    parser.add_argument("--process_all", action="store_true",
                        help="If input is a directory, process all images")

    parser.add_argument("--checkpoint", default=os.environ.get(
        "UGNS_CHECKPOINT", str(paths.CHECKPOINT_DIR / "ugns_prior.pt")), help="Model checkpoint")
    parser.add_argument("--config", default=os.environ.get(
        "UGNS_CONFIG", str(paths.CHECKPOINT_DIR / "config.json")), help="Model config")

    parser.add_argument("--denoiser_file", default="",
                        help="Optional path to an alternate denoiser module")
    parser.add_argument("--denoiser_class", default="WNDDNMDenoiser",
                        help="Denoiser class name inside the module")

    parser.add_argument("--ensemble_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Paper preset pins this to N; changing it changes ensemble samples")
    parser.add_argument("--ddim_steps", type=int, default=100)
    parser.add_argument("--guidance_start", type=float, default=0.3)
    parser.add_argument("--guidance_end", type=float, default=0.0)
    parser.add_argument("--guidance_power", type=float, default=2.0)
    parser.add_argument("--lpf_mode", type=str, default="wavelet")
    parser.add_argument("--lpf_kernel_size", type=int, default=3)
    parser.add_argument("--lpf_wavelet", type=str, default="sym4")
    parser.add_argument("--lpf_level", type=int, default=3)
    parser.add_argument("--range_weight", type=float, default=1.0)
    parser.add_argument("--null_weight", type=float, default=1.0)
    parser.add_argument("--timestep_schedule", type=str, default="karras")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--guidance_schedule", type=str, default="cosine")
    parser.add_argument("--use_linear_range", action="store_true", default=True,
                        help="Use linear domain weight map (enabled by default)")
    parser.add_argument("--db_range", type=float, default=60)
    parser.add_argument("--use_otsu_gate", action="store_true", default=True,
                        help="Enable Otsu-based soft background gate (default)")
    parser.add_argument("--no_otsu_gate", action="store_false", dest="use_otsu_gate",
                        help="Disable Otsu-based soft background gate")
    parser.add_argument("--linear_mask_center", type=float, default=0.4)
    parser.add_argument("--linear_mask_steepness", type=float, default=3.5)
    parser.add_argument("--max_intensity_cap", type=float, default=0.92)
    parser.add_argument("--otsu_strength", type=float, default=0.7)
    parser.add_argument("--use_wavelet", action="store_true", default=True)
    parser.add_argument("--no_wavelet", action="store_false", dest="use_wavelet")
    parser.add_argument("--use_srad", action="store_true", default=True)
    parser.add_argument("--srad_iterations", type=int, default=15)
    parser.add_argument("--srad_q0", type=float, default=0.4)
    parser.add_argument("--srad_delta_t", type=float, default=0.25)
    parser.add_argument("--use_best_sample_fusion", action="store_true", default=True,
                        help="Enable best-sample-aware fusion (default)")
    parser.add_argument("--no_best_sample_fusion", action="store_false", dest="use_best_sample_fusion",
                        help="Disable best-sample-aware fusion")
    parser.add_argument("--fusion_strategy", type=str, default="paper",
                        choices=["paper", "frequency", "uncertainty_blend", "topk"])
    parser.add_argument("--fusion_w_min", type=float, default=0.7)
    parser.add_argument("--fusion_w_max", type=float, default=0.98)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--fusion_temperature", type=float, default=0.3)
    parser.add_argument("--sampler", type=str, default="ddim")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--snapshot_count", type=int, default=10,
                        help="Number of snapshots; each is moved to CPU when captured")
    parser.add_argument("--snapshot_seed", type=int, default=42)
    parser.add_argument("--capture_step_ratio", type=float, default=0.5, help="Step ratio for single-step capture")

    preset_defaults = load_preset_defaults(selected_preset.preset)
    valid_destinations = {action.dest for action in parser._actions}
    unknown_keys = sorted(set(preset_defaults) - valid_destinations)
    if unknown_keys:
        parser.error(f"Unknown preset keys for exporter: {unknown_keys}")
    parser.set_defaults(**preset_defaults)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"WN-DDNM Process Image Export")
    print(f"{'='*60}\n")

    denoiser_cls = load_denoiser_class(args.denoiser_file, args.denoiser_class)
    denoiser = denoiser_cls(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        ensemble_size=args.ensemble_size,
        batch_size=args.batch_size,
        ddim_steps=args.ddim_steps,
        guidance_start=args.guidance_start,
        guidance_end=args.guidance_end,
        guidance_power=args.guidance_power,
        lpf_mode=args.lpf_mode,
        lpf_kernel_size=args.lpf_kernel_size,
        lpf_wavelet=args.lpf_wavelet,
        lpf_level=args.lpf_level,
        range_weight=args.range_weight,
        null_weight=args.null_weight,
        timestep_schedule=args.timestep_schedule,
        eta=args.eta,
        guidance_schedule=args.guidance_schedule,
        use_linear_range=args.use_linear_range,
        db_range=args.db_range,
        use_otsu_gate=args.use_otsu_gate,
        linear_mask_center=args.linear_mask_center,
        linear_mask_steepness=args.linear_mask_steepness,
        max_intensity_cap=args.max_intensity_cap,
        otsu_strength=args.otsu_strength,
        use_wavelet=args.use_wavelet,
        use_srad=args.use_srad,
        srad_iterations=args.srad_iterations,
        srad_q0=args.srad_q0,
        srad_delta_t=args.srad_delta_t,
        use_best_sample_fusion=args.use_best_sample_fusion,
        fusion_strategy=args.fusion_strategy,
        fusion_w_min=args.fusion_w_min,
        fusion_w_max=args.fusion_w_max,
        topk=args.topk,
        fusion_temperature=args.fusion_temperature,
        sampler=args.sampler,
    )

    input_path = args.input
    if os.path.isdir(input_path):
        files = list_image_files(input_path)
        if not files:
            raise RuntimeError(f"No image files found in {input_path}")
        if args.process_all:
            input_files = [os.path.join(input_path, f) for f in files]
            print(f"Processing {len(input_files)} images from: {input_path}")
        else:
            sample_index = max(0, min(args.sample_index, len(files) - 1))
            input_files = [os.path.join(input_path, files[sample_index])]
    else:
        input_files = [input_path]

    for idx, input_file in enumerate(input_files, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(input_files)}] Processing: {os.path.basename(input_file)}")
        print(f"{'='*60}")
        process_single_image(denoiser, input_file, args.output_dir, args)

    print(f"\n{'='*60}")
    print(f"Export complete! Output saved to: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
