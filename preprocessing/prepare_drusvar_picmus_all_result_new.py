#!/usr/bin/env python3
"""
Convert DRUSvar PICMUS DAS HDF5 files to PNGs (B-mode + linear envelope).

Input and output roots follow ``UGNS_DATA_ROOT`` and ``UGNS_RESULTS_ROOT``.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import paths  # registers code subdirs on sys.path


import argparse
from pathlib import Path

import h5py
import numpy as np
import scipy.io
from PIL import Image


PHANTOM_MAP = {
    1: "DAS_SR",         # simu_reso -> SR
    2: "DAS_SC",         # simu_cont -> SC
    3: "DAS_ER_iq",      # expe_reso
    4: "DAS_EC_iq",      # expe_cont
    5: "DAS_CC",         # expe_cross -> CC
    6: "DAS_CL",         # expe_long -> CL
}

PW_DIR_MAP = {
    1: "test_images_pw1_256x256",
    3: "test_images_pw3_256x256",
    11: "test_images_pw11_256x256",
    75: "gt_images_pw75_256x256",
}


def to_bmode_uint8(env, dynamic_range=60.0):
    eps = np.finfo(float).eps
    db = 20 * np.log10(env + eps)
    db = db - db.max()
    db = np.clip(db, -dynamic_range, 0)
    img = (db + dynamic_range) / dynamic_range
    return (img * 255).astype(np.uint8)


def to_linear_uint8(env):
    max_val = float(env.max())
    if max_val <= 0:
        max_val = 1.0
    img = np.clip(env / max_val, 0, 1)
    return (img * 255).astype(np.uint8)


def save_image(img_uint8, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_uint8, mode="L").save(path)


def orient_env(env):
    # Align orientation for downstream processing/visualization.
    # Flip vertically, then rotate 270° CCW (i.e., 90° CW).
    env = np.flipud(env)
    env = np.rot90(env, k=3)
    return env


def process_hdf5_file(hdf5_path, output_base, dynamic_range, save_bmode, save_linear):
    file_id = int(hdf5_path.stem)
    if file_id not in PHANTOM_MAP:
        print(f"Skipping unknown file id: {hdf5_path.name}")
        return

    base_name = PHANTOM_MAP[file_id]

    with h5py.File(hdf5_path, "r") as f:
        real = f["US/US_DATASET0000/data/real"][()]
        imag = f["US/US_DATASET0000/data/imag"][()]
        pw_list = f["US/US_DATASET0000/number_plane_waves"][()]

    data = real + 1j * imag

    for idx, pw in enumerate(pw_list):
        pw_int = int(pw)
        if pw_int not in PW_DIR_MAP:
            continue

        dir_name = PW_DIR_MAP[pw_int]
        pw_tag = f"pw{pw_int:02d}"
        filename = f"{base_name}_{pw_tag}.png"

        env = np.abs(data[idx])
        env = orient_env(env)

        if save_bmode:
            bmode_img = to_bmode_uint8(env, dynamic_range)
            out_dir = output_base / dir_name
            save_image(bmode_img, out_dir / filename)

        if save_linear:
            linear_img = to_linear_uint8(env)
            out_dir = output_base / f"{dir_name}_linear"
            save_image(linear_img, out_dir / filename)


def load_mat_image(mat_path):
    mat_data = scipy.io.loadmat(mat_path)
    if "x" not in mat_data:
        raise ValueError(f"Missing 'x' in {mat_path}")
    data = np.squeeze(mat_data["x"])
    if data.ndim != 2:
        raise ValueError(f"Expected 2D data in {mat_path}, got shape {data.shape}")
    return data


def save_stat_images(env, output_base, output_dir_name, filename, dynamic_range, save_bmode, save_linear):
    if save_bmode:
        bmode_img = to_bmode_uint8(env, dynamic_range)
        save_image(bmode_img, output_base / output_dir_name / filename)
    if save_linear:
        linear_img = to_linear_uint8(env)
        save_image(linear_img, output_base / f"{output_dir_name}_linear" / filename)


def process_drus_stats(stats_dir, output_base, dynamic_range, save_bmode, save_linear):
    phan_to_suffix = {
        3: "ER",
        4: "EC",
        5: "CC",
        6: "CL",
    }
    repeat = 10

    for phan_id, suffix in phan_to_suffix.items():
        is_vitro = phan_id < 5
        dataset_type = "vitro" if is_vitro else "vivo"

        deno_path = Path(stats_dir) / f"DENO{dataset_type}"
        drus_path = Path(stats_dir) / f"DRUS{dataset_type}"

        deno_temps = []
        drus_temps = []

        for c in range(repeat):
            if is_vitro:
                file_num = phan_id + c * 4
            else:
                file_num = (phan_id - 4) + c * 2

            deno_file = deno_path / f"{file_num}_-1.mat"
            drus_file = drus_path / f"{file_num}_-1.mat"

            if deno_file.exists():
                deno_temps.append(load_mat_image(deno_file))
            if drus_file.exists():
                drus_temps.append(load_mat_image(drus_file))

        if deno_temps:
            deno_mean = np.mean(deno_temps, axis=0)
            env = np.abs(deno_mean)
            save_stat_images(
                env,
                output_base,
                "DENOmean",
                f"DENOmean_{suffix}.png",
                dynamic_range,
                save_bmode,
                save_linear,
            )
            print(f"  ✓ DENOmean_{suffix}.png")
        else:
            print(f"  ⚠ Missing DENO data for phan {phan_id}")

        if drus_temps:
            drus_mean = np.mean(drus_temps, axis=0)
            drus_var = np.var(drus_temps, axis=0)

            env_mean = np.abs(drus_mean)
            env_var = np.abs(drus_var)

            save_stat_images(
                env_mean,
                output_base,
                "DRUSmean",
                f"DRUSmean_{suffix}.png",
                dynamic_range,
                save_bmode,
                save_linear,
            )
            save_stat_images(
                env_var,
                output_base,
                "DRUSvar",
                f"DRUSvar_{suffix}.png",
                dynamic_range,
                save_bmode,
                save_linear,
            )
            print(f"  ✓ DRUSmean_{suffix}.png")
            print(f"  ✓ DRUSvar_{suffix}.png")
        else:
            print(f"  ⚠ Missing DRUS data for phan {phan_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Export DRUSvar PICMUS DAS HDF5 to B-mode + linear PNGs"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(paths.DATA_DIR / "drusvar_picmus" / "DAS"),
        help="Directory with DAS HDF5 files (1.hdf5..6.hdf5)",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default=str(paths.RESULTS_DIR / "ALL_RESULT_NEW"),
        help="Output base directory",
    )
    parser.add_argument(
        "--dynamic-range",
        type=float,
        default=60.0,
        help="Dynamic range in dB for B-mode compression",
    )
    parser.add_argument(
        "--save-bmode",
        action="store_true",
        help="Save B-mode (log-compressed) images",
    )
    parser.add_argument(
        "--save-linear",
        action="store_true",
        help="Save linear envelope images",
    )
    parser.add_argument(
        "--stats-dir",
        type=str,
        default=str(paths.DATA_DIR / "drusvar_picmus" / "statistics"),
        help="Directory with DENO/DRUS .mat results",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip DENOmean/DRUSmean/DRUSvar export",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_base = Path(args.output_base)
    stats_dir = Path(args.stats_dir)

    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    save_bmode = args.save_bmode or not args.save_linear
    save_linear = args.save_linear or not args.save_bmode

    hdf5_files = sorted(input_dir.glob("*.hdf5"), key=lambda p: int(p.stem))
    if not hdf5_files:
        raise SystemExit(f"No .hdf5 files found in: {input_dir}")

    print("=" * 80)
    print("DRUSvar PICMUS DAS export")
    print(f"Input: {input_dir}")
    print(f"Output: {output_base}")
    print(f"B-mode: {save_bmode}, Linear: {save_linear}")
    print(f"Dynamic range: {args.dynamic_range} dB")
    print("=" * 80)

    for hdf5_path in hdf5_files:
        print(f"\nProcessing: {hdf5_path.name}")
        process_hdf5_file(
            hdf5_path,
            output_base,
            args.dynamic_range,
            save_bmode,
            save_linear,
        )

    if args.skip_stats:
        print("\nSkipping DENO/DRUS statistics export.")
    elif not stats_dir.exists():
        print(f"\n⚠ Stats directory not found: {stats_dir}")
    else:
        print("\nComputing DENOmean/DRUSmean/DRUSvar...")
        process_drus_stats(
            stats_dir,
            output_base,
            args.dynamic_range,
            save_bmode,
            save_linear,
        )

    print("\nDone.")
    print(f"Output base: {output_base}")


if __name__ == "__main__":
    main()
