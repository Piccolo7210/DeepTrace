"""
Kaggle script (single cell): computes AEROBLADE Delta_Min (LPIPS2, min over the SD1 /
SD2 / Kandinsky autoencoders) for EVERY image in the dataset — both real and
generated — using both T4 GPUs concurrently. Writes results/real_scores.csv and
results/gen_scores.csv.

Design notes:

* The parent writes its worker out to a real .py file on disk and launches it with
  subprocess, one process per GPU. This is deliberate: multiprocessing's "spawn" start
  method (required for CUDA) cannot pickle a function defined in a notebook cell, so a
  multiprocessing-based version fails with `AttributeError: Can't get attribute
  '_worker' on <module '__main__'>` when this is pasted into a cell. Separate `python`
  invocations communicate only through files, so pasting works fine.

* GPU 0 gets ALL of real/ plus the whole-directory generated/<model>/ sets listed in
  GPU0_GENERATED_MODELS below. GPU 1 gets every other generated/<model>/ directory,
  in full. No model directory is ever split across GPUs, and real/ is never split
  either — each is scored as one contiguous unit on one GPU.

* LPIPS layer: the paper's LPIPS2 is index 1 in the lpips package's 0-based
  retPerLayer list (index 0 = relu1_2 = paper's LPIPS1). Using index 2 computes the
  paper's LPIPS3 instead, which measurably degrades real/generated separability.

Usage: paste this whole file into one Kaggle cell, set DATA_ROOT below to your
dataset's mount path, and run. Requires: !pip install -q diffusers==0.39.0 transformers
accelerate lpips
"""

import csv
import subprocess
import sys
from pathlib import Path

# ---- EDIT THIS to match the Kaggle "Data" pane path for your uploaded dataset ----
DATA_ROOT = "/kaggle/input/datasets/mustakimbinmohsin/image-dataset/data"
# -----------------------------------------------------------------------------------

# ---- EDIT THIS if generated/ gains or loses model folders. Whatever's listed here
# runs on GPU 0 (alongside all of real/); every other generated/<model>/ dir runs on
# GPU 1, in full. Only consulted when 2 GPUs are visible.
GPU0_GENERATED_MODELS = {
    "CompVis-stable-diffusion-v1-1-ViT-L-14-openai",
    "kandinsky-community-kandinsky-2-1-ViT-L-14-openai",
    "midjourney-v4",
    "midjourney-v5",
}
# -----------------------------------------------------------------------------------

RESULTS_DIR = Path("/kaggle/working/results")

WORKER_SOURCE = r'''
import argparse
import csv
import time
from pathlib import Path

IMAGE_SIZE = 512
LPIPS_LAYER_INDEX = 1   # 0-based index into lpips retPerLayer == the paper's LPIPS2
LPIPS_NET = "vgg"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

AE_CONFIGS = {
    "sd1": {"repo_id": "stable-diffusion-v1-5/stable-diffusion-v1-5", "subfolder": "vae", "type": "kl"},
    "sd2": {"repo_id": "sd2-community/stable-diffusion-2-1", "subfolder": "vae", "type": "kl"},
    "kandinsky": {"repo_id": "kandinsky-community/kandinsky-2-1", "subfolder": "movq", "type": "vq"},
}


def _iter_images(dir_path):
    return sorted(p for p in Path(dir_path).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def build_work_list(data_root, models, include_real):
    """This worker's slice: all of real/ (if include_real) plus each named
    generated/<model>/ directory in full. The caller (parent process) has already
    decided the full partition, so no further splitting happens here."""
    data_root = Path(data_root)
    items = []
    if include_real:
        items += [("real", "real", p) for p in _iter_images(data_root / "real")]
    generated_root = data_root / "generated"
    for name in models:
        items += [("generated", name, p) for p in _iter_images(generated_root / name)]
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--models", default="", help="'|'-separated generated/<model> dir names for this GPU")
    parser.add_argument("--include-real", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = args.device
    models = [m for m in args.models.split("|") if m]

    import torch
    from diffusers import AutoencoderKL, VQModel
    from PIL import Image
    from torchvision import transforms
    import lpips

    preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    def load_image_tensor(path):
        img = Image.open(path).convert("RGB")
        return (preprocess(img) * 2 - 1).unsqueeze(0).to(device)

    def reconstruct(vae, image_tensor, ae_type):
        with torch.no_grad():
            if ae_type == "kl":
                latents = vae.encode(image_tensor).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                return vae.decode(latents / vae.config.scaling_factor).sample
            elif ae_type == "vq":
                return vae.decode(vae.encode(image_tensor).latents).sample
            raise ValueError(f"Unknown ae_type: {ae_type}")

    print(f"[{device}] loading autoencoders...", flush=True)
    aes = {}
    for name, cfg in AE_CONFIGS.items():
        model_cls = AutoencoderKL if cfg["type"] == "kl" else VQModel
        model = model_cls.from_pretrained(
            cfg["repo_id"], subfolder=cfg["subfolder"], torch_dtype=torch.float32
        )
        model.to(device)
        model.eval()
        aes[name] = model

    lpips_model = lpips.LPIPS(net=LPIPS_NET, spatial=False).to(device)
    lpips_model.eval()
    print(f"[{device}] models ready", flush=True)

    items = build_work_list(args.data_root, models, args.include_real)
    n_real = sum(1 for s, _, _ in items if s == "real")
    print(
        f"[{device}] {len(items)} images ({n_real} real, {len(items) - n_real} generated) "
        f"from: {'real + ' if args.include_real else ''}{', '.join(models)}",
        flush=True,
    )

    start = time.time()
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subset", "model", "file", "delta_min"])
        for i, (subset, model_name, path) in enumerate(items, 1):
            x = load_image_tensor(str(path))
            distances = []
            for name, vae in aes.items():
                x_tilde = reconstruct(vae, x, ae_type=AE_CONFIGS[name]["type"])
                with torch.no_grad():
                    _, layer_dists = lpips_model(x, x_tilde, retPerLayer=True, normalize=False)
                distances.append(layer_dists[LPIPS_LAYER_INDEX].item())
            score = min(distances)
            writer.writerow([subset, model_name, path.name, score])

            if i % 200 == 0 or i == len(items):
                f.flush()
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (len(items) - i) / rate if rate > 0 else float("inf")
                print(
                    f"[{device}] {i}/{len(items)}  {path.name}: {score:.4f}  "
                    f"elapsed={elapsed/60:.1f}min  rate={rate:.2f}img/s  eta={eta/60:.1f}min",
                    flush=True,
                )

    print(f"[{device}] done in {(time.time()-start)/60:.1f}min -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
'''


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — check the notebook's Accelerator setting.")
    n_gpus = torch.cuda.device_count()
    print(f"Visible GPUs: {n_gpus}")
    num_parts = 2 if n_gpus >= 2 else 1
    if num_parts == 1:
        print("WARNING: only 1 GPU visible — set Accelerator to 'GPU T4 x2' to use both.")

    all_models = sorted(
        p.name for p in (Path(DATA_ROOT) / "generated").iterdir() if p.is_dir()
    )
    if num_parts == 1:
        # Single GPU: everything runs on it.
        part_models = [all_models]
        part_include_real = [True]
    else:
        gpu0_models = [m for m in all_models if m in GPU0_GENERATED_MODELS]
        gpu1_models = [m for m in all_models if m not in GPU0_GENERATED_MODELS]
        missing = GPU0_GENERATED_MODELS - set(all_models)
        if missing:
            print(f"WARNING: GPU0_GENERATED_MODELS names not found under generated/: {sorted(missing)}")
        part_models = [gpu0_models, gpu1_models]
        part_include_real = [True, False]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    worker_path = Path("_aeroblade_worker.py")
    worker_path.write_text(WORKER_SOURCE)

    part_paths = [RESULTS_DIR / f"scores_gpu{part}.csv" for part in range(num_parts)]

    # Launch every worker before waiting on any of them, so the GPUs run concurrently.
    procs = []
    for part in range(num_parts):
        cmd = [
            sys.executable, "-u", str(worker_path),
            "--device", f"cuda:{part}",
            "--data-root", DATA_ROOT,
            "--models", "|".join(part_models[part]),
            "--out", str(part_paths[part]),
        ]
        if part_include_real[part]:
            cmd.append("--include-real")
        print("Launching:", " ".join(cmd))
        procs.append(subprocess.Popen(cmd))

    for part, p in enumerate(procs):
        if p.wait() != 0:
            raise RuntimeError(f"Worker for part {part} failed with exit code {p.returncode}")

    # Split the per-GPU CSVs back out into the two files the local scripts expect.
    real_rows, gen_rows = [], []
    for part_path in part_paths:
        with open(part_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["subset"] == "real":
                    real_rows.append([row["file"], row["delta_min"]])
                else:
                    gen_rows.append([row["model"], row["file"], row["delta_min"]])

    with open(RESULTS_DIR / "real_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "delta_min"])
        writer.writerows(real_rows)

    with open(RESULTS_DIR / "gen_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "file", "delta_min"])
        writer.writerows(gen_rows)

    print(f"\nWrote {len(real_rows)} real scores -> {RESULTS_DIR / 'real_scores.csv'}")
    print(f"Wrote {len(gen_rows)} generated scores -> {RESULTS_DIR / 'gen_scores.csv'}")
    print("\nDownload both CSVs and place them in the project root, then run:")
    print("  python src/evaluate.py")
    print("  python src/calibrate_target_fpr.py --target-fpr 0.05")


if __name__ == "__main__":
    main()
