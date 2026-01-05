import os
import csv
import argparse
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image

from prior.red import REDPrior
from algorithm.admm_solver import ADMM_Solver
from utils.dataset import GetDataset
from utils.operators import InpaintingOperator
from utils.metrics import calculate_psnr


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_image(tensor: torch.Tensor, path: str) -> None:
    """
    Save a PyTorch tensor as an image file.
    
    Args:
        tensor: Tensor of shape [C, H, W] or [B, C, H, W] with values in [0, 1].
        path: Output file path (PNG or JPG).
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Ensure on CPU and detached
    tensor = tensor.cpu().detach()
    
    # Clamp to [0, 1]
    tensor = torch.clamp(tensor, 0.0, 1.0)
    
    # Convert to PIL Image
    to_pil = transforms.ToPILImage()
    img = to_pil(tensor)
    img.save(path)



def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="RED-ADMM reconstruction on image dataset")
    parser.add_argument("--data", type=str, default="data", help="Path to dataset (PNG images)")
    parser.add_argument("--model-path", type=str, default=os.path.join("prior", "drunet_color.pth"), help="Pretrained denoiser model path")
    parser.add_argument("--out-dir", type=str, default="results/" + datetime.now().strftime("%Y%m%d"), help="Output directory for results and logs")
    parser.add_argument("--img-size", type=int, default=256, help="Image size for resizing")
    parser.add_argument("--outer-iters", type=int, default=150, help="Number of ADMM outer iterations")
    parser.add_argument("--inner-denoiser-iters", type=int, default=1, help="Number of inner denoiser iterations")
    parser.add_argument("--alpha", type=float, default=2.0, help="ADMM relaxation parameter")
    parser.add_argument("--lambda-val", type=float, default=0.055, help="Regularization strength")
    parser.add_argument("--beta", type=float, default=0.0061, help="ADMM dual variable penalty")
    parser.add_argument("--final-denoising-strength", type=float, default=5.65, help="Final denoising strength for ADMM")
    parser.add_argument("--sigma", type=float, default=5/255, help="Noise level")
    parser.add_argument("--keep-ratio", type=float, default=0.3, help="Inpainting mask: fraction of pixels to keep")
    parser.add_argument("--num-images", type=int, default=None, help="Max number of images to process (None = all)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device (cuda:0 or cpu)")
    parser.add_argument("--no-csv", action="store_true", help="Skip saving CSV results")
    return parser.parse_args()


def red_admm_main(args) -> None:
    """
    Main RED-ADMM reconstruction pipeline.
    
    Args:
        args: Parsed arguments with dataset, model, and hyperparameter settings.
    """
    device = torch.device(args.device)
    ensure_dir(args.out_dir)
    
    print(f"Device: {device}")
    print(f"Output directory: {args.out_dir}")

    # Load dataset
    try:
        dataset = GetDataset(args.data, img_size=(args.img_size, args.img_size))
        print(f"Loaded {len(dataset)} images from {args.data}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))

    # Load prior
    try:
        red_prior = REDPrior(n_channels=3, model_path=args.model_path, device=device)
        print(f"Loaded denoiser from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # CSV logging
    csv_path = os.path.join(args.out_dir, "results.csv") if not args.no_csv else None
    if csv_path:
        ensure_dir(os.path.dirname(csv_path))
        csvfile = open(csv_path, "w", newline="")
        writer = csv.writer(csvfile)
        writer.writerow(["img_idx", "psnr_degraded", "psnr_reconstructed", "elapsed_time"])
        csvfile.flush()
    else:
        csvfile = None
        writer = None

    # Create subdirectories for images
    obs_dir = os.path.join(args.out_dir, "observations")     # degraded images
    rec_dir = os.path.join(args.out_dir, "reconstructions")  # restored images
    ref_dir = os.path.join(args.out_dir, "references")       # original clean images
    ensure_dir(obs_dir)
    ensure_dir(rec_dir)
    ensure_dir(ref_dir)

    # Process images
    num_processed = 0
    psnr_list = []
    
    with tqdm(total=min(len(dataset), args.num_images or len(dataset)), desc="Processing") as pbar:
        for batch_idx, batch in enumerate(dataloader):
            if args.num_images and batch_idx >= args.num_images:
                break

            clean = batch.to(torch.float32).to(device)
            
            # Physics operator
            physics = InpaintingOperator(
                image_shape=clean.shape[-2:],
                keep_ratio=args.keep_ratio,
                device=device
            )

            # ADMM solver
            red_admm = ADMM_Solver(
                operator=physics,
                prior=red_prior,
                device=device,
                outer_iters=args.outer_iters,
                inner_denoiser_iters=args.inner_denoiser_iters,
                alpha=args.alpha,
                lambda_val=args.lambda_val,
                beta=args.beta,
                final_denoising_strength=args.final_denoising_strength,
            )

            # Observation and reconstruction
            sigma_val = torch.tensor(args.sigma, device=device)
            y = physics.observe(clean, args.sigma)
            x0 = physics.initialize(y)

            with torch.no_grad():
                x_est, elapsed = red_admm(y, x0, sigma_val)

            # Metrics
            psnr_degraded = calculate_psnr(clean.cpu(), y.cpu())
            psnr_reconstructed = calculate_psnr(clean.cpu(), x_est.cpu())
            psnr_list.append(psnr_reconstructed)

            # Save images
            clean_save_path = os.path.join(ref_dir, f"ref_{batch_idx:04d}.png")
            obs_save_path   = os.path.join(obs_dir, f"obs_{batch_idx:04d}.png")
            rec_save_path   = os.path.join(rec_dir, f"rec_{batch_idx:04d}.png")
            
            save_image(clean, clean_save_path)
            save_image(y, obs_save_path)
            save_image(x_est, rec_save_path)

            # Log results
            if writer:
                writer.writerow([batch_idx, psnr_degraded, psnr_reconstructed, elapsed])
                csvfile.flush()

            num_processed += 1
            pbar.update(1)
            pbar.set_postfix({"PSNR (recon)": f"{psnr_reconstructed:.2f} dB"})

    if csvfile:
        csvfile.close()
        print(f"Results saved to {csv_path}")

    # Summary statistics
    if psnr_list:
        avg_psnr = sum(psnr_list) / len(psnr_list)
        max_psnr = max(psnr_list)
        min_psnr = min(psnr_list)
        print(f"\n=== Summary ===")
        print(f"Images processed: {num_processed}")
        print(f"Average PSNR: {avg_psnr:.4f} dB")
        print(f"Max PSNR: {max_psnr:.4f} dB")
        print(f"Min PSNR: {min_psnr:.4f} dB")
        print(f"\nImages saved to:")
        print(f"  - References: {ref_dir}")
        print(f"  - Observations (degraded): {obs_dir}")
        print(f"  - Reconstructions (restored): {rec_dir}")


if __name__ == "__main__":
    args = parse_args()
    red_admm_main(args)
