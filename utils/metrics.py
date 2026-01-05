import torch
import math
from typing import Tuple

def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, border: int = 0, max_value: float = 1.0) -> float:
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two PyTorch tensors.

    Args:
        img1 (torch.Tensor): The reference image tensor (Ground Truth).
        img2 (torch.Tensor): The test image tensor (Reconstructed).
        border (int): Number of pixels to ignore on each border (crop). Defaults to 0.
        max_value (float): The maximum possible pixel value. Must be 1.0 for [0, 1] images.

    Returns:
        float: The PSNR value in decibels (dB).
    """
    # 1. Dimension Check and Preparation
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    # Move to CPU and cast to float for consistent calculation
    img1 = img1.cpu().float()
    img2 = img2.cpu().float()

    # 2. Boundary Handling (Cropping)
    if border > 0:
        # Tensors are expected to be in a format where the last two dimensions are (H, W),
        # e.g., [C, H, W] or [B, C, H, W].
        if img1.dim() < 2:
            raise ValueError("Tensor too small. Requires at least (H, W).")
            
        # Define slicing indices for Height and Width
        h_start = border
        h_end = img1.shape[-2] - border
        w_start = border
        w_end = img1.shape[-1] - border
        
        # Crop the last two dimensions (H and W)
        img1 = img1[..., h_start:h_end, w_start:w_end]
        img2 = img2[..., h_start:h_end, w_start:w_end]

    # 3. Mean Squared Error (MSE) Calculation
    # MSE is calculated across all elements of the cropped tensors.
    mse = torch.mean((img1 - img2)**2)

    # 4. PSNR Calculation
    if mse == 0:
        # Perfect image match
        return float('inf')
        
    # PSNR Formula: 10 * log10(MAX^2 / MSE)
    psnr = 10. * torch.log10(max_value**2 / mse)
    
    return psnr.item()