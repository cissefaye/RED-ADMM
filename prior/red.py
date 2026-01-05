import torch
import torch.nn as nn
from .network_unet import UNetRes as net
from .test_utils import test_mode


class REDPrior(nn.Module):
    r"""
    Regularization by Denoising (RED) prior using DRUNet as the denoiser.
    """

    def __init__(self, n_channels: int, model_path: str, device: str):
        """
        Initialize the RED prior with a DRUNet denoiser.

        Parameters
        ----------
        n_channels : int
            Number of input channels.
        model_path : str
            Path to the pre-trained DRUNet model.
        device : str
            Execution device ('cpu' or 'cuda').
        """
        super().__init__()
        self.model = net(
            in_nc=n_channels + 1,
            out_nc=n_channels,
            nc=[64, 128, 256, 512],
            nb=4,
            act_mode='R',
            downsample_mode="strideconv",
            upsample_mode="convtranspose"
        )
        self.model.load_state_dict(torch.load(model_path), strict=True)
        self.model.eval()
        for _, v in self.model.named_parameters():
            v.requires_grad = False
        self.model = self.model.to(device)
        self.device = device

        print(f"Model path: {model_path}")

    @property
    def display_name(self) -> str:
        """Return the display name of the prior."""
        return "RED"

    @torch.inference_mode()
    def denoise(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Apply DRUNet denoising to the input image.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape [B, C, H, W].
        sigma : float
            Denoising strength.

        Returns
        -------
        torch.Tensor
            Denoised image.
        """
        x_min, x_max = torch.min(x), torch.max(x)
        x = (x - x_min) / (x_max - x_min)
        x = x.to(x.device, dtype=torch.float32)
        sigma = sigma.to(self.device)
        x_input = torch.cat((x, sigma.float().repeat(1, 1, x.shape[2], x.shape[3])), dim=1)
        x_denoised = test_mode(self.model, x_input, mode=2, refield=32, min_size=256, modulo=16)
        return (x_max - x_min) * x_denoised + x_min
