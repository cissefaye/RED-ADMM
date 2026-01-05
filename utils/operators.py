import torch
import torch.nn as nn

class InpaintingOperator(nn.Module):
    """
    Forward operator for inpainting problems.
    This operator applies a binary mask to the input image, simulating missing pixels.
    """

    def __init__(
        self, 
        image_shape,
        keep_ratio, 
        dtype=torch.float32, 
        device='cpu',
        seed: int = 24
    ):
        """
        Initialize a random binary mask for inpainting.

        Args:
            image_shape (tuple): (H, W) dimensions of the image.
            keep_ratio (float): Probability of keeping each pixel (between 0 and 1).
            dtype (torch.dtype): Data type of the mask (e.g., torch.float32).
            device (str): PyTorch device ('cpu' or 'cuda').
        """
        
        self.H, self.W = image_shape
        self.dtype = dtype
        self.seed  = seed
        self.device = device

        # Generate a random binary mask with values in {0, 1}
        torch.manual_seed(seed)
        mask = (torch.rand((1, 1, self.H, self.W), dtype=dtype, device=device) < keep_ratio).to(dtype)
        self.mask = mask.to(device)
        
    def forward(self, x):
        """
        Apply the binary mask to simulate missing data.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Masked image (same shape as x)
        """
        return self.mask * x

    def adjoint(self, y):
        """
        The adjoint of the inpainting operator is itself.

        Args:
            y (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Masked tensor
        """
        return self.mask * y
    
    def observe(self, x, sigma):
        """
        Generate a noisy observation y = H(x) + w
        """
        torch.manual_seed(self.seed)
        x = x.to(self.device)
        Hx = self.forward(x)
        noise = sigma * torch.randn_like(Hx)
        mask = self.mask.to(self.device)
        return mask * (Hx + noise)

    def initialize(self, y: torch.Tensor) -> torch.Tensor:
        """
        Initialize the variable x given the observation y.
        """
        return torch.randn_like(y, device=self.device, dtype=self.dtype)
    