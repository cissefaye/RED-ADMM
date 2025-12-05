# Regularization by Denoising (RED) 

This repository provides simple Pytorch implementation of Regularization by Denoising (RED) for solving inverse problems (e.g. image inpainting) using an ADMM solver.

Repository structure
- `algorithm/`: ADMM solver implementation (`admm_solver.py`).
- `prior/`: prior model and network definitions; pretrained weights expected at `prior/drunet_color.pth`.
- `utils/`: helper modules (dataset loader, operators, metrics, plotting).
- `requirements.txt`: Python dependencies.

Prerequisites
- Python 3.8 or newer is recommended.
- CUDA-compatible GPU is recommended for faster inference, but the code can run on CPU.

Installation 
```powershell
# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Model weights
- Place the denoiser weights at `prior/` before running the scripts.

Saving results
- Create a `results/` folder to store reconstructions, metric logs (CSV), and any checkpoints.


🖼 Example Images
🔍 Input (Degraded / Observed)
<img src="images/obs_0000.png" />
✨ Output (Reconstruction via RED-ADMM)
<img src="images/rec_0000.png"/>
🧾 Original Reference Image
<img src="images/test_image.png"/>

📚 Reference

Romano, Y., Elad, M., & Milanfar, P. (2017). 
"The little engine that could: Regularization by denoising (RED)."
SIAM journal on imaging sciences, 10(4), 1804-1844.





