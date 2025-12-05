# Regularization by Denoising (RED) 

Overview
- This repository contains an implementation of Regularization by Denoising (RED) for solving inverse problems (e.g. image inpainting) using an ADMM solver and a denoising prior (DRUNet / UNet-based denoiser).

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

Quick start
- Open `Untitled.ipynb` (in VS Code or Jupyter) and run the cells. The notebook loads the prior once and processes images found under `data/*.png`.

Notes & best practices
- Reproducibility: set a global seed for `random`, `numpy`, and `torch` if you need reproducible experiments.
- Inference performance: if using a GPU, using `torch.cuda.amp.autocast()` for inference can accelerate runs and reduce memory usage.
- Memory: the denoiser should be used in evaluation mode (`model.eval()`), and inference should be wrapped in `torch.no_grad()` to avoid building computation graphs.
- Configuration: consider moving hyperparameters (e.g. `alpha`, `lambda_val`, `beta`, iteration counts) to a `config.yaml` or use `argparse` for easier experimentation.

Saving results
- Create a `results/` folder to store reconstructions, metric logs (CSV), and any checkpoints.

Need help?
- If you want, I can:
	- Apply safe notebook updates (add seeding, `model.eval()`, `torch.no_grad()`, PSNR logging),
	- Create a `scripts/run.py` entrypoint with `argparse` and a `config.yaml`, or
	- Add CI and basic unit tests.

---


