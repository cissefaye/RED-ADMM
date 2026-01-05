import torch
import torch.nn as nn
import numpy as np
import time
from typing import Callable, Tuple, Optional


# --------------------------------
# get rho and sigma
# --------------------------------
def get_rho_sigma(sigma=2.55/255, iter_num=15, modelSigma1=49.0, modelSigma2=2.55, w=1.0):
    '''
    One can change the sigma to implicitly change the trade-off parameter
    between fidelity term and prior term
    '''
    modelSigmaS = np.logspace(np.log10(modelSigma1), np.log10(modelSigma2), iter_num).astype(np.float32)
    modelSigmaS_lin = np.linspace(modelSigma1, modelSigma2, iter_num).astype(np.float32)
    sigmas = (modelSigmaS*w+modelSigmaS_lin*(1-w))/255.
    rhos = list(map(lambda x: 0.23*(sigma**2)/(x**2), sigmas))
    return rhos, sigmas


class ADMM_Solver(nn.Module):
    """
    Implements an iterative solver for inverse problems using the Alternating Direction 
    Method of Multipliers (ADMM) framework, incorporating a denoiser (Prior) 
    in the regularization step.
    
    Objective:
        Minimize E(x) = 1/(2sigma^2)||Hx-y||_2^2 + 0.5*lambda*x^T*(x-denoise(x))
        via the ADMM method.
    """

    def __init__(
        self, 
        operator: Callable,
        prior: nn.Module,  
        device: torch.device,
        outer_iters: int,
        inner_denoiser_iters: int = 1,
        alpha: float = 1.0,
        beta: float = 1.0,
        lambda_val: float = 1.0,
        final_denoising_strength: float = 2.65,
    ):
        """
        Initializes the RED-ADMM solver.
        """
        super().__init__()
        self.operator = operator
        self.prior = prior
        self.device = device
        self.outer_iters = outer_iters
        self.inner_denoiser_iters = inner_denoiser_iters
        self.alpha = alpha
        self.beta = beta
        self.lambda_val = lambda_val
        self.final_denoising_strength = final_denoising_strength
        
        # Ensure the prior (denoiser) is in evaluation mode
        self.prior.eval()
        
    def cg_x_update(self, y, v, u, sigma, x0, max_iter=50, tol=1e-6):
        """
        Solves:
             x = argmin_z { 1/(2sigma^2) * ||A z - y||^2 + 0.5 * beta ||z - (v - u)||_2^2 }
        """

        # Define the linear operator M(z)
        def M(z):
            return (1/sigma**2) * self.operator.adjoint(self.operator.forward(z)) + self.beta * z

        # Build the right-hand side b
        b = (1/sigma**2) * self.operator.adjoint(y) + self.beta * (v - u)

        # === Conjugate Gradient ===
        x = x0
        r = b - M(x)
        d = r.clone()

        for _ in range(max_iter):
            Md = M(d)
            rr = torch.sum(r**2)
            alpha = rr / torch.sum(d * Md)

            x = x + alpha * d
            r = r - alpha * Md

            # Stopping criterion
            if torch.norm(r) / torch.norm(b) < tol:
                break

            beta_cg = torch.sum(r**2) / rr
            d = r + beta_cg * d

        return x


    def forward(
        self, 
        y: torch.Tensor, 
        x0: torch.Tensor, 
        sigma: float,
    ) -> Tuple[torch.Tensor, float]:
        """
        Executes the ADMM iterative reconstruction process.

        :param y: The measurement vector (degraded image).
        :param X0: The initial estimate of the clean image.
        :return: A tuple containing the final estimated image and the elapsed time.
        """
        
        start_time = time.time()
        
        # Initialize primal and dual variables
        x_est = x0.clone().to(self.device)
        v_est = x0.clone().to(self.device)
        u_est = torch.zeros_like(x_est).to(self.device)

        if isinstance(sigma, torch.Tensor):
            sigma = sigma.item() if sigma.numel() == 1 else float(sigma.cpu().numpy())
        else:
            sigma = float(sigma)
        modelSigma2 = self.final_denoising_strength
        modelSigma1 = 49.0
        _, sigmas = get_rho_sigma(sigma=max(0.255/255., sigma), iter_num=self.outer_iters, modelSigma1=modelSigma1, modelSigma2=modelSigma2, w=1.0)
        self.sigmas = torch.tensor(sigmas).to(self.device)
        
        for k in range(self.outer_iters):
            sigma_denoiser = self.sigmas[k]
            
            # --- Part 1: X-Subproblem (Data Fidelity Term) ---
            # Solves: x = argmin_z { 1/(2sigma^2) * ||A z - y||^2 + 0.5 * beta ||z - (v - u)||_2^2 }
            b = self.operator.adjoint(y)/sigma**2 + self.beta*(v_est - u_est)
            denom = self.operator.mask/sigma**2 + self.beta
            x_est = b / denom
            #x_est =  self.cg_x_update(y, v_est, u_est, sigma, x_est, max_iter=150)
            
            # --- Relaxation Step ---
            x_hat = self.alpha * x_est + (1.0 - self.alpha) * v_est
            
            # --- Part 2: V-Subproblem (Regularization Term) ---
            # Approximates the solution of:
            # v = argmin_z { lambda * g(z) + 0.5 * beta ||z - (x_hat + u)||_2^2 }
            
            for j in range(self.inner_denoiser_iters):
                with torch.no_grad():
                    f_v_est = self.prior.denoise(v_est, sigma=sigma_denoiser)
                v_est = (self.beta*(x_hat + u_est) + self.lambda_val*f_v_est) / (self.lambda_val + self.beta)
            
            # --- Part 3: U-Subproblem (Dual Variable Update) ---
            u_est = u_est + x_hat - v_est
            
        end_time = time.time()
        elapsed = end_time - start_time
        
        return x_est, elapsed
