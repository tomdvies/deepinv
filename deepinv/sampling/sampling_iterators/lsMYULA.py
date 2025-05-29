import torch.nn as nn
from deepinv.sampling.utils import projbox
import torch
from torch import Tensor
import time as time
import numpy as np
from deepinv.physics import LinearPhysics
from deepinv.optim import PnP
from deepinv.physics import Physics
from deepinv.optim.prior import Prior, ScorePrior
import deepinv as dinv
from deepinv.sampling.sampling_iterators.sampling_iterator import SamplingIterator
from deepinv.optim.data_fidelity import DataFidelity

from typing import Dict, Optional, Tuple, Any


class lsMYULAIterator(SamplingIterator):
    r"""
    Latent Space Moreau-Yosida Unadjusted Langevin Algorithm (ls-MYULA).
    The state dictionary `X_in` should contain `{'z': Z_curr}`.
    The output dictionary will be `{'x': X_grad_next, 'z': Z_next}`.
    """

    def __init__(self, algo_params: Dict[str, float], clip_values: Optional[Tuple[float, float]] = None):
        super().__init__(algo_params)

        # Check for required algorithm parameters
        missing_params = []
        if "step_size" not in self.algo_params:
            missing_params.append("step_size")
        if "lambda" not in self.algo_params: # Parameter for Moreau-Yosida of the prior
            missing_params.append("lambda")
        if "rho_sq" not in self.algo_params: # Augmentation parameter rho^2
            missing_params.append("rho_sq")
        # Optional: "b" for bias in the forward model y = Ax + b + noise

        if missing_params:
            raise ValueError(
                f"Missing required parameters for LsMyulaIterator: {', '.join(missing_params)}"
            )

        self.clip_values = clip_values

    def initialize_latent_variables(
        self,
        x_init: Tensor,
        y: Tensor,
        physics: Physics,
        cur_data_fidelity: DataFidelity,
        cur_prior: Prior,
    ) -> Dict[str, Tensor]:
        r"""
        Initializes latent variables for the ls-MYULA iterator.
        The primary variable is Z. X_grad is derived.
        The 'x' in the returned dictionary will be the initial X_grad.
        """
        Z_init = x_init.clone()

        # X_grad_init = E[x | y, Z_init, rho_sq]
        # This is prox_{rho_sq * f_y}(Z_init)
        X_grad_init = cur_data_fidelity.prox(
            Z_init, 
            y,
            physics, 
            gamma=self.algo_params["rho_sq"]
        )

        if self.clip_values:
            X_grad_init = torch.clamp(X_grad_init, self.clip_values[0], self.clip_values[1])
            Z_init = torch.clamp(Z_init, self.clip_values[0], self.clip_values[1])


        return {"x": X_grad_init, "z": Z_init}

    def forward(
        self,
        X_in: Dict[str, Tensor],
        y: Tensor,
        physics: LinearPhysics,
        cur_data_fidelity: DataFidelity,
        cur_prior: Prior,
        *args,
        **kwargs,
    ) -> Dict[str, Tensor]:
        
        Z_curr = X_in["z"]

        # prox gives mean as it is gaussian (there is closed form)
        # TODO: replace w/ proper closed form

        X_grad_next = cur_data_fidelity.prox(
            Z_curr,
            y,
            physics,
            gamma=self.algo_params["rho_sq"]
        )

        if self.clip_values:
            X_grad_next = torch.clamp(X_grad_next, self.clip_values[0], self.clip_values[1])

        # Update Z 
        prox_prior_val = cur_prior.prox(Z_curr, gamma=self.algo_params["lambda"])
        grad_Moreau_prior = (1.0 / self.algo_params["lambda"]) * (Z_curr - prox_prior_val)

        coupling_term_grad = (1.0 / self.algo_params["rho_sq"]) * (Z_curr - X_grad_next)

        # noise
        noise_std_dev = np.sqrt(2 * self.algo_params["step_size"])
        noise = torch.randn_like(Z_curr) * noise_std_dev

        # Update Z
        Z_next = (
            Z_curr
            - self.algo_params["step_size"] * grad_Moreau_prior
            - self.algo_params["step_size"] * coupling_term_grad
            + noise
        )

        if self.clip_values:
            Z_next = torch.clamp(Z_next, self.clip_values[0], self.clip_values[1])
            
        return {"x": X_grad_next, "z": Z_next}
