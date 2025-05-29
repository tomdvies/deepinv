from torch import Tensor
import torch
import numpy as np
import time as time
from deepinv.physics import Physics
from deepinv.optim.prior import Prior, ScorePrior
from deepinv.sampling.sampling_iterators.sampling_iterator import SamplingIterator
from deepinv.optim.data_fidelity import DataFidelity
import wandb
from typing import Dict, Optional, Tuple, Any

from deepinv.optim import BurgEntropy



class MLAIterator(SamplingIterator):
    def __init__(self, algo_params: Dict[str, float], **kwargs):
        super().__init__(algo_params)

        # Raise an error if these are not supplied
        missing_params = []
        if "step_size" not in algo_params:
            missing_params.append("step_size")
        if "alpha" not in algo_params:
            missing_params.append("alpha")
        if "sigma" not in algo_params:
            missing_params.append("sigma")

        if missing_params:
            raise ValueError(
                f"Missing required parameters for MYULA: {', '.join(missing_params)}"
            )

        self.potential = BurgEntropy()

    def forward(
        self,
        X: Dict[str, Tensor],
        y: Tensor,
        physics: Physics,
        cur_data_fidelity: DataFidelity,
        cur_prior: Prior,
        *args,
        **kwargs,
    ) -> Dict[str, Tensor]:
        x = X["x"]
        noise = torch.randn_like(x) *torch.sqrt(2 * self.algo_params["step_size"] * self.potential.hessian(x))
        yk1 = self.potential.grad(x)
        lhood = -cur_data_fidelity.grad(x, y, physics)
        lprior = (
            -cur_prior.grad(x, self.algo_params["sigma"]) * self.algo_params["alpha"]
        )
        yk2 = yk1 + self.algo_params["step_size"] * (lhood + lprior) + noise
        xk = self.potential.grad_conj(yk2).clamp(1e-6,None)
        return {"x": xk}  # Return the updated x_t
