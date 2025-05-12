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
from deepinv.sampling.sampling_iterators.sample_iterator import SamplingIterator
from deepinv.optim.data_fidelity import DataFidelity

from typing import Dict, Optional, Tuple, Any

class ULAIterator(SamplingIterator):
    #TODO: latex for MYULA
    r"""
    Moreau-Yosida Unadjusted Langevin Algorithm.

    The algorithm runs the following markov chain iteration
    (Algorithm 1 from https://arxiv.org/pdf/2206.05350):

    where :math:`x_{k}` is the :math:`k` th sample of the Markov chain,
    :math:`\log p(y|x)` is the log-likelihood function, :math:`\log p(x)` is the log-prior,
    :math:`\eta>0` is the step size, :math:`\alpha>0` controls the amount of regularization,
    :math:`z\sim \mathcal{N}(0,I)` is a standard Gaussian vector.

    :return: Next state :math:`X_{t+1}` in the Markov chain
    :rtype: torch.Tensor

    """
    def __init__(self,algo_params:Dict[str, float], clip=None):
        super().__init__(algo_params)

        # Raise an error if these are not supplied
        missing_params = []
        if "step_size" not in algo_params:
            missing_params.append("step_size")
        if "lambda" not in algo_params:
            missing_params.append("lambda")

        if missing_params:
            raise ValueError(
                f"Missing required parameters for MYULA: {', '.join(missing_params)}"
            )

        self.clip = clip

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
        noise = torch.randn_like(x) * np.sqrt(2 * self.algo_params["step_size"])
        lhood = -cur_data_fidelity.grad(x, y, physics)
        
        lprior = (1/self.algo_params["lambda"]) * cur_prior.prox(x,gamma= self.algo_params["lambda"]) 
        x_t = (1- self.algo_params["step_size"]/self.algo_params["lambda"])*x + self.algo_params["step_size"] * (lhood + lprior) + noise
        if self.clip:
            x_t = projbox(x_t, self.clip[0], self.clip[1])
        return {"x": x_t}  # Return the updated x_t

