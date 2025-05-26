from torch import Tensor
import torch
import numpy as np
import time as time
from deepinv.physics import Physics
from deepinv.optim.prior import Prior, ScorePrior
from deepinv.sampling.sampling_iterators.sample_iterator import SamplingIterator
from deepinv.optim.data_fidelity import DataFidelity
import wandb
from typing import Dict, Optional, Tuple, Any

from deepinv.optim import BurgEntropy


# gradient of conjugate of phi
def grad_phi_star(x):
    if x.any() < 1e-6:
        print("warning, div by almost zero")
    return -1 / x


# gradient of phi
def grad_phi(x):
    if x.any() < 1e-6:
        print("warning, div by almost zero")
    return -1 / x


# second derivative of phi
def d2_phi(x):
    return 1 / x**2


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
        # print(f"new iteration")
        noise = torch.randn_like(x) *torch.sqrt(2 * self.algo_params["step_size"] * self.potential.hessian(x))
        # print(f"max: {torch.max(noise)} min: {torch.min(noise)}")
        # lprior = (
        #     -cur_prior.grad(x, self.algo_params["sigma"]) * self.algo_params["alpha"]
        # )
        # yk1 = grad_phi(x)
        yk1 = self.potential.grad(x)
        # print(f"max: {torch.max(yk1)} min: {torch.min(yk1)}")
        # yk2 = yk1 + self.algo_params["step_size"] * (lhood + lprior) + noise
        lhood = -cur_data_fidelity.grad(x, y, physics)
        # print(f"max: {torch.max(lhood)} min: {torch.min(lhood)}")
        lprior = (
            -cur_prior.grad(x, self.algo_params["sigma"]) * self.algo_params["alpha"]
        )
        # print(f"max: {torch.max(lprior)} min: {torch.min(lprior)}")
        yk2 = yk1 + self.algo_params["step_size"] * (lhood + lprior) + noise
        # print(f"max: {torch.max(yk2)} min: {torch.min(yk2)}")
        xk = self.potential.grad_conj(yk2).clamp(1e-6,None)
        # print(f"max: {torch.max(xk)} min: {torch.min(xk)}")

        # x_t = self.potential.grad_conj(
        #         self.potential.grad(x) +
        #         self.algo_params["step_size"] * (lhood + lprior)
        #         + noise
        # )
        # x_t = self.potential.grad_conj(self.potential.grad(x) - tau * grad_f(x, noisy)
        #                 - (config.alpha * tau * cur_prior.grad(x, sigma))
        #                 + torch.sqrt(2 * tau * self.potential.hessian(x)) * torch.randn_like(x))
        # wandb.log({"noise": wandb.Image(noise.squeeze().cpu()),
        #            "lhood": wandb.Image(lhood.squeeze().cpu()),
        #            "lprior": wandb.Image(lprior.squeeze().cpu()),
        #            "potential_grad_x": wandb.Image(self.potential.grad(x).squeeze().cpu()),
        #            "hessian": wandb.Image(self.potential.hessian(x).squeeze().cpu()),
        #            "grad_conj": wandb.Image(self.potential.grad(x).squeeze().cpu()),
        #            })
        return {"x": xk}  # Return the updated x_t
