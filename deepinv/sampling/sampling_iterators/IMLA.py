import torch
from torch import Tensor
from typing import Dict, Optional, Tuple, Any

from deepinv.physics import Physics
from deepinv.optim.prior import Prior
from deepinv.optim.distance import L2Distance
from deepinv.optim.data_fidelity import DataFidelity
from deepinv.optim.optimizers import optim_builder
from deepinv.sampling.sampling_iterators.sample_iterator import SamplingIterator


class _InnerIMLADataFidelity(DataFidelity):
    """
    this is f(v) + 1/delta l2(v, u) 
    where apriori we have subbed v = 1/2x + 1/2 Xn
    and u = Xn + root(delta/2) xi
    """
    def __init__(self, original_data_fidelity: DataFidelity, u: Tensor, delta: float):
        super().__init__()
        self.original_df = original_data_fidelity
        self.u = u        
        self.d = L2Distance(sigma=2/delta)

    def fn(self, v: Tensor, y: Tensor, physics: Physics, *args, **kwargs) -> Tensor:
        # orig data fidelity cost 
        original_cost = self.original_df.fn(v, y, physics, *args, **kwargs)
        
        # l2 from v to u
        quad_cost = self.d.fn(v,self.u)

        return original_cost + quad_cost

    def grad(self, v: Tensor, y: Tensor, physics: Physics, *args, **kwargs) -> Tensor:
        # orig grad term
        original_grad = self.original_df.grad(v, y, physics, *args, **kwargs)

        # quadratic gradient
        quad_grad = self.d.grad(v,self.u)

        return original_grad + quad_grad



class IMLAIterator(SamplingIterator):
    """
    Implicit Midpoint Langevin Algorithm (IMLA) Iterator (using optim_builder).
    """

    def __init__(
        self,
        algo_params: Dict[str, Any],
        inner_optim_params: Dict[str, Any] = {
            "iteration": "PGD",
            "params_algo": {"stepsize": 1e-4},
            "max_iter": 50,
            "crit_conv": 1e-4,
        },
        clip: Optional[Tuple[float, float]] = None,
    ):
        super().__init__(algo_params)

        missing_params = []
        if "step_size" not in self.algo_params:
            missing_params.append("step_size")
        if "lambda" not in self.algo_params:
            missing_params.append("lambda")
        if missing_params:
            raise ValueError(
                f"Missing required IMLA parameters: {', '.join(missing_params)}"
            )

        self.inner_iteration = inner_optim_params.get("iteration", "PGD")
        self.inner_params_algo = inner_optim_params.get(
            "params_algo", {"stepsize": 1e-4}
        )
        self.inner_max_iter = int(inner_optim_params.get("max_iter", 50))
        self.inner_crit_conv = inner_optim_params.get("crit_conv", "residual")
        self.inner_early_stop = inner_optim_params.get("early_stop", True)
        self.inner_verbose = inner_optim_params.get("verbose", False)
        self.clip = clip


    def forward(
        self,
        x: Tensor,  # current state X_n
        y: Tensor,  # original measurement
        physics: Physics,  # original physics
        cur_data_fidelity: DataFidelity,  # original data fidelity
        cur_prior: Prior,  # original prior
    ) -> Tensor:
        delta = self.algo_params["step_size"]
        lambd = self.algo_params["lambda"]

        # prep for inner problem
        xi = torch.randn_like(x)
        sqrt_delta_half = torch.sqrt(
            torch.tensor(delta / 2.0, device=x.device, dtype=x.dtype)
        )

        # sub u = x + root(delta/2) xi
        u = x + sqrt_delta_half * xi

        # custom data fidelity for the inner problem
        inner_data_fidelity = _InnerIMLADataFidelity(
            original_data_fidelity=cur_data_fidelity, u=u, delta=delta
        )

        # config and run optimiser
        inner_params = self.inner_params_algo.copy()
        try:
            # pass the ORIGINAL prior and the NEW inner_data_fidelity
            # BUG: slow startup here i think
            # we should really cache this object and update the u value on data fidelity manutally
            # check ids of objects for caching
            inner_model = optim_builder(
                iteration=self.inner_iteration,
                prior=cur_prior,
                data_fidelity=inner_data_fidelity,
                max_iter=self.inner_max_iter,
                crit_conv=self.inner_crit_conv,
                early_stop=self.inner_early_stop,
                verbose=self.inner_verbose,
                params_algo={"lambda": lambd, **inner_params},
            )
        except Exception as e:
            raise RuntimeError(f"Failed to build inner optimizer: {e}") from e

        v_star = inner_model(
            y=y,  # original y
            physics=physics,  # original physics
        )

        # undo the sub X_{n+1} = 2*v - X_n
        x_next = 2.0 * v_star.detach() - x.detach()

        return x_next

