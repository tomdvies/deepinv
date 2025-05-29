import torch
from torch import Tensor
from typing import Dict, Optional, Tuple, Any

from deepinv.physics import Physics
from deepinv.optim.prior import Prior
from deepinv.optim.distance import L2Distance
from deepinv.optim.data_fidelity import DataFidelity
from deepinv.optim.optimizers import optim_builder
from deepinv.sampling.sampling_iterators.sampling_iterator import SamplingIterator


# TODO: add check for explicit prior
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
        self.d = L2Distance(sigma=2 / delta)

    def fn(self, v: Tensor, y: Tensor, physics: Physics, *args, **kwargs) -> Tensor:
        # orig data fidelity cost
        original_cost = self.original_df.fn(v, y, physics, *args, **kwargs)

        # l2 from v to u
        quad_cost = self.d.fn(v, self.u)

        return original_cost + quad_cost

    def grad(self, v: Tensor, y: Tensor, physics: Physics, *args, **kwargs) -> Tensor:
        # orig grad term
        original_grad = self.original_df.grad(v, y, physics, *args, **kwargs)

        # quadratic gradient
        quad_grad = self.d.grad(v, self.u)

        return original_grad + quad_grad


# BUG: IMLA tests not passing atm
class IMLAIterator(SamplingIterator):
    """
    Implicit Midpoint Langevin Algorithm (IMLA) Iterator (using optim_builder).
    """

    def __init__(
        self,
        algo_params: Dict[str, Any],
        inner_optim_params: Dict[str, Any] = {
            "iteration": "PGD",
            "params_algo": {"lambda": 1.0, "stepsize": 1e-6},
            "max_iter": 5000,
            "thres_conv": 1e-4,
            "crit_conv": "residual",
            "verbose": True,
            "early_stop": True,
        },
        clip: Optional[Tuple[float, float]] = None,
    ):
        super().__init__(algo_params)

        missing_params = []
        if "step_size" not in self.algo_params:
            missing_params.append("step_size")
        if missing_params:
            raise ValueError(
                f"Missing required IMLA parameters: {', '.join(missing_params)}"
            )

        self.inner_params_algo = inner_optim_params

    def forward(
        self,
        X: Dict[str, Tensor],  # current state X_n
        y: Tensor,  # original measurement
        physics: Physics,  # original physics
        cur_data_fidelity: DataFidelity,  # original data fidelity
        cur_prior: Prior,  # original prior
        iteration: int,
    ) -> Dict[str, Tensor]:
        x = X["x"]
        if not hasattr(self, "ids"):
            self.ids = [id(cur_data_fidelity), id(cur_prior)]
        delta = self.algo_params["step_size"]

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
        try:
            # pass the ORIGINAL prior and the NEW inner_data_fidelity
            # BUG: slow startup here i think
            # we should really cache this object and update the u value on data fidelity manutally
            # check ids of objects for caching
            inner_model = optim_builder(
                prior=cur_prior,
                data_fidelity=inner_data_fidelity,
                **self.inner_params_algo,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to build inner optimizer: {e}") from e

        v_star = inner_model(
            y=y,  # original y
            physics=physics,  # original physics
        )

        # undo the sub X_{n+1} = 2*v - X_n
        x_next = 2.0 * v_star.detach() - x.detach()

        return {"x": x_next}
