import sys
import warnings
from collections import deque
from collections.abc import Iterable
import torch
from torch import nn
from tqdm import tqdm
from deepinv.physics import Physics
from deepinv.optim.optim_iterators import *
from deepinv.optim.fixed_point import FixedPoint
from deepinv.optim.prior import Zero, Prior
from deepinv.loss.metric.distortion import PSNR
from deepinv.models import Reconstructor
from deepinv.optim.data_fidelity import DataFidelity
from deepinv.sampling.sampling_iterators.sample_iterator import SamplingIterator
from deepinv.sampling.utils import Welford


class BaseSample(nn.Module):
    r"""
    Base class for Monte Carlo sampling.

    This class can be used to create new Monte Carlo samplers by implementing the sampling kernel through :class:`deepinv.sampling.SamplingIterator`:

    ::

        # define your sampler (possibly a Markov kernel which depends on the previous sample)
        class MyIterator(SamplingIterator):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, physics, data_fidelity, prior, params_algo):
                # run one sampling kernel iteration
                new_x = f(x, y, physics, data_fidelity, prior, params_algo)
                return new_x
        
        # create the sampler
        sampler = BaseSampler(MyIterator(), prior, data_fidelity, iterator_params)

        # compute posterior mean and variance of reconstruction of x
        mean, var = sampler(y, physics)

    This class computes the mean and variance of the chain using Welford's algorithm, which avoids storing the whole
    Monte Carlo samples. It can also maintain a history of the `history_size` most recent samples.

    :param deepinv.sampling.SamplingIterator iterator: The sampling iterator that defines the MCMC kernel
    :param deepinv.optim.DataFidelity data_fidelity: Negative log-likelihood function linked with the noise distribution in the acquisition physics
    :param deepinv.optim.Prior prior: Negative log-prior
    :param dict params_algo: Dictionary containing the parameters for the algorithm 
    :param int num_iter: Number of Monte Carlo iterations. Default: 100
    :param float burnin_ratio: Percentage of iterations used for burn-in period (between 0 and 1). Default: 0.2
    :param int thinning: Integer to thin the Monte Carlo samples (keeping one out of `thinning` samples). Default: 10
    :param list g_statistics: List of functions for which to compute posterior statistics. Default: ``[lambda x: x]``
    :param int history_size: Number of most recent samples to store in memory. Default: 5
    :param bool verbose: Whether to print progress of the algorithm. Default: ``False``
    """

    def __init__(
        self,
        iterator: SamplingIterator,
        data_fidelity: DataFidelity,
        prior: Prior,
        params_algo={"lambda": 1.0, "stepsize": 1.0},
        num_iter=100,
        burnin_ratio=0.2,
        thinning=10,
        g_statistics=[lambda x: x],
        history_size=5,
        verbose=False,
    ):
        super(BaseSample, self).__init__()
        self.iterator = iterator
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.params_algo = params_algo
        self.num_iter = num_iter
        self.burnin_ratio = burnin_ratio
        self.thinning = thinning
        self.g_statistics = g_statistics
        self.verbose = verbose
        self.history_size = history_size
        # Stores last history_size samples note float('inf') => we store the whole chain
        self.history = deque(maxlen=history_size)

    def forward(
        self,
        y: torch.Tensor,
        physics: Physics,
        X_init: torch.Tensor | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        r"""
        Execute the MCMC sampling chain and compute posterior statistics.

        This method runs the main MCMC sampling loop to generate samples from the posterior
        distribution and compute their statistics using Welford's online algorithm.

        :param torch.Tensor y: The observed measurements/data tensor
        :param Physics physics: Forward operator of your inverse problem.
        :param torch.Tensor X_init: Initial state of the Markov chain. If None, uses ``physics.A_adjoint(y)`` as the starting point
            Default: ``None``
        :param int seed: Optional random seed for reproducible sampling.
            Default: ``None``
        :param kwargs: Additional arguments passed to the sampling iterator (e.g., proposal distributions)
        :return: | If a single g_statistic was specified: Returns tuple (mean, var) of torch.Tensors
            | If multiple g_statistics were specified: Returns tuple (means, vars) of lists of torch.Tensors

        Example:
            >>> # Basic usage with default settings
            >>> sampler = BaseSample(iterator, data_fidelity, prior)
            >>> mean, var = sampler(measurements, forward_operator)

            >>> # Using multiple statistics
            >>> sampler = BaseSample(
            ...     iterator, data_fidelity, prior,
            ...     g_statistics=[lambda x: x, lambda x: x**2]
            ... )
            >>> means, vars = sampler(measurements, forward_operator)
        """
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Initialization
        if X_init is None:
            X_t = physics.A_adjoint(y)
        else:
            X_t = X_init

        self.history = deque([X_t], maxlen=self.history_size)

        # Initialize Welford trackers for each g_statistic
        statistics = []
        for g in self.g_statistics:
            statistics.append(Welford(g(X_t)))

        # Run the chain
        for i in tqdm(range(self.num_iter), disable=(not self.verbose)):
            X_t = self.iterator(
                X_t,
                y,
                physics,
                self.data_fidelity,
                self.prior,
                self.params_algo,
                **kwargs,
            )

            if i >= (self.num_iter * self.burnin_ratio) and i % self.thinning == 0:
                self.history.append(X_t)

                for j, (g, stat) in enumerate(zip(self.g_statistics, statistics)):
                    stat.update(g(X_t))

            if self.verbose and i % (self.num_iter // 10) == 0:
                print(f"Iteration {i}/{self.num_iter}")

        # Return means and variances for all g_statistics
        means = [stat.mean() for stat in statistics]
        vars = [stat.var() for stat in statistics]

        # Unwrap single statistics
        if len(self.g_statistics) == 1:
            return means[0], vars[0]
        return means, vars

    def get_history(self) -> list[torch.Tensor]:
        r"""
        Retrieve the stored history of samples.

        Returns a list of samples. 
        
        Only includes samples after the burn-in period and, thinning.

        :return: List of stored samples from oldest to newest
        :rtype: list[torch.Tensor]

        Example:
            >>> sampler = BaseSample(iterator, data_fidelity, prior, history_size=5)
            >>> _ = sampler(measurements, forward_operator)
            >>> samples = sampler.get_history()
            >>> latest_sample = samples[-1]  # Get most recent sample
        """
        return list(self.history)
