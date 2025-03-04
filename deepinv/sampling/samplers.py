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
from deepinv.sampling.utils import Welford, projbox
from deepinv.sampling.sampling_iterators import *


# TODO: add in some common statistics like mean/ pixel? 
# to avoid having to supply g_statistics functions manually
# TODO: check_conv stuff
# TODO: Reconstructor, return sample mean
# TODO: Add rng
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
    :param tuple(int,int) clip: Tuple of (min, max) values to clip/project the samples into a bounded range during sampling. 
        Useful for images where pixel values should stay within a specific range (e.g., (0,1) or (0,255)). Default: ``None``
    :param float burnin_ratio: Percentage of iterations used for burn-in period (between 0 and 1). Default: 0.2
    :param int thinning: Integer to thin the Monte Carlo samples (keeping one out of `thinning` samples). Default: 10
    :param int history_size: Number of most recent samples to store in memory. Default: 5
    :param bool verbose: Whether to print progress of the algorithm. Default: ``False``
    """

    def __init__(
        self,
        iterator: SamplingIterator,
        data_fidelity: DataFidelity,
        prior: Prior,
        params_algo={"lambda": 1.0, "stepsize": 1.0},
        # NOTE: max_iter
        num_iter=100,
        # TODO: pass to iterator
        clip = None,
        burnin_ratio=0.2,
        thinning=10,
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
        self.verbose = verbose
        self.clip = clip
        self.history_size = history_size
        # Stores last history_size samples note float('inf') => we store the whole chain
        self.history = deque(maxlen=history_size)

    def forward(
        self,
        y: torch.Tensor,
        physics: Physics,
        X_init: torch.Tensor | None = None,
        seed: int | None = None,
        g_statistics=[lambda x: x],
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
        :param list g_statistics: List of functions for which to compute posterior statistics. Default: ``[lambda x: x]``
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
        for g in g_statistics:
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
            if self.clip:
                X_t = projbox(X_t, self.clip[0], self.clip[1]) 
            if i >= (self.num_iter * self.burnin_ratio) and i % self.thinning == 0:
                self.history.append(X_t)

                for j, (g, stat) in enumerate(zip(g_statistics, statistics)):
                    stat.update(g(X_t))

        # Return means and variances for all g_statistics
        means = [stat.mean() for stat in statistics]
        vars = [stat.var() for stat in statistics]

        # Unwrap single statistics
        if len(g_statistics) == 1:
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


def create_iterator(
    iterator: SamplingIterator | str,
) -> SamplingIterator:
    r"""
    Helper function for creating an iterator instance of the :class:`deepinv.sampling.SamplingIterator` class.

    :param iterator: Either a SamplingIterator instance or a string naming the iterator class
    :return: SamplingIterator instance
    """
    if isinstance(iterator, str):
        # If a string is provided, create an instance of the named class
        iterator_fn = str_to_class(iterator + "Iterator")
        return iterator_fn()
    else:
        # If already a SamplingIterator instance, return as is
        return iterator


def sample_builder(
    iterator: SamplingIterator | str,
    data_fidelity: DataFidelity,
    prior: Prior,
    params_algo={},
    num_iter=100,
    clip = None,
    burnin_ratio=0.2,
    thinning=10,
    history_size=5,
    verbose=False,
):
    # TODO: make these docs better
    r"""
    Helper function for building an instance of the :class:`deepinv.optim.BaseSample` class.

    :param iterator: Either a SamplingIterator instance or a string naming the iterator class
    :param data_fidelity: Negative log-likelihood function
    :param prior: Negative log-prior
    :param params_algo: Dictionary containing the parameters for the algorithm
    :param num_iter: Number of Monte Carlo iterations
    :param clip: Tuple of (min, max) values to clip samples
    :param burnin_ratio: Percentage of iterations for burn-in
    :param thinning: Integer to thin the Monte Carlo samples
    :param history_size: Number of recent samples to store
    :param verbose: Whether to print progress
    :return: Configured BaseSample instance in eval mode
    """
    iterator = create_iterator(iterator)
    # Note we put the model in evaluation mode (.eval() is a PyTorch method inherited from nn.Module)
    return BaseSample(
        iterator,
        data_fidelity=data_fidelity,
        prior=prior,
        params_algo=params_algo,
        num_iter=num_iter,
        clip=clip,
        burnin_ratio=burnin_ratio,
        thinning=thinning,
        history_size=history_size,
        verbose=verbose
    ).eval()


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)
