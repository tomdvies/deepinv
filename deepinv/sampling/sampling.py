import sys
from collections import deque
from typing import Union, Dict, Callable, List, Tuple

import torch
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf as sm_plot_acf

from deepinv.models import Reconstructor
from deepinv.optim.data_fidelity import DataFidelity
from deepinv.optim.prior import Prior
from deepinv.optim.utils import check_conv
from deepinv.physics import Physics, LinearPhysics
from deepinv.sampling.sampling_iterators import *
from deepinv.sampling.utils import Welford


class BaseSampling(Reconstructor):
    r"""
    Base class for Monte Carlo sampling.

    This class aims to sample from the posterior distribution :math:`p(x|y)`, where :math:`y` represents the observed
    measurements and :math:`x` is the (unknown) image to be reconstructed. The sampling process generates a
    sequence of states (samples) :math:`X_0, X_1, \ldots, X_N` from a Markov chain. Each state :math:`X_k` contains the
    current estimate of the unknown image, denoted :math:`x_k`, and may include other latent variables.
    The class then computes statistics (e.g., image posterior mean, image posterior variance) from the samples :math:`X_k`.

    This class can be used to create new Monte Carlo samplers by implementing the sampling kernel through :class:`deepinv.sampling.SamplingIterator`:

    ::

        # define your sampler (possibly a Markov kernel which depends on the previous sample)
        class MyIterator(SamplingIterator):
            def __init__(self):
                super().__init__()

            def initialize_latent_variables(x, y, physics, data_fidelity, prior):
                # initialize a latent variable
                latent_z = g(x, y, physics, data_fidelity, prior)
                return {"x": x, "z": latent_z}

            def forward(self, X, y, physics, data_fidelity, prior, params_algo):
                # run one sampling kernel iteration
                new_X = f(X, y, physics, data_fidelity, prior, params_algo)
                return new_X

        # create the sampler
        sampler = BaseSampler(MyIterator(), prior, data_fidelity, iterator_params)

        # compute posterior mean and variance of reconstruction of x
        mean, var = sampler.sample(y, physics)

    This class computes the mean and variance of the chain using Welford's algorithm, which avoids storing the whole
    Monte Carlo samples. It can also maintain a history of the `history_size` most recent samples.

    Note on retained sample calculation:
        With the default parameters (max_iter=100, burnin_ratio=0.2, thinning=10), the number
        of samples actually used for statistics is calculated as follows:

        - Total iterations: 100
        - Burn-in period: 100 * 0.2 = 20 iterations (discarded)
        - Remaining iterations: 80
        - With thinning of 10, we keep iterations 20, 30, 40, 50, 60, 70, 80, 90
        - This results in 8 retained samples used for computing the posterior statistics

    :param deepinv.sampling.SamplingIterator iterator: The sampling iterator that defines the MCMC kernel
    :param deepinv.optim.DataFidelity data_fidelity: Negative log-likelihood function linked with the noise distribution in the acquisition physics
    :param deepinv.optim.Prior prior: Negative log-prior
    :param int max_iter: The number of Monte Carlo iterations to perform. Default: 100
    :param float burnin_ratio: Percentage of iterations used for burn-in period (between 0 and 1). Default: 0.2
    :param int thinning: Integer to thin the Monte Carlo samples (keeping one out of `thinning` samples). Default: 10
    :param float thresh_conv: The convergence threshold for the mean and variance. Default: ``1e-3``
    :param Callable callback: A function that is called on every (thinned) sample state dictionary for diagnostics. It is called with the current sample `X`, the current `statistics` (a list of Welford objects), and the current iteration number `iter` as keyword arguments.
    :param history_size: Number of most recent samples to store in memory. If `True`, all samples are stored. If `False`, no samples are stored. If an integer, it specifies the number of most recent samples to store. Default: 5
    :param bool verbose: Whether to print progress of the algorithm. Default: ``False``
    """

    def __init__(
        self,
        iterator: SamplingIterator,
        data_fidelity: DataFidelity,
        prior: Prior,
        max_iter: int = 100,
        callback: Callable = lambda X, **kwargs: None,
        burnin_ratio: float = 0.2,
        thresh_conv: float = 1e-3,
        crit_conv: str = "residual",
        thinning: int = 10,
        history_size: Union[int, bool] = 5,
        verbose: bool = False,
    ):
        super(BaseSampling, self).__init__()
        self.iterator = iterator
        self.data_fidelity = data_fidelity
        self.prior = prior
        self.max_iter = max_iter
        self.burnin_ratio = burnin_ratio
        self.thresh_conv = thresh_conv
        self.crit_conv = crit_conv
        self.callback = callback
        self.mean_convergence = False
        self.var_convergence = False
        self.thinning = thinning
        self.verbose = verbose
        self.history_size = history_size

        # Initialize history to zero
        if history_size is True:
            self.history = []
        elif history_size:
            self.history = deque(maxlen=history_size)
        # else:
        #     self.history = False

    def forward(
        self,
        y: torch.Tensor,
        physics: Physics,
        x_init: Union[torch.Tensor, None] = None,
        seed: Union[int, None] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Run the MCMC sampling chain and return the posterior sample mean.

        :param torch.Tensor y: The observed measurements
        :param Physics physics: Forward operator of your inverse problem
        :param torch.Tensor x_init: Initial state of the Markov chain. If None, uses ``physics.A_adjoint(y)`` as the starting point
            Default: ``None``
        :param int seed: Optional random seed for reproducible sampling.
            Default: ``None``
        :return: Posterior sample mean
        :rtype: torch.Tensor
        """

        # pass back out sample mean
        return self.sample(y, physics, x_init=x_init, seed=seed, **kwargs)[0]

    def sample(
        self,
        y: torch.Tensor,
        physics: Physics,
        x_init: Union[torch.Tensor, None] = None,
        seed: Union[int, None] = None,
        g_statistics: Union[Callable, List[Callable]] = [lambda d: d["x"]],
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Execute the MCMC sampling chain and compute posterior statistics.

        This method runs the main MCMC sampling loop to generate samples from the posterior
        distribution and compute their statistics using Welford's online algorithm.

        :param torch.Tensor y: The observed measurements/data tensor
        :param Physics physics: Forward operator of your inverse problem.
        :param torch.Tensor x_init: Initial state of the Markov chain. If None, uses ``physics.A_adjoint(y)`` as the starting point
            Default: ``None``
        :param int seed: Optional random seed for reproducible sampling.
            Default: ``None``
        :param list g_statistics: List of functions for which to compute posterior statistics.
            The sampler will compute the posterior mean and variance of each function in the list.
            The input to these functions is a dictionary `d` which contains the current state of the sampler alongside any latent variables. `d["x"]` will always be the current image. See specific iterators for details on what (if any) latent variables they provide.
            Default: ``lambda d: d["x"]`` (identity function on the image).
        :param Union[List[Callable], Callable] g_statistics: List of functions for which to compute posterior statistics, or a single function.
        :param kwargs: Additional arguments passed to the sampling iterator (e.g., proposal distributions)
        :return: | If a single g_statistic was specified: Returns tuple (mean, var) of torch.Tensors
            | If multiple g_statistics were specified: Returns tuple (means, vars) of lists of torch.Tensors

        Example:
            >>> # Basic usage with default settings
            >>> sampler = BaseSampling(iterator, data_fidelity, prior)
            >>> mean, var = sampler.sample(measurements, forward_operator)

            >>> # Using multiple statistics
            >>> sampler = BaseSampling(
            ...     iterator, data_fidelity, prior,
            ...     g_statistics=[lambda X: X["x"], lambda X: X["x"]**2]
            ... )
            >>> means, vars = sampler.sample(measurements, forward_operator)
        """

        # Don't store computational graphs
        with torch.no_grad():
            # Set random seed if provided
            if seed is not None:
                torch.manual_seed(seed)

            # Initialization of both our image chain and any latent variables
            if x_init is None:
                # if linear take adjoint (pseudo-inverse can be a bit unstable) else fall back to pseudoinverse
                if isinstance(physics, LinearPhysics):
                    X = self.iterator.initialize_latent_variables(
                        physics.A_adjoint(y), y, physics, self.data_fidelity, self.prior
                    )
                else:
                    X = self.iterator.initialize_latent_variables(
                        physics.A_dagger(y), y, physics, self.data_fidelity, self.prior
                    )
            else:
                X = self.iterator.initialize_latent_variables(
                    x_init, y, physics, self.data_fidelity, self.prior
                )

            if self.history_size:
                self.history.append(X)

            self.mean_convergence = False
            self.var_convergence = False

            if not isinstance(g_statistics, List):
                g_statistics = [g_statistics]

            # Initialize Welford trackers for each g_statistic
            statistics = []
            for g in g_statistics:
                statistics.append(Welford(g(X)))

            # Initialize for convergence checking
            mean_prevs = [stat.mean().clone() for stat in statistics]
            var_prevs = [stat.var().clone() for stat in statistics]

            # Run the chain
            for it in tqdm(range(self.max_iter), disable=(not self.verbose)):
                X = self.iterator(
                    X,
                    y,
                    physics,
                    self.data_fidelity,
                    self.prior,
                    it,
                    **kwargs,
                )

                if (
                    it >= (self.max_iter * self.burnin_ratio)
                    and it % self.thinning == 0
                ):
                    self.callback(X=X, statistics=statistics, iter=it)
                    # Store previous means and variances for convergence check
                    if it >= (self.max_iter - self.thinning):
                        mean_prevs = [stat.mean().clone() for stat in statistics]
                        var_prevs = [stat.var().clone() for stat in statistics]

                    if self.history_size:
                        self.history.append(X)

                    for _, (g, stat) in enumerate(zip(g_statistics, statistics)):
                        stat.update(g(X))

            # Check convergence for all statistics
            self.mean_convergence = True
            self.var_convergence = True

            if it > 1:
                # Check convergence for each statistic
                for j, stat in enumerate(statistics):
                    if not check_conv(
                        {"est": (mean_prevs[j],)},
                        {"est": (stat.mean(),)},
                        it,
                        self.crit_conv,
                        self.thresh_conv,
                        self.verbose,
                    ):
                        self.mean_convergence = False

                    if not check_conv(
                        {"est": (var_prevs[j],)},
                        {"est": (stat.var(),)},
                        it,
                        self.crit_conv,
                        self.thresh_conv,
                        self.verbose,
                    ):
                        self.var_convergence = False

            # Return means and variances for all g_statistics
            means = [stat.mean() for stat in statistics]
            vars = [stat.var() for stat in statistics]

            # Unwrap single statistics
            if len(g_statistics) == 1:
                return means[0], vars[0]
        return means, vars


    def plot_acf(self, g_statistic: Union[Callable, None] = None, lags: Union[int, None] = None,
                 title: str = "ACF Plot (Tensor)", save_path: Union[str, None] = None,
                 img_type: str = "png", **kwargs):
        r"""
        Plot autocorrelation functions for Fourier components of the stored samples.

        This method computes the 2D Fourier transform of the samples (e.g., images)
        in the history, calculates the variance of each Fourier component across the
        samples, and then plots the ACF for the components with minimum, maximum,
        and median variance.

        :param Callable g_statistic: A function to extract the data (e.g., image tensor)
            from each history item (dictionary). If ``None``, defaults to extracting
            the tensor associated with the key ``"x"``. The function must return a
            ``torch.Tensor``. Default: ``None``.
        :param int, optional lags: The number of lags to include in the ACF plot.
            If ``None``, ``statsmodels.graphics.tsaplots.plot_acf`` will use its default.
            Default: ``None``.
        :param str title: Title for the plot. Default: ``"ACF Plot (Tensor)"``.
        :param str, optional save_path: Directory path to save the plot. If ``None``,
            the plot is not saved. Default: ``None``.
        :param str img_type: Image type/extension for saving (e.g., "png", "pdf").
            Default: ``"png"``.
        :param kwargs: Additional keyword arguments passed to
            ``statsmodels.graphics.tsaplots.plot_acf``.
        :raises RuntimeError: If history storage was disabled (``history_size=False``).
        """
        # TODO: return ax return fig plot... (look at deepinv.utils.plot)
        if self.history is False:
            raise RuntimeError(
                "Cannot plot ACF: history storage is disabled (history_size=False)"
            )

        chain_items = self.get_chain() # This now returns List[Dict[str, Any]]
        if not chain_items or len(chain_items) < 2: # Need at least 2 samples for variance/ACF
            print("History is empty or too short (need at least 2 samples), cannot plot ACF.")
            return
        with torch.no_grad(): 
            if g_statistic is None:
                g = lambda X: X["x"] # Default to extracting "x" from the dict
            else:
                g = g_statistic

            try:
                MC_chain = None # Initialize as None
                for item_idx, item in enumerate(chain_items):
                    tensor_val = g(item)                    
                    # Add a new leading dimension for stacking
                    current_tensor_item_expanded = tensor_val.unsqueeze(0)

                    if MC_chain is None:
                        MC_chain = current_tensor_item_expanded
                    else:
                        # Ensure consistent shapes for concatenation (excluding the batch dimension)
                        if current_tensor_item_expanded.shape[1:] != MC_chain.shape[1:]: # Check shape from 2nd dim onwards
                            raise ValueError(
                                f"Inconsistent item shapes for concatenation. Expected shape like {MC_chain.shape[1:]}, got {current_tensor_item_expanded.shape[1:]} for item {item_idx}"
                            )
                        MC_chain = torch.cat((MC_chain, current_tensor_item_expanded), dim=0)
            except Exception as e:
                print(f"Error processing chain for ACF plot: {e}")
                print("Ensure g_statistic extracts a suitable tensor from history items (dictionaries).")
                return

            MC_chain_fourier = torch.abs(torch.fft.fft2(MC_chain, dim=(-2, -1)))
                
            variance_array_fourier = torch.var(MC_chain_fourier, dim=0)

            variance_flat = variance_array_fourier.reshape(-1) # Flatten

            if variance_flat.numel() == 0: # Use numel() for tensor size
                print("Fourier variance array is empty. Cannot plot ACF.")
                return

            mc_chain_fourier_flat_features = MC_chain_fourier.reshape(MC_chain_fourier.shape[0], -1)


            ind_min_variance = torch.argmin(variance_flat)
            chain_elem_min_variance = mc_chain_fourier_flat_features[:, ind_min_variance]

            ind_max_variance = torch.argmax(variance_flat)
            chain_elem_max_variance = mc_chain_fourier_flat_features[:, ind_max_variance]

            sorted_indices = torch.argsort(variance_flat)
            ind_median_variance = sorted_indices[len(sorted_indices) // 2]
            chain_elem_median_variance = mc_chain_fourier_flat_features[:, ind_median_variance]

            rc_params = {
                'figure.figsize': (15, 15), 'font.size': 12, 
                'axes.titlesize': 14, 'axes.labelsize': 14, 
                'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 10
            }
            with plt.rc_context(rc_params):
                fig, ax = plt.subplots()

                sm_plot_acf(chain_elem_median_variance, ax=ax, label='Median-speed component', alpha=None, lags=lags, **kwargs)
                sm_plot_acf(chain_elem_max_variance, ax=ax, label='Slowest component', alpha=None, lags=lags, **kwargs)
                sm_plot_acf(chain_elem_min_variance, ax=ax, label='Fastest component', alpha=None, lags=lags, **kwargs)

                handles, labels_from_plot = ax.get_legend_handles_labels()
                handles = handles[1::2]
                labels_from_plot = labels_from_plot[1::2]
                
                ax.legend(handles=handles, labels=labels_from_plot, loc='best', shadow=True, numpoints=1)
                ax.set_title(title)
                ax.set_ylabel("ACF")
                ax.set_xlabel("Lags")

                if save_path:
                    safe_title = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in title).rstrip().replace(' ', '_')
                    filename = f"acr_{safe_title}.{img_type}"
                    full_save_path = f"{save_path}/{filename}"
                    try:
                        plt.savefig(full_save_path, bbox_inches='tight', dpi=300)
                        print(f"ACF plot saved to {full_save_path}")
                    except Exception as e:
                        print(f"Error saving plot to {full_save_path}: {e}")
                plt.show()

    def get_chain(self) -> List[Dict[str, torch.Tensor]]: # Corrected type hint
        r"""
        Retrieve the stored history of samples.

        Returns a list of dictionaries, where each dictionary contains the state of the sampler.

        Only includes samples after the burn-in period and thinning.

        :return: List of stored sample states (dictionaries) from oldest to newest. Each dictionary contains the sample `"x": x` along with any latent variables.
        :rtype: list[dict]
        :raises RuntimeError: If history storage was disabled (history_size=False)

        Example:
            >>> sampler = BaseSampling(iterator, data_fidelity, prior, history_size=5)
            >>> _ = sampler(measurements, forward_operator)
            >>> history = sampler.get_chain()
            >>> latest_state = history[-1]  # Get most recent state dictionary
            >>> latest_sample = latest_state["x"] # Get sample from state
        """
        if self.history is False:
            raise RuntimeError(
                "Cannot get chain: history storage is disabled (history_size=False)"
            )
        return list(self.history)

    @property
    def mean_has_converged(self) -> bool:
        r"""
        Returns a boolean indicating if the posterior mean verifies the convergence criteria.
        """
        return self.mean_convergence

    @property
    def var_has_converged(self) -> bool:
        r"""
        Returns a boolean indicating if the posterior variance verifies the convergence criteria.
        """
        return self.var_convergence


def create_iterator(
    iterator: Union[SamplingIterator, str], cur_params, **kwargs
) -> SamplingIterator:
    r"""
    Helper function for creating an iterator instance of the :class:`deepinv.sampling.SamplingIterator` class.

    :param iterator: Either a SamplingIterator instance or a string naming the iterator class
    :return: SamplingIterator instance
    """
    if isinstance(iterator, str):
        # If a string is provided, create an instance of the named class
        iterator_fn = getattr(sys.modules[__name__], iterator + "Iterator")
        return iterator_fn(cur_params, **kwargs)
    else:
        # If already a SamplingIterator instance, return as is
        return iterator


def sampling_builder(
    iterator: Union[SamplingIterator, str],
    data_fidelity: DataFidelity,
    prior: Prior,
    params_algo: Dict = {},
    max_iter: int = 100,
    thresh_conv: float = 1e-3,
    burnin_ratio: float = 0.2,
    thinning: int = 10,
    history_size: Union[int, bool] = 5,
    verbose: bool = False,
    callback: Callable = lambda X, **kwargs: None,
    **kwargs,
) -> BaseSampling:
    r"""
    Helper function for building an instance of the :class:`deepinv.sampling.BaseSampling` class.
    See the docs for :class:`deepinv.sampling.BaseSampling` for examples and more information.

    :param iterator: Either a SamplingIterator instance or a string naming the iterator class
    :param data_fidelity: Negative log-likelihood function
    :param prior: Negative log-prior
    :param params_algo: Dictionary containing the parameters for the algorithm
    :param max_iter: Number of Monte Carlo iterations
    :param burnin_ratio: Percentage of iterations for burn-in
    :param thinning: Integer to thin the Monte Carlo samples
    :param history_size: Number of most recent samples to store in memory. If `True`, all samples are stored. If `False`, no samples are stored. If an integer, it specifies the number of most recent samples to store. Default: 5
    :param verbose: Whether to print progress
    :param Callable callback: A function that is called on every (thinned) sample state dictionary for diagnostics. It is called with the current sample `X`, the current `statistics` (a list of Welford objects), and the current iteration number `iter` as keyword arguments.
    :param kwargs: Additional keyword arguments passed to the iterator constructor when a string is provided as the iterator parameter
    :return: Configured BaseSampling instance in eval mode
    """
    iterator = create_iterator(iterator, params_algo, **kwargs)
    # Note we put the model in evaluation mode (.eval() is a PyTorch method inherited from nn.Module)
    return BaseSampling(
        iterator,
        data_fidelity=data_fidelity,
        prior=prior,
        max_iter=max_iter,
        thresh_conv=thresh_conv,
        burnin_ratio=burnin_ratio,
        thinning=thinning,
        history_size=history_size,
        verbose=verbose,
        callback=callback,
    ).eval()
