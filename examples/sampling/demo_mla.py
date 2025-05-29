import deepinv as dinv
from deepinv.utils.plotting import plot
import torch
from deepinv.utils.demo import load_url_image

# %%
# Load image from the internet
# --------------------------------------------
#
# This example uses an image of Lionel Messi from Wikipedia.

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

url = (
    "https://upload.wikimedia.org/wikipedia/commons/b/b4/"
    "Lionel-Messi-Argentina-2022-FIFA-World-Cup_%28cropped%29.jpg"
)
x = load_url_image(url=url, img_size=32).to(device)

# %%
# Define forward operator and noise model
# --------------------------------------------------------------
#
# This example uses the identity forward operator and Poisson noise as the noise model.

physics = dinv.physics.DecomposablePhysics()#
gain = 1/40
physics.noise_model = dinv.physics.PoissonNoise(gain=gain)

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

# %%
# Define the likelihood
# --------------------------------------------------------------
#
# Since the noise model is Poisson.

# load Poisson Likelihood
likelihood = dinv.optim.PoissonLikelihood(gain=gain)

# %%
# Define the prior
# -------------------------------------------
#
# The score a distribution can be approximated using Tweedie's formula via the
# :class:`deepinv.optim.ScorePrior` class.
#
# .. math::
#
#            \nabla \log p_{\sigma}(x) \approx \frac{1}{\sigma^2} \left(D(x,\sigma)-x\right)
#
# This example uses a pretrained DnCNN model.
# From a Bayesian point of view, the score plays the role of the gradient of the
# negative log prior
# The hyperparameter ``sigma_denoiser`` (:math:`sigma`) controls the strength of the prior.
#
# In this example, we use a pretrained DnCNN model using the :class:`deepinv.loss.FNEJacobianSpectralNorm` loss,
# which makes sure that the denoiser is firmly non-expansive (see
# `"Building firmly nonexpansive convolutional neural networks" <https://hal.science/hal-03139360>`_), and helps to
# stabilize the sampling algorithm.

sigma_denoiser = 2 / 255
prior = dinv.optim.ScorePrior(
    # denoiser=dinv.models.DRUNet()
    denoiser=dinv.models.DnCNN(pretrained="download_lipschitz")
).to(device)

# %%
# Create the MCMC sampler
# --------------------------------------------------------------
#
# Here we use the Mirror Langevin Algorithm (MLA) to sample from the posterior defined in
# :class:`deepinv.sampling.MLAIterator`.
# The hyperparameter ``step_size`` controls the step size of the MCMC sampler,
# ``regularization`` controls the strength of the prior and
# ``iterations`` controls the number of iterations of the sampler.


def call(statistics, **kwargs):
    print(f"PSNR: {dinv.metric.PSNR()(x, statistics[0].mean()).item():.2f} dB")
regularization = 1
step_size = 7e-6
iterations = int(5e3) if torch.cuda.is_available() else 5000
params = {
    "step_size": step_size,
    "alpha": regularization,
    "sigma": sigma_denoiser,
}
f = dinv.sampling.sampling_builder(
    "MLA",
    prior=prior,
    data_fidelity=likelihood,
    max_iter=iterations,
    params_algo=params,
    callback=call,
    burnin_ratio=0.1,
    thinning=5,
    clip=(1e-4, None),
    verbose=True,
)

# %%
# Generate the measurement
# --------------------------------------------------------------
# We apply the forward model to generate the noisy measurement.

y = physics(x).clamp(min=1e-6)

# compute linear inverse
x_lin = physics.A_adjoint(y)

# %%
# Run sampling algorithm and plot results
# --------------------------------------------------------------
# The sampling algorithm returns the posterior mean and variance.
# We compare the posterior mean with a simple linear reconstruction.

mean, var = f.sample(y, physics)

# compute PSNR
print(f"Linear reconstruction PSNR: {dinv.metric.PSNR()(x, x_lin).item():.2f} dB")
print(f"Posterior mean PSNR: {dinv.metric.PSNR()(x, mean).item():.2f} dB")

# plot results
error = (mean - x).abs().sum(dim=1).unsqueeze(1)  # per pixel average abs. error
std = var.sum(dim=1).unsqueeze(1).sqrt()  # per pixel average standard dev.
imgs = [x_lin, x, mean, std / std.flatten().max(), error / error.flatten().max()]
plot(
    imgs,
    titles=["measurement", "ground truth", "post. mean", "post. std", "abs. error"],
)
