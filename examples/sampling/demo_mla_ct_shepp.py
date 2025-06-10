import deepinv as dinv
from deepinv.utils.plotting import plot
import torch
from deepinv.utils.demo import get_image_url, load_url_image

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

# %%
# Load image (shepp logan phantom)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

im_name = ("SheppLogan.png")
img_size = 64

x = load_url_image(get_image_url(im_name), img_size=img_size, grayscale=True,
    resize_mode="resize",
    device=device).to(device)

# %%
# Define forward operator and noise model
# --------------------------------------------------------------
#
# This example uses the CT operator and Poisson noise as the noise model.

noise_level_img = 1/20#0.03
angles = 100
n_channels = 1  # 3 for color images, 1 for gray-scale images
physics = dinv.physics.Tomography(
    img_width=img_size,
    angles=angles,
    circle=False,
    device=device,
    noise_model=dinv.physics.PoissonNoise(gain=noise_level_img,
                                          clip_positive=True),
)

PI = 4 * torch.ones(1).atan()
SCALING = (PI / (2 * angles)).to(device)  # approximate operator norm of A^T A

mu = 1 / 50.0 * (362.0 / img_size)
N0 = 1024.0


# %%
# Define the likelihood
# --------------------------------------------------------------
#
# Since the noise model is Poisson.

# load Poisson Likelihood
#likelihood = dinv.optim.LogPoissonLikelihood(mu=mu, N0=N0)
likelihood = dinv.optim.PoissonLikelihood(gain=noise_level_img)#, bkg=1e-8)
#likelihood = dinv.optim.data_fidelity.L2()

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
    #  denoiser=dinv.models.GSDRUNet(in_channels=n_channels,
    #  out_channels=n_channels, pretrained="download"),
    denoiser=dinv.models.DnCNN(in_channels=n_channels,
    out_channels=n_channels, pretrained="download_lipschitz")
 ).to(device)


# class GSPnP(dinv.optim.prior.RED):
#     r"""
#     Gradient-Step Denoiser prior.
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.explicit_prior = True

#     def forward(self, x, *args, **kwargs):
#         r"""
#         Computes the prior :math:`g(x)`.

#         :param torch.tensor x: Variable :math:`x` at which the prior is computed.
#         :return: (torch.tensor) prior :math:`g(x)`.
#         """
#         return self.denoiser.potential(x, *args, **kwargs)
# pretrained_path = "..\\..\\..\\bregman_sampling\\BregmanPnP\\GS_denoising\\ckpts\\Prox-DRUNet.ckpt"
# prior = GSPnP(denoiser=dinv.models.GSDRUNet(in_channels=n_channels,
#           out_channels=n_channels, 
#                 pretrained="download").to(device))


#prior = dinv.optim.ScorePrior(dinv.models.TVDenoiser(n_it_max=100)).to(device)

# %%
# Create the MCMC sampler
# --------------------------------------------------------------
#
# Here we use the Mirror Langevin Algorithm (MLA) to sample from the posterior defined in
# :class:`deepinv.sampling.MLAIterator`.
# The hyperparameter ``step_size`` controls the step size of the MCMC sampler,
# ``regularization`` controls the strength of the prior and
# ``iterations`` controls the number of iterations of the sampler.
y = physics(x).clamp(min=1e-3)


def call(statistics, iter, **kwargs):
    #print(statistics[0].mean().min(), statistics[0].mean().max())
    print("Nans in statistics: ", torch.isnan(statistics[0].mean()).any(), iter)
    print(f"PSNR: {dinv.metric.PSNR()(x, statistics[0].mean()).item():.2f} dB")

regularization = noise_level_img * 20
step_size = 0.01 * SCALING

iterations = int(1000) if torch.cuda.is_available() else 100


params = {
    "step_size": step_size,
    "alpha": regularization,
    "sigma": sigma_denoiser,
    "inner_iter": 10,
    "eta": 0.05,
}
f = dinv.sampling.sampling_builder(
    "MLA",
    prior=prior,
    data_fidelity=likelihood,
    max_iter=iterations,
    params_algo=params,
    callback=call,
    burnin_ratio=0,
    thinning=10,
    clip=(1e-4, None),
    verbose=True,
)

# %%
# Generate the measurement
# --------------------------------------------------------------
# We apply the forward model to generate the noisy measurement.


# compute linear inverse
x_lin = physics.A_dagger(y)

temp = likelihood.grad(x_lin, y, physics)  # compute gradient of the likelihood at x_lin
print(temp)
# %%
# Run sampling algorithm and plot results
# --------------------------------------------------------------
# The sampling algorithm returns the posterior mean and variance.
# We compare the posterior mean with a simple linear reconstruction.

mean, var = f.sample(y, physics, 
                     x_init=physics.A_dagger(y).clamp(min=1e-3))

# compute PSNR
print(f"Linear reconstruction PSNR: {dinv.metric.PSNR()(x, x_lin).item():.2f} dB")
print(f"Posterior mean PSNR: {dinv.metric.PSNR()(x, mean).item():.2f} dB")

#%%
# plot results
error = (mean - x).abs().sum(dim=1).unsqueeze(1)  # per pixel average abs. error
std = var.sum(dim=1).unsqueeze(1).sqrt()  # per pixel average standard dev.
imgs = [y, x_lin, x, mean, std, error ]
plot(
    imgs,
    titles=["measurement", "lin recon", "ground truth", "post. mean", "post. std", "abs. error"],
)

# %%
