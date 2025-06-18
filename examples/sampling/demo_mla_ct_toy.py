#%%
import deepinv as dinv
from deepinv.utils.plotting import plot
import torch
from deepinv.utils.demo import load_torch_url, load_dataset

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

# %%
# Load image (toy LoDoPaB-CT dataset)

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

url = "https://huggingface.co/datasets/deepinv/LoDoPaB-CT_toy/resolve/main/LoDoPaB-CT_small.pt"
dataset = load_torch_url(url)
#not working
#dataset = load_dataset("../../../../data/ct/LoDoPaB-CT_small")
train_imgs = dataset["train_imgs"].to(device)
test_imgs = dataset["test_imgs"].to(device)
img_size = train_imgs.shape[-1]


# %%
# Define forward operator and noise model
# --------------------------------------------------------------
#
# This example uses the CT operator and Log Poisson noise as the noise model.

# mu = 1 / 50.0 * (362.0 / img_size)
# N0 = 1024.0
gain = 1/20
num_angles = 100
PI = 4 * torch.ones(1).atan()
SCALING = (PI / (2 * num_angles)).to(device)  # approximate operator norm of A^T A
noise_model = dinv.physics.PoissonNoise(gain=gain, normalize=True)
data_fidelity = dinv.optim.PoissonLikelihood(gain=gain, bkg=1e-12, denormalize=False)
angles = torch.linspace(20, 160, steps=num_angles)
physics = dinv.physics.Tomography(
    img_width=img_size, angles=angles,
    device=device, noise_model=noise_model
)

observation = physics(test_imgs)

plot(physics.A_adjoint(observation))


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
n_channels = 1
prior = dinv.optim.ScorePrior(
    # denoiser=dinv.models.DRUNet()
    denoiser=dinv.models.DnCNN(in_channels=n_channels,
    out_channels=n_channels, pretrained="download")
).to(device)

# Set up the denoising prior. Note that we use a Gaussian noise denoiser, even if the observation noise is Poisson.
# prior = dinv.optim.prior.RED(
#     denoiser=dinv.models.DnCNN(
#         in_channels=n_channels,
#         out_channels=n_channels, 
#         pretrained="download_lipschitz").to(device))


# %%
# Create the MCMC sampler
# --------------------------------------------------------------
#
# Here we use the Mirror Langevin Algorithm (MLA) to sample from the posterior defined in
# :class:`deepinv.sampling.MLAIterator`.
# The hyperparameter ``step_size`` controls the step size of the MCMC sampler,
# ``regularization`` controls the strength of the prior and
# ``iterations`` controls the number of iterations of the sampler.

#define callback
def call(statistics, **kwargs):
    print(f"PSNR: {dinv.metric.PSNR()(test_imgs, statistics[0].mean()).item():.2f} dB")
        #   LPIPS: {dinv.metric.LPIPS(device=device)(test_imgs, statistics[0].mean()).item():.2f}""")
    

regularization = 1
step_size = 1e-4

regularization = gain * 10
step_size = 0.01 * SCALING

iterations = int(10000) if torch.cuda.is_available() else 100
params = {
    "step_size": step_size,
    "alpha": regularization,
    "sigma": sigma_denoiser,
}
f = dinv.sampling.sampling_builder(
    "MLA",
    prior=prior,
    data_fidelity=data_fidelity,
    max_iter=iterations,
    params_algo=params,
    callback=call,
    burnin_ratio=0.1,
    thinning=1000,
    verbose=True,
)

# %%
# Generate the measurement
# --------------------------------------------------------------
# We apply the forward model to generate the noisy measurement.


# compute linear inverse
x_lin = physics.A_dagger(observation)

# %%
# Run sampling algorithm and plot results
# --------------------------------------------------------------
# The sampling algorithm returns the posterior mean and variance.
# We compare the posterior mean with a simple linear reconstruction.

mean, var = f.sample(observation, physics, x_init=x_lin)

# compute PSNR
print(f"Linear reconstruction PSNR: {dinv.metric.PSNR()(test_imgs, x_lin).item():.2f} dB")
print(f"Posterior mean PSNR: {dinv.metric.PSNR()(test_imgs, mean).item():.2f} dB")

# plot results
error = (mean - test_imgs).abs().sum(dim=1).unsqueeze(1)  # per pixel average abs. error
std = var.sum(dim=1).unsqueeze(1).sqrt()  # per pixel average standard dev.
imgs = [observation, x_lin, test_imgs, mean, std / std.flatten().max(), error / error.flatten().max()]
plot(
    imgs,
    titles=["measurement", "lin recon", "ground truth", "post. mean", "post. std", "abs. error"],
)
