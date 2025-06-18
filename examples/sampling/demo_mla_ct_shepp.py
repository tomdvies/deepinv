import deepinv as dinv
from deepinv.utils.plotting import plot
import torch
from deepinv.utils.demo import get_image_url, load_url_image
import wandb

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
mu = 1 / 50.0 * (362.0 / img_size)
N0 = 1024.0

noise_level_img = 1/20#0.03
angles = 100
n_channels = 1  # 3 for color images, 1 for gray-scale images
physics = dinv.physics.Tomography(
    img_width=img_size,
    angles=angles,
    circle=False,
    device=device,
    noise_model=dinv.physics.PoissonNoise(gain=noise_level_img,)
                                        #   clip_positive=True),
    #noise_model=dinv.physics.LogPoissonNoise(mu=mu, N0=N0)
)

PI = 4 * torch.ones(1).atan()
SCALING = (PI / (2 * angles)).to(device)  # approximate operator norm of A^T A




# %%
# Define the likelihood
# --------------------------------------------------------------
#
# Since the noise model is Poisson.

# load Poisson Likelihood
likelihood = dinv.optim.LogPoissonLikelihood(mu=mu, N0=N0)
#likelihood = dinv.optim.PoissonLikelihood(gain=noise_level_img, bkg=1e-8)
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
    psnr_log = dinv.metric.PSNR()(x, statistics[0].mean()).item()
    print(f"PSNR: {psnr_log:.2f} dB")
    wandb.log({"PSNR" : psnr_log,
               "Variance" : wandb.Image(torch.sqrt((statistics[0].var())).cpu().squeeze(), caption="Variance"),
               "Posterior mean" : wandb.Image(statistics[0].mean().cpu().squeeze(), caption="Mean")},
               step=iter+1)


regularization = noise_level_img * 100
step_size = 0.01 * SCALING

iterations = int(50000) if torch.cuda.is_available() else 100
thin = 1000
algo = "MLA"

params = {
    "step_size": step_size,
    "alpha": regularization,
    "sigma": sigma_denoiser,
    "inner_iter": 10,
    "eta": 0.05,
    "iterations": iterations,
    "thinning": thin,  
    "algo": algo,
    "noise_level": noise_level_img,
}

project = "mla_ct_shepp"
exp_name = "shepp_logan"
wandb.init(entity='bloom', project="tk_"+ project, name = exp_name , config=params, save_code=True)
config = wandb.config

f = dinv.sampling.sampling_builder(
    algo,
    prior=prior,
    data_fidelity=likelihood,
    max_iter=iterations,
    params_algo=params,
    callback=call,
    burnin_ratio=0,
    thinning=thin,
    clip=(1e-4, None),
    verbose=True,
)

# %%
# Generate the measurement
# --------------------------------------------------------------
# We apply the forward model to generate the noisy measurement.


# compute linear inverse
x_lin = physics.A_dagger(y)

wandb.log({"Observation" : wandb.Image((y/torch.max(y)).cpu().squeeze(), caption="Observation")})
wandb.log({"Ground truth" : wandb.Image((x).cpu().squeeze(), caption="Ground truth")})

temp = likelihood.grad(x_lin, y, physics)  # compute gradient of the likelihood at x_lin
print(temp)
# %%
# Run sampling algorithm and plot results
# --------------------------------------------------------------
# The sampling algorithm returns the posterior mean and variance.
# We compare the posterior mean with a simple linear reconstruction.
init_ = ((physics.A_adjoint(y)-physics.A_adjoint(y).min())/physics.A_adjoint(y).max())

mean, var = f.sample(y, physics, 
                     x_init=x_lin.clamp(min=1e-4)) 

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
wandb.finish()