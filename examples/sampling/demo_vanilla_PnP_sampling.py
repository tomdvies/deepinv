"""
Vanilla PnP for computed tomography (CT).
====================================================================================================

This example shows how to use a standart PnP algorithm with DnCNN denoiser for computed tomography.

"""

import deepinv as dinv
from pathlib import Path
import torch
from deepinv.models import DnCNN
from deepinv.optim.data_fidelity import L2
from deepinv.utils.demo import load_url_image, get_image_url
from deepinv.utils.plotting import plot
import wandb
import random

import os
os.environ["WANDB_NOTEBOOK_NAME"] = "demo_vanilla_PnP_sampling.py"

# %%
# Setup paths for data loading and results.
# ----------------------------------------------------------------------------------------
#
BASE_DIR = Path(".")
RESULTS_DIR = BASE_DIR / "results"

# %%
# Load image and parameters
# ----------------------------------------------------------------------------------------

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
# Set up the variable to fetch dataset and operators.
method = "PnP"
img_size = 64
x = load_url_image(
    get_image_url("SheppLogan.png"),
    img_size=img_size,
    grayscale=True,
    resize_mode="resize",
    device=device,
)
operation = "tomography"


# %%
# Set the forward operator
# --------------------------------------------------------------------------------
# We use the :class:`deepinv.physics.Tomography`
# class from the physics module to generate a CT measurements.


noise_level_img = 0.03  # Gaussian Noise standard deviation for the degradation
angles = 100
n_channels = 1  # 3 for color images, 1 for gray-scale images
physics = dinv.physics.Tomography(
    img_width=img_size,
    angles=angles,
    circle=False,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
)

PI = 4 * torch.ones(1).atan()
SCALING = (PI / (2 * angles)).to(device)  # approximate operator norm of A^T A

# Use parallel dataloader if using a GPU to fasten training,
# otherwise, as all computes are on CPU, use synchronous data loading.
num_workers = 4 if torch.cuda.is_available() else 0


# %%
# Set up the PnP algorithm to solve the inverse problem.
# --------------------------------------------------------------------------------
# We use DNCNN pretrained denoiser :class:`deepinv.models.DnCNN`.
#
# Set up the PnP algorithm parameters : the ``stepsize``, ``g_param`` the noise level of the denoiser.
# Attention: The choice of the stepsize is crucial as it also defines the amount of regularization.  Indeed, the regularization parameter ``lambda`` is implicitly defined by the stepsize.
# Both the stepsize and the noise level of the denoiser control the regularization power and should be tuned to the specific problem.
# The following parameters have been chosen manually.

# Select the data fidelity term
data_fidelity = L2()

# Specify the denoising prior
denoiser = DnCNN(
    in_channels=n_channels,
    out_channels=n_channels,
    pretrained="download_lipschitz",  # automatically downloads the pretrained weights, set to a path to use custom weights.
    device=device,
)
prior = dinv.optim.ScorePrior(denoiser=denoiser)


# %%
# Evaluate the model on the problem and plot the results.
# --------------------------------------------------------------------
#
# The model returns the output and the metrics computed along the iterations.
# For computing PSNR, the ground truth image ``x_gt`` must be provided.

y = physics(x)
x_lin = (
    physics.A_adjoint(y) * SCALING
)  # rescaled linear reconstruction with the adjoint operator


#%%
iterations = 100000
sigma_denoiser = 2 / 255
thin = 10000
algo = "MLA"
#algo = "SKROCK"

params = {
    "step_size": 0.01 * SCALING,
    "alpha": noise_level_img * 20,
    "sigma": sigma_denoiser,
    "inner_iter": 10,
    "eta": 0.05,
    "iterations": iterations,
    "thin": thin,
    "algo": algo,
}

def call(statistics, iter, **kwargs):
    psnr_log = dinv.metric.PSNR()(x, statistics[0].mean()).item()
    print(f"PSNR: {psnr_log:.2f} dB")
    wandb.log({"PSNR" : psnr_log,
               "Variance" : wandb.Image(statistics[0].var().cpu().squeeze(),
                                        caption="Variance"),
               "Posterior mean" : wandb.Image(statistics[0].mean().cpu().squeeze(),
                                        caption="Mean")},
               step=iter+1)


f = dinv.sampling.sampling_builder(
    algo,
    prior=prior,
    data_fidelity=data_fidelity,
    max_iter=iterations,
    params_algo=params,
    callback=call,
    burnin_ratio=0.1,
    thinning=thin,
    verbose=True,
)

#%% init wandb
project = "mla_ct_shepp"
counter = random.randint(0, 1000)
exp_name = "shepp_logan_gauss_" + str(counter)
wandb.init(entity='bloom', project="tk_"+ project, 
           name = exp_name , config=params, save_code=True)

#%% log measurement
wandb.log({"Observation" : wandb.Image((y/torch.max(y)).cpu().squeeze(), caption="Observation")})

#%% run sampling
mean, var = f.sample(y, physics, 
                     x_init=physics.A_dagger(y))

# compute PSNR
print(f"Linear reconstruction PSNR: {dinv.metric.PSNR()(x, physics.A_dagger(y)).item():.2f} dB")
print(f"Posterior mean PSNR: {dinv.metric.PSNR()(x, mean).item():.2f} dB")

#%%
# plot results
error = (mean - x).abs().sum(dim=1).unsqueeze(1)  # per pixel average abs. error
std = var.sum(dim=1).unsqueeze(1).sqrt()  # per pixel average standard dev.
imgs = [y, physics.A_dagger(y), x, mean, std, error ]
plot(
    imgs,
    titles=["measurement", "lin recon", "ground truth", "post. mean", "post. std", "abs. error"],
)

# %%
wandb.finish()
