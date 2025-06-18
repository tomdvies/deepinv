import deepinv as dinv
from deepinv.utils.plotting import plot
import torch
from deepinv.utils.demo import load_url_image
import wandb
import random
import torch.nn.functional as F

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)

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
im_px = 150  # image size in pixels
x = load_url_image(url=url, img_size=im_px).to(device)

#%% some preprocessing
def preprocess(img):
    (z, q, h) = (0.0, 0.25, 0.5)
    sparse = torch.tensor([[q, h, q],
                        [h, z, h],
                        [q, h, q]],
                        device=device).unsqueeze(0).unsqueeze(0)

    dense = torch.tensor([[z, q, z],
                        [q, z, q],
                        [z, q, z]],
                        device=device).unsqueeze(0).unsqueeze(0)
    
    img[0,0,:,:] = \
        torch.where(img[0,0,:,:] > 0.0,
        img[0,0,:,:],
        F.conv2d(img[0,0,:,:].unsqueeze(0).unsqueeze(0), sparse, padding='same')[0,0,:,:])
    img[0,1,:,:] = \
        torch.where(img[0,1,:,:] > 0.0,
        img[0,1,:,:],
        F.conv2d(img[0,1,:,:].unsqueeze(0).unsqueeze(0), dense,  padding='same')[0,0,:,:])
    img[0,2,:,:] = \
        torch.where(img[0,2,:,:] > 0.0,
        img[0,2,:,:],
        F.conv2d(img[0,2,:,:].unsqueeze(0).unsqueeze(0), sparse, padding='same')[0,0,:,:])

    img = torch.dstack((img[0,0,:,:],
                            img[0,1,:,:],
                            img[0,2,:,:]))
    
    result = torch.swapaxes(torch.swapaxes(img, 2,0), 1,2)
    print(result.shape)
    dinv.utils.plot(result)
    return result

# %%
# Define forward operator and noise model
# --------------------------------------------------------------
#
# This example uses the identity forward operator and Poisson noise as the noise model.
gain = 1/20

physics = dinv.physics.Demosaicing(
    img_size=(im_px, im_px),
    noise_model = dinv.physics.PoissonNoise(gain=gain),
    device=device)


# %%
# Define the likelihood
# --------------------------------------------------------------
#
# Since the noise model is Poisson.

mu = 1 / 50.0 * (362.0 / im_px)
N0 = 1024.0

# load Poisson Likelihood
likelihood = dinv.optim.PoissonLikelihood(gain=gain, bkg=1e-4)
#likelihood = dinv.optim.LogPoissonLikelihood(mu, N0)

#likelihood = dinv.optim.L2()

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
    #  denoiser = dinv.optim.prior.TVPrior(n_it_max=20),
    #denoiser=dinv.models.DRUNet(pretrained='download'),
    #denoiser = dinv.models.GSDRUNet(pretrained="download"),
    denoiser=dinv.models.DnCNN(in_channels=3, 
            out_channels=3, pretrained="download_lipschitz")
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


def call(statistics, iter, **kwargs):
    psnr_log = dinv.metric.PSNR()(x, statistics[0].mean()).item()
    print(f"PSNR: {psnr_log:.2f} dB; max int {statistics[0].mean().max():.2f}")
    dinv.utils.plot(statistics[0].mean())
    wandb.log({"PSNR" : psnr_log,
            "Variance" : wandb.Image((statistics[0].var()).sqrt().cpu().squeeze(),
                                    caption="Variance"),
            "Posterior mean" : wandb.Image(statistics[0].mean().cpu().squeeze(),
                                    caption="Mean")},
            step=iter+1)
    
regularization = 1
step_size = torch.tensor(1e-4).to(device)
thin = 100
iterations = int(10000) if torch.cuda.is_available() else 5000
algo = "MLA"  # or SKROCK
params = {
    "step_size": step_size,
    "alpha": regularization,
    "sigma": sigma_denoiser,
    "iterations": iterations,
    "thinning": thin,
    "inner_iter": 10,  
    "eta": 0.05,  
    "algo": algo,
}

f = dinv.sampling.sampling_builder(
    algo,
    prior=prior,
    data_fidelity=likelihood,
    max_iter=iterations,
    params_algo=params,
    callback=call,
    burnin_ratio=0.01,
    thinning=thin,
    clip=(1e-4, None),
    verbose=True,
)

# %%
# Generate the measurement
# --------------------------------------------------------------
# We apply the forward model to generate the noisy measurement.

y = physics(x).clamp(min=1e-4)

# compute linear inverse
x_lin = physics.A_adjoint(y).clamp(min=1e-4)                   

#%% init wandb
project = "mla_demosaicing"
counter = random.randint(0, 1000)
exp_name = "messi_" + str(counter)
use_wandb = True
if use_wandb:
    wandb.init(entity='bloom', project="tk_"+ project, 
            name = exp_name , config=params, save_code=True)
else:
    wandb.init(mode="disabled")
# #%% log measurement
wandb.log({"Observation" : wandb.Image((y/torch.max(y)).cpu().squeeze(), caption="Observation")})
# log ground truth
wandb.log({"Ground truth" : wandb.Image((x).cpu().squeeze(), caption="Ground truth")})

# %%
# Run sampling algorithm and plot results
# --------------------------------------------------------------
# The sampling algorithm returns the posterior mean and variance.
# We compare the posterior mean with a simple linear reconstruction.

mean, var = f.sample(y, physics, x_init=preprocess(physics.A_adjoint(y)).clamp(min=1e-4))

# compute PSNR
print(f"Linear reconstruction PSNR: {dinv.metric.PSNR()(x, x_lin).item():.2f} dB")
print(f"Posterior mean PSNR: {dinv.metric.PSNR()(x, mean).item():.2f} dB")

# plot results
error = (mean - x).abs()  # per pixel average abs. error
std = var.sqrt()  # per pixel average standard dev.
imgs = [x_lin, x, mean, std / std.flatten().max(), error / error.flatten().max()]
plot(
    imgs,
    titles=["measurement", "ground truth", "post. mean", "post. std", "abs. error"],
)

wandb.finish()