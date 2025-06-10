import deepinv as dinv
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
# This example uses the CT operator 
# and Poisson noise as the noise model.

noise_level_img = 1/20
angles = 100
n_channels = 1  
physics = dinv.physics.Tomography(
    img_width=img_size,
    angles=angles,
    circle=False,
    device=device,
    noise_model=dinv.physics.PoissonNoise(gain=noise_level_img)
)

# %%
# Define the Poisson likelihood
# --------------------------------------------------------------
#
# Since the noise model is Poisson.

likelihood = dinv.optim.PoissonLikelihood(gain=noise_level_img)#, bkg=1e-8)

# %%
# Generate the measurement
# --------------------------------------------------------------
# We apply the forward model to generate the noisy measurement.

y = physics(x).clamp(min=1e-3)  

# compute linear inverse
x_lin = physics.A_dagger(y).clamp(min=1e-3)

#compute gradient of likelihood given the data
grad_val = likelihood.grad(x_lin, y, physics)  
print(grad_val)



# %%
