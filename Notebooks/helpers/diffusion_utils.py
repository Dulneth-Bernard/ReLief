import torch
import torch.nn.functional as F
import config

# Pre-calculate schedule on import
betas = torch.linspace(0.0001, 0.02, config.TIMESTEPS).to(config.DEVICE)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)

def forward_diffusion_sample(x_0, t, device):
    """ Adds noise to image x_0 at timestep t """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None]
    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

def get_loss(model, x_0, t, device):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)