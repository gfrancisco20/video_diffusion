# -*- coding: utf-8 -*-
"""
Conditional Denoising Diffusion and V-Diffusion class for multi-channel videos of shape [img_channel, num_frames, img_size, img_size]
These classes implement recent ameliorations of DDPM-based generative models and support conditional generation.
"""

import torch
from tqdm import tqdm
import logging
import numpy as np

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self,  
                 noise_steps=1000,  # Total number of diffusion timesteps
                 beta_start=1e-4,  # Starting beta value for linear schedule
                 beta_end=0.02,  # Ending beta value for linear schedule
                 s = 0.1,  # Parameter for cosine schedule
                 img_size=256,  # Image size (assuming square images)
                 img_channel=1,  # Number of channels in the image (e.g., grayscale=1, RGB=3)
                 device="cuda",  # Device to use ('cuda' for GPU or 'cpu')
                 num_frames=6,  # Number of frames for video data
                 schedule="linear"  # Noise schedule to use ('linear' or 'cosine')
                 ):
        """
        Diffusion class implements the forward and reverse diffusion processes for conditional generative modeling of videos.
        This class supports both linear and cosine noise schedules and offers sampling using DDPM and DDIM approaches.
        It also implements fast-diffusion (sample_timesteps_fast method) that allow to improve DDIM sampling quality and reduce training time.
        """
        self.noise_steps = noise_steps # timestesps
        self.img_channel = img_channel
        self.img_size = img_size
        self.device = device
        self.num_frames = num_frames
        self.reverse_steps = None

        self.schedule = schedule
        if schedule == "linear":
          self.beta_start = beta_start
          self.beta_end = beta_end
          self.beta = self.prepare_linear_schedule().to(device)
        elif schedule == 'cosine':
          self.s = s
          self.beta = self.prepare_cosine_schedule(s).to(device)
        else:
          raise Exception(f"UNDEFINED SCHEDULER : Choose implemented scceduler  : ['linear', 'cosine]")
        self.alpha = 1. - self.beta
        self.alphas_prev = torch.cat([torch.tensor([1.0]).to(device), self.alpha[:-1]], dim=0)
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alpha_hat[:-1]], dim=0)
        # self.alphas_cumprod_prev = torch.from_numpy(np.append(1, self.alpha_hat[:-1].cpu().numpy())).to(device)
    
    def prepare_cosine_schedule(self, s=0.1):
        """
        Cosine variance schedule proposed by Nichol and Dhariwal (2021) https://doi.org/10.48550/arXiv.2102.09672
        """
        steps = self.noise_steps
        t = torch.linspace(0, steps, steps + 1, dtype=torch.float32)
        f = torch.cos(((t / steps) + s) / (1 + s) * np.pi / 2) ** 2
        alpha_cumprod = f / f[0]  # Normalize to start at 1
        beta = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
        beta = torch.clamp(beta, min=1e-5, max=0.9999)  # Clamp values to avoid extreme betas
        return beta
    
    def prepare_linear_schedule(self):
        """
          Original liine variance schedule proposed in by Ho et al. (2020) https://doi.org/10.48550/arXiv.2006.11239
        """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps) # linear variance schedule as proposed by Ho et al 2020

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ # equation in the paper from Ho et al that describes the noise processs

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
      
    def sample_timesteps_fast(self, n, substeps):
        """
        Fast-DDPM implementation : Jiang et al. 2024 https://doi.org/10.48550/arXiv.2405.14802
        to be only used with DDIM sampling with ddim_steps=substeps 
        """
        step_intervalls = self.noise_steps // self.substeps
        return torch.randint(low=1, high=substeps, size=(n,)) * step_intervalls
      
    def sample_ddpm(self, model, n, y, labels, cfg_scale=3):
        with torch.no_grad(): # algorithm 2 from DDPM
            x = torch.randn((n, self.img_channel, self.num_frames, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0): # reverse loop from T to 1
                t = (torch.ones(n) * i).long().to(self.device) # create timesteps tensor of length n
                predicted_noise = model(x, y, labels, t)
                if cfg_scale > 0 and labels is not None:
                    uncond_predicted_noise = model(x, y, None, t)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                alpha = self.alpha[t][:, None, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None, None] # this is noise, created in one
                beta = self.beta[t][:, None, None, None, None]
                # SAMPLING adjusted from Stable diffusion

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        return x
      
    def sample_ddim(self, model, n, y, labels, cfg_scale=3, eta=1, ddim_steps = 100):
        logging.info(f"Sampling {n} new images....")
        if self.reverse_steps is None:
          self.reverse_steps = ddim_steps
          self.reverse_intervalls = self.noise_steps // self.reverse_steps
          # overwriting noise-steps if not perfect multtiple of reverse_steps:
          if self.noise_steps != self.reverse_steps * self.reverse_intervalls:
              self.noise_steps = self.reverse_steps * self.reverse_intervalls
              print(f'WARNING : noise_steps not divisible by reverse_steps. noise_steps changed to {self.noise_steps}')
        with torch.no_grad(): # algorithm 2 from DDPM
            x = torch.randn((n, self.img_channel, self.num_frames, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.reverse_steps)), position=0): # reverse loop from T to 1
                t = (torch.ones(n) * i * self.reverse_intervalls).long().to(self.device) # create timesteps tensor of length n
                t_prev = (torch.ones(n) * (i-1) * self.reverse_intervalls).long().to(self.device)
                predicted_noise = model(x, y, labels, t)
                if cfg_scale > 0 and labels is not None:
                    uncond_predicted_noise = model(x, y, None, t)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)

                alpha_hat = self.alpha_hat[t][:, None, None, None, None] # this is noise, created in one
                alpha_hat_prev = self.alpha_hat[t_prev][:, None, None, None, None]

                sqrt_alpha_hat = torch.sqrt(alpha_hat)
                sqrt_alpha_hat_prev = torch.sqrt(alpha_hat_prev)

                sigma  = torch.sqrt(1 - alpha_hat)
                sigma_prev = torch.sqrt(1 - alpha_hat_prev)

                x = (sqrt_alpha_hat_prev / sqrt_alpha_hat) * x  + (sigma_prev - (sqrt_alpha_hat_prev / sqrt_alpha_hat) * sigma) * predicted_noise
                if eta != 0:
                    noise = torch.randn_like(x)  # Add stochasticity
                    nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                    x += eta * nonzero_mask * sigma_prev * noise
        return x 

    def sample(self, model, y, labels, cfg_scale=3, eta=1, sampling_mode='ddpm', ddim_steps = 100):
        n = y.shape[0]
        logging.info(f"Sampling {n} new images....")
        model.eval() # evaluation mode
        if sampling_mode == 'ddpm':
            x = self.sample_ddpm(model, n, y, labels, cfg_scale)
        elif sampling_mode == 'ddim':
            x = self.sample_ddim(model, n, y, labels, cfg_scale, eta, ddim_steps)
        else:
            raise Exception('The sampler {} is not implemented'.format(sampling_mode))
            
        model.train() 
        return x


class VDiffusion:
    """
    VDiffusion class implements the reverse diffusion process using v-prediction models, 
    with support for rescaled linear schedules and enforcing zero terminal SNR.
    It is based on the flawless rescaled linear schedule from (Lin et al., 2023 ;  https://doi.org/10.48550/arXiv.2305.08891) and 
    v-prediction from (Salimans and Ho, 2022 : https://doi.org/10.48550/arXiv.2202.00512).
    """
    def __init__(self, 
                 noise_steps=1000,  # Number of timesteps in the diffusion process
                 beta_start=1e-4,  # Starting beta value for linear schedule
                 beta_end=0.02,  # Ending beta value for linear schedule
                 img_size=256,  # Size of the image (assumed square images)
                 img_channel=1,  # Number of channels in the image (grayscale=1, RGB=3)
                 device="cuda",  # Device to run the process ('cuda' for GPU or 'cpu')
                 num_frames=6):  # Number of frames for video data
        self.noise_steps = noise_steps # timestesps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_channel = img_channel
        self.img_size = img_size
        self.device = device
        self.num_frames = num_frames

        self.beta = self.enforce_zero_terminal_snr(self.prepare_noise_schedule().to(device))
        self.alpha = 1. - self.beta
        self.alphas_prev = torch.cat([torch.tensor([1.0]).to(device), self.alpha[:-1]], dim=0)
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alpha_hat[:-1]], dim=0)

        self.posterior_variance = self.beta * (1. - self.alphas_cumprod_prev) / (1. - self.alpha_hat)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

        self.posterior_mean_coef1 = (self.beta * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alpha_hat))
        self.posterior_mean_coef2 = (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alpha) / (1 - self.alpha_hat)

        # self.alphas_cumprod_prev = torch.from_numpy(np.append(1, self.alpha_hat[:-1].cpu().numpy())).to(device)
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps) # linear variance schedule as proposed by Ho et al 2020

    def enforce_zero_terminal_snr(self, betas):
        # Convert betas to alphas_hat_sqrt
        alphas = 1 - betas
        alphas_hat = torch.cumprod(alphas, dim=0)
        alphas_hat_sqrt = torch.sqrt(alphas_hat)

        # Store old values.
        alphas_hat_sqrt_0 = alphas_hat_sqrt[0].clone()
        alphas_hat_sqrt_T = alphas_hat_sqrt[-1].clone()
        # Shift so last timestep is zero.
        alphas_hat_sqrt -= alphas_hat_sqrt_T
        # Scale so first timestep is back to old value.
        alphas_hat_sqrt *= alphas_hat_sqrt_0 / (
        alphas_hat_sqrt_0 - alphas_hat_sqrt_T)

        # Convert alphas_hat_sqrt to betas
        alphas_hat = alphas_hat_sqrt ** 2
        alphas = alphas_hat[1:] / alphas_hat[:-1]
        alphas = torch.cat([alphas_hat[0:1], alphas])
        betas = 1 - alphas
        return betas

    def v_images(self, x0, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None, None]
        Ɛ = torch.randn_like(x0)
        return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * Ɛ, sqrt_one_minus_alpha_hat * x0 - sqrt_alpha_hat * Ɛ # (xt, vt) from https://arxiv.org/abs/2202.00512

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, y, labels, cfg_scale=3, eta=1, sampling_mode='ddpm', sample_steps=None):
        n = y.shape[0]
        if sample_steps is None:
            sample_steps = self.noise_steps
        logging.info(f"Sampling {n} new images....")
        model.eval() # evaluation mode
        with torch.no_grad(): # algorithm 2 from DDPM
            x = torch.randn((n, self.img_channel, self.num_frames, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(0, sample_steps)), position=0): # reverse loop from T to 1
                t = (torch.ones(n) * i).long().to(self.device) # create timesteps tensor of length n
                v_pred = model(x, y, labels, t)
                if cfg_scale > 0 and labels is not None:
                    uncond_v = model(x, y, None, t)
                    v_pred = torch.lerp(uncond_v, v_pred, cfg_scale)

                alpha = self.alpha[t][:, None, None, None, None]
                # alpha_hat = self.alpha_hat[t][:, None, None, None, None] # this is noise, created in one
                # alpha_prev = self.alphas_cumprod_prev[t][:, None, None, None, None]
                # beta = self.beta[t][:, None, None, None, None]

                # posterior_variance = self.posterior_variance[t][:, None, None, None, None]
                log_var = self.posterior_log_variance_clipped[t][:, None, None, None, None]
                posterior_mean_coef1 = self.posterior_mean_coef1[t][:, None, None, None, None]
                posterior_mean_coef2 = self.posterior_mean_coef2[t][:, None, None, None, None]

                if sampling_mode == 'ddpm':
                    if i > 1:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)
                    # x_hat = alpha * x - torch.sqrt(beta) * v_pred
                    x_hat = alpha * x - torch.exp(0.5 * log_var) * v_pred
                    if i > 0:
                        x = posterior_mean_coef1 * (x_hat) + posterior_mean_coef2 * x + torch.exp(0.5 * log_var) * noise
                    x = x_hat
                else:
                    print('The sampler {} is not implemented'.format(sampling_mode))
                    break
        model.train()
        return x