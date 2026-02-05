import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
import numpy as np


def make_beta_schedule(schedule, n_timestep, beta_0=1e-4, beta_T=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = torch.linspace(beta_0 ** 0.5, beta_T ** 0.5, n_timestep).double() ** 2
    elif schedule == 'linear':
        betas = torch.linspace(beta_0, beta_T, n_timestep).double() 
    elif schedule == 'const':
        betas = beta_T * torch.ones(n_timestep).double() 
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / torch.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * torch.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


class DDPM_Trainer(nn.Module):
    def __init__(self, model, configs):
        super().__init__()
        self.model = model
        self.T = configs['T']
        self.ratio = configs['uncon_ratio']
        betas =  make_beta_schedule(configs['beta_scdl'], self.T, configs['beta_1'], configs['beta_T'])
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        # calculations for diffusion q(x_t | x_{t-1})
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar).float())
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar).float())

    def forward(self, condit, x_0):
        B = x_0.shape[0]
        t = torch.randint(0, self.T, size=(B,), device=x_0.device)
        sqrt_alphas_bar_t = torch.gather(self.sqrt_alphas_bar, 0, t).to(x_0.device)
        sqrt_one_minus_alphas_bar_t = torch.gather(self.sqrt_one_minus_alphas_bar, 0, t).to(x_0.device)
        noise = torch.randn_like(x_0)
        # add noise accord to the step t
        x_t = (sqrt_alphas_bar_t.view(-1, 1, 1, 1) * x_0 + sqrt_one_minus_alphas_bar_t.view(-1, 1, 1, 1) * noise)
        # randomly remove the condition from half of the batch, each sample has self.ratio chance to be unconditional
        rnd_cond = np.random.choice([True, False], size=B, p=[self.ratio, 1-self.ratio])
        inputs = torch.cat([condit, x_t], dim=1)
        loss = F.mse_loss(self.model(inputs, t, rnd_cond), noise, reduction='none')
        return loss
        

class DDIM_Sampler(nn.Module):
    def __init__(self, model, configs, ddim_sampling_steps=50, eta=1):
        """
        Sampling process of Denoising Diffusion Implicit Models (DDIM), Jiaming Song et al.
        """
        super().__init__()
        self.model = model
        self.T = configs['T']
        self.ddim_steps = ddim_sampling_steps
        self.eta = eta

        betas = make_beta_schedule(configs['beta_scdl'], self.T, configs['beta_1'], configs['beta_T'])
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_bar', alphas_bar)

        self.register_buffer('tau', torch.linspace(-1, self.T-1, steps=self.ddim_steps+1, dtype=torch.long)[1:])
        alpha_tau_i = alphas_bar[self.tau]
        alpha_tau_i_min_1 = F.pad(alphas_bar[self.tau[:-1]], pad=(1, 0), value=1.)  # alpha_0 = 1

        self.register_buffer('sigma', eta * (((1 - alpha_tau_i_min_1) / (1 - alpha_tau_i) *
                                            (1 - alpha_tau_i / alpha_tau_i_min_1)).sqrt()))
        self.register_buffer('coeff', (1 - alpha_tau_i_min_1 - self.sigma ** 2).sqrt())
        self.register_buffer('sqrt_alpha_i_min_1', alpha_tau_i_min_1.sqrt())
        self.register_buffer('sqrt_recip_alphas_bar', torch.sqrt(1. / self.alphas_bar))
        self.register_buffer('sqrt_recipm1_alphas_bar', torch.sqrt(1. / self.alphas_bar - 1))
        assert self.coeff[0] == 0.0 and self.sqrt_alpha_i_min_1[0] == 1.0, 'DDIM parameter error'

    def ddim_p_sample(self, condit, x_t, i, clip=False):
        t = self.tau[i]
        batched_time = torch.full((x_t.shape[0],), t, dtype=torch.long).cuda()
        pred_noise = self.model(torch.cat([condit, x_t], dim=1), batched_time) 
        x0 = self.sqrt_recip_alphas_bar[t] * x_t - self.sqrt_recipm1_alphas_bar[t] * pred_noise
        if clip:
            x0.clamp_(-1, 1)
            pred_noise = (self.sqrt_recip_alphas_bar[t] * x_t - x0) / self.sqrt_recipm1_alphas_bar[t]

        mean = self.sqrt_alpha_i_min_1[i] * x0 + self.coeff[i] * pred_noise
        noise = torch.randn_like(x_t) if i > 0 else 0.
        x_t_minus_1 = mean + self.sigma[i] * noise
        return x_t_minus_1

    @torch.no_grad()
    def forward(self, condit, x_T, save_process=False):
        x_t = x_T
        for i in tqdm(reversed(range(self.ddim_steps)), desc='DDIM Sampling', total=self.ddim_steps, position=1, leave=False):
            x_t = self.ddim_p_sample(condit, x_t, i)
            if save_process:
                torchvision.utils.save_image(x_t, 'figures/%s.png'%str(i).zfill(3), normalize=True)
        x_0 = x_t
        return x_0