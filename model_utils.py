import torch

# # DDPM
# T = 1000  # diffusion steps
# beta_start = 1e-4
# beta_end = 0.02
# betas = torch.linspace(beta_start, beta_end, T)

# alphas = 1.0 - betas
# alphas_cumprod = torch.cumprod(alphas, dim=0)

# def q_sample(x0, t, noise=None):
#     """Add noise to the clean image x0 at step t"""
#     if noise is None:
#         noise = torch.randn_like(x0)
#     sqrt_alpha_cumprod = alphas_cumprod[t].sqrt().to(x0.device)
#     sqrt_one_minus_alpha_cumprod = (1 - alphas_cumprod[t]).sqrt().to(x0.device)
#     return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise


class LinearNoiseSchedule:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise):
        sqrt_alpha_cumprod = self.alphas_cumprod[t].sqrt().unsqueeze(-1)
        sqrt_one_minus_alpha = (1 - self.alphas_cumprod[t]).sqrt().unsqueeze(-1)
        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha * noise

    def predict_x0(self, xt, t, noise_pred):
        sqrt_alpha_cumprod = self.alphas_cumprod[t].sqrt().unsqueeze(-1)
        sqrt_one_minus_alpha = (1 - self.alphas_cumprod[t]).sqrt().unsqueeze(-1)
        return (xt - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha_cumprod
    
    def p_sample(self, xt, t, noise_pred, guidance_grad=None):
        if t == 0:
            return self.predict_x0(xt, t, noise_pred)
        beta_t = self.betas[t].unsqueeze(-1)
        alpha_t = self.alphas[t].unsqueeze(-1)
        pred_x0 = self.predict_x0(xt, t, noise_pred)
        mean = alpha_t.sqrt() * pred_x0 + (1 - alpha_t).sqrt() * noise_pred
        if guidance_grad is not None:
            mean = mean + beta_t * guidance_grad
        noise = torch.randn_like(xt)
        return mean + beta_t.sqrt() * noise

    # def p_sample(self, xt, t, noise_pred, guidance_grad=None):
    #     if t == 0:
    #         return self.predict_x0(xt, t, noise_pred)
    #     beta_t = self.betas[t].unsqueeze(-1)
    #     alpha_t = self.alphas[t].unsqueeze(-1)
    #     alpha_bar_t = self.alphas_cumprod[t].unsqueeze(-1)
    #     pred_x0 = self.predict_x0(xt, t, noise_pred)
    #     mean = alpha_t.sqrt() * pred_x0 + (1 - alpha_t).sqrt() * noise_pred
    #     mean = (1 / alpha_t.sqrt()) * (xt - beta_t / (1 - alpha_bar_t).sqrt() * guided_noise_pred) # alt
    #     if guidance_grad is not None:
    #         mean = mean + beta_t * guidance_grad
    #     noise = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)
    #     return mean + beta_t.sqrt() * noise