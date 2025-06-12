import torch

class LinearNoiseSchedule:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise):
        sqrt_alpha_cumprod = self.alphas_cumprod[t].to().sqrt().unsqueeze(-1)
        sqrt_one_minus_alpha = (1 - self.alphas_cumprod[t]).sqrt().unsqueeze(-1)
        return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha * noise

    # looks good
    def predict_x0(self, xt, t, noise_pred):
        sqrt_alpha_cumprod = self.alphas_cumprod[t].sqrt().unsqueeze(-1)
        sqrt_one_minus_alpha = (1 - self.alphas_cumprod[t]).sqrt().unsqueeze(-1)
        return (xt - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha_cumprod
 
    def p_sample_1(self, xt, t, noise_pred, guidance_grad=None, guidance_scale=1):
            t = torch.as_tensor(t, device=xt.device)
            # get schedule parameters
            beta_t = self.betas[t].unsqueeze(-1)
            alpha_t = self.alphas[t].unsqueeze(-1)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - self.alphas_cumprod[t]).unsqueeze(-1)
            # compute the mean of the reverse process
            coef1 = (1.0 - alpha_t) / sqrt_one_minus_alpha_cumprod_t
            mean = (xt - coef1 * noise_pred) / sqrt_alpha_t
            # apply guidance if provided
            if guidance_grad is not None and guidance_scale != 0:
                # print(f"guidance grad: {guidance_grad}")
                # print(f"mean before: {mean}")
                mean = mean + guidance_scale * guidance_grad
                # print(f"mean after: {mean}")
            # add noise for t > 0
            if t.max() > 0:
                # sample noise
                z = torch.randn_like(xt)
                variance = torch.sqrt(beta_t)
                mean = mean + variance * z
            
            return mean