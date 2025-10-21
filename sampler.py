import torch

class Sampler:
    def __init__(self, noise_scheduler, noise_model):
        self.noise_scheduler = noise_scheduler
        self.noise_model = noise_model

    @torch.no_grad()
    def sample_images(self, shape, device):
        x = torch.randn(shape, device=device)
        for t_idx in reversed(range(self.noise_scheduler.timesteps)):
            t_tensor = torch.tensor([t_idx], device=device, dtype=torch.long)
            pred_noise = self.noise_model(x, t_tensor)

            alpha_t = self.noise_scheduler.alphas[t_idx]
            alpha_cum_prod_t = self.noise_scheduler.alphas_cum_prod[t_idx]
            beta_t = self.noise_scheduler.betas[t_idx]
            
            coeff_x = 1 / torch.sqrt(alpha_t)
            coeff_noise = beta_t / torch.sqrt(1 - alpha_cum_prod_t)
            
            mean_est = coeff_x * (x - coeff_noise * pred_noise)

            if t_idx > 0:
                z = torch.randn_like(x)
                x = mean_est + torch.sqrt(beta_t) * z
            else:
                x = mean_est
        return x