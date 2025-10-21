import torch

class NoiseScheduler:
    def __init__(self, timesteps = 1000, beta_start = 0.0001, beta_end = 0.02):
        self.timestep = timesteps
        self.betas = torch.linspace(beta_start, beta_end,  timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cum_pred = torch.cumprod(self.alphas, axis = 0) # How much of the original image is still left
        self.sqrt_alphas_cum_pred = torch.sqrt(self.alphas_cum_pred) # Square root of cumulative alpha, used to scale the original imag
        self.sqrt_one_minus_alphas = torch.sqrt(1.0 - self.alphas_cum_pred)# Square root of 1 - cumulative alpha, used to scale the noise



    def add_noise(self, images_orig, noise_in, time_step):
        s_acp_t = self.sqrt_alphas_cum_prod[time_step].to(images_orig.device)
        s_om_acp_t = self.sqrt_one_minus_alphas_cum_prod[time_step].to(images_orig.device)
        
        s_acp_t = s_acp_t.view(-1, 1, 1, 1)
        s_om_acp_t = s_om_acp_t.view(-1, 1, 1, 1)


        return s_acp_t * images_orig + s_om_acp_t * noise_in

