import torch
import torch.nn as nn
import torch.optim as optim
from config import Configuration
from noise_scheduler import NoiseScheduler
from unet import UNetModel
from sampler import Sampler
from utils import get_data_loader, save_samples, save_model


def main():
    config = Configuration()
    data_loader = get_data_loader(config)

    noise_scheduler = NoiseScheduler(timesteps=config.timesteps)
    noise_model = UNetModel(in_channels=config.channels, out_channels=config.channels, feature_sizes=config.feature_sizes).to(config.device)
    sampler_inst = Sampler(noise_scheduler, noise_model)

    optimizer = optim.Adam(noise_model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    print(f"Starting training on {config.device}...")
    for epoch_idx in range(config.epochs):
        for batch_idx, (images, _) in enumerate(data_loader):
            images = images.to(config.device)
            time_steps = torch.randint(0, config.timesteps, (images.shape[0],), device=config.device).long()
            rand_noise = torch.randn_like(images).to(config.device)
            noisy_imgs = noise_scheduler.add_noise(images, rand_noise, time_steps)
            pred_noise = noise_model(noisy_imgs, time_steps)
            loss = criterion(pred_noise, rand_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch[{epoch_idx+1}/{config.epochs}], Step[{batch_idx+1}/{len(data_loader)}], Loss:{loss.item():.4f}")

        if (epoch_idx + 1) % config.save_interval == 0 or epoch_idx == config.epochs -1:
            print(f"Epoch {epoch_idx+1} finished. Generating samples...")
            sampled_imgs = sampler_inst.sample_images((config.num_samples, config.channels, config.image_size, config.image_size), config.device)
            save_samples(sampled_imgs, epoch_idx + 1, config)
            print(f"Samples saved.")

    print("Training finished!")
    save_model(noise_model, config)
    print("Model saved.")

if __name__ == '__main__':
    main()