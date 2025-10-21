from dataclasses import dataclass
import torch

@dataclass
class Configuration:
    image_size = 64
    batch_size = 64
    epochs = 50
    learning_rate = 0.0001
    timesteps = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    channels = 3
    feature_sizes = (64, 128, 256, 512)
    save_interval = 5
    num_samples = 8
    data_path = '/home/wizard/python/256/'
    model_path = 'face_diffusion_model.pth'
    sample_format = 'face_samples_epoch_{}.png'