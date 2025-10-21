import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os


def get_data_loader(config):
    transformations = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    print(f"Loading images from local path: {config.data_path}")
    dataset = datasets.ImageFolder(root=config.data_path, transform=transformations)
        
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=os.cpu_count() // 2)

def save_samples(images, epoch, config):
    os.makedirs('samples', exist_ok=True)
    images = (images + 1) / 2
    save_image(images, os.path.join('samples', config.sample_format.format(epoch)), nrow=config.num_samples // 2)

def save_model(model, config):
    torch.save(model.state_dict(), config.model_path)