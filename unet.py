import torch
import torch.nn as nn
class Block(nn.Modul):
    def __init__(self,in_channels,out_channels,time_dim = None):
        super().__init__()
        self.conv_layers = nn.Sequential( # for start the layers 
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(), 
        )

        if time_dim: self.time_mlp = nn.Linear(time_dim, out_channels) # add time dimention
    def forward(self, x, time_emb=None):
        x = self.conv_layers(x) 
        if time_emb is not None and hasattr(self, 'time_mlp'):
            x = x + self.time_mlp(time_emb)[:, :, None, None]
        return x
    
class SinEmbedding(nn.Module):
    def __init__(self, dimention):
        super().__init__()
        self.dimention = dimention

    def forward(self, time):
        device = time.device
        half_dim = self.dimension // 2 # splitting the embedding for sin and cos
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
         # log makes the frequencies logarithmic instead of linear
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings) 
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class UNetModel (nn.Module):
    def __init__(self, in_channels = 3, out_channels = 3 , feature_sizes = (64,128,256), time_dim = 64):
        super().__init__()
        self.time_mlp = nn.Sequential(SinEmbedding(time_dim),nn.Linear(time_dim,time_dim),nn.Relu())
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.max_pool = nn.MaxPool2d(2,2)# keep important parts, remove small/empty parts

        current_in_channels = in_channels
        for feature_size in feature_sizes:
            self.down_layers.append(Block(current_in_channels,feature_size , time_dim))
            current_in_channels = feature_sizes

        self.bottleneck = Block(feature_sizes[-1], feature_sizes[-1]* 2 , time_dim)
        # start with largest feature size in bottleneck to store complex features

        for feature_size in reversed(feature_sizes):
            self.up_layers.append(nn.ConvTranspose2d(feature_size * 2 , feature_size,2,2))
            self.up_layers.append(Block(feature_size * 2 , feature_size,time_dim))

        self.final_conv = nn.Conv2d(feature_sizes[0], out_channels , 1)# convert final features to output image

    def forward(self, x, time):
        time_emb = self.time_mlp(time)
        skips = []

        for down_layer in self.down_layers:
            x = down_layer(x, time_emb)
            skips.append(x)
            x = self.max_pool(x)

        x = self.bottleneck(x, time_emb)
        skips = skips[::-1]

        for i in range(0, len(self.up_layers), 2):
            x = self.up_layers[i](x)
            skip = skips[i // 2]
            if x.shape != skip.shape:
                skip = nn.functional.interpolate(skip, size=x.shape[2:])
            x = self.up_layers[i + 1](torch.cat((skip, x), dim=1), time_emb)

        return self.final_conv(x)