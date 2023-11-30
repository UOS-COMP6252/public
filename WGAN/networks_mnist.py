import torch
import torch.nn as nn

class TBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size,stride,pad,norm_type= None):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch,out_ch,kernel_size,stride, pad,bias=False),
            norm_layer(out_ch,norm_type),
            nn.ReLU()
        )
    def forward(self,x):
        return self.net(x)
class CBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size,stride,pad,norm_type=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size,stride, pad, bias=False),
            norm_layer(out_ch,norm_type),
            nn.LeakyReLU(0.2)
        )
    def forward(self,x):
        return self.net(x)   
    
class Generator(nn.Module):
    #Outputs 64x64 pixel images

    def __init__(
        self,
        z_dim=100,
        out_ch=3,norm_type=None,
        final_activation=None
    ):
        super().__init__()
        self.z_dim = z_dim
        self.out_ch = out_ch
        self.final_activation=None if final_activation is None else getattr(torch,final_activation)

        self.net = nn.Sequential(
            # * Layer 1: 1x1
            TBlock(self.z_dim,128, 4,1, 0,norm_type),
            # * Layer 2: 4x4
            TBlock(128,64,4,2,1,norm_type),
            # * Layer 3: 8x8
            TBlock(64,32,4,2,1,norm_type),
            # * Layer 4: 16x16
            nn.ConvTranspose2d(32, self.out_ch, kernel_size=4, stride=2, padding=3, bias=False),
            # * 28x28
        )

    def forward(self, x):
        x = self.net(x)
        return x if self.final_activation is None else self.final_activation(x)

      

class Discriminator(nn.Module):
    def __init__(self, in_ch=3,norm_type:str="batch",final_activation=None):
        super().__init__()
        self.in_ch = in_ch
        self.final_activation=None if final_activation is None else getattr(torch,final_activation)
        self.net = nn.Sequential(
            # * 28x28
            CBlock(self.in_ch,32,4,2,1,norm_type),
            # * 14x14
            CBlock(32,64,4,2,1,norm_type),
            # * 7x7
            CBlock(64,32,3,2,1,norm_type),
            # * 4x4 
            nn.Conv2d(32, 1, kernel_size=4, stride=1, padding=0, bias=False)
            # * 1x1
        )

    def forward(self, x):
        x = self.net(x)
        return x if self.final_activation is None else self.final_activation(x)
    
class norm_layer(nn.Module):
    def __init__(self, num_channels,norm_type: str = None):
        super().__init__()
        if norm_type == "BatchNorm2d":
            self.norm = nn.BatchNorm2d(num_channels)
        elif norm_type == "GroupNorm":
            self.norm = nn.GroupNorm(num_channels, num_channels)
        elif norm_type is None or norm_type == "None":
            self.norm=None
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")
        
    def forward(self, x):
        return x if self.norm is None else self.norm(x)
