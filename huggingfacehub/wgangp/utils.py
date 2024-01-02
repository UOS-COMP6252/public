#import yaml
#from munch import DefaultMunch
import torch
import torch.nn as nn

def norm(img):
            low=float(img.min())
            high=float(img.max())
            img.sub_(low).div_(max(high - low, 1e-5))
def random_sample(batch_size, z_dim, device):
        # input to the generator
        # z_dim channels, 1x1 pixels
        return torch.randn(batch_size,z_dim, 1, 1).to(device)


def init_weight(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        # the weights are initialized according to
        # https://arxiv.org/abs/1511.06434
        nn.init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            if m.bias.data is not None:
                m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()
