import torch
from networks import Generator
from utils import random_sample
import yaml
from munch import DefaultMunch
import os
from PIL import Image
import numpy as np
from torchvision.utils import make_grid
import torch.functional as F
from torchvision.utils import make_grid
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--weights-path", type=str, help="weights path")
args=parser.parse_args()



cfg_path = "config.yml"
with open(cfg_path, "r") as f:
    print(f"Loading config file: {cfg_path}")
    cfg = yaml.safe_load(f)
cfg = DefaultMunch.fromDict(cfg)
generator=Generator(
            z_dim=cfg.z_dim,
            out_ch=cfg.img_ch,norm_type=cfg.g_norm_type,
            final_activation=cfg.g_final_activation
        )
dir_list=os.listdir(cfg.weights_dir)

if args.weights_path is None:
    if dir_list:
            gen_files=[f for f in dir_list if f.startswith("gen")]
            dis_files=[f for f in dir_list if f.startswith("dis")]
            gen_files.sort()
            dis_files.sort()

            generator.load_state_dict(torch.load(cfg.weights_dir+"/"+gen_files[-1]))
           
            print(f"loaded weights from {cfg.weights_dir}/{gen_files[-1]} and {cfg.weights_dir}/{dis_files[-1]}")
    else:
        exit("No weights found")
else:
    generator.load_state_dict(torch.load(args.weights_path))
    print(f"loaded weights from {args.weights_path}")

def norm(img):
            low=float(img.min())
            high=float(img.max())
            img.sub_(low).div_(max(high - low, 1e-5))

def recover_image(tensor):
        tensor=tensor.cpu().numpy().transpose(1, 2,0)*255
        
        return Image.fromarray(tensor.astype(np.uint8))
    
    
generator.to(cfg.device)
generator.eval()

with torch.no_grad():
  for k in range(cfg.num_sample_epochs):
    noise = random_sample(cfg.batch_size,cfg.z_dim,cfg.device)
    fake_images = generator(noise)  
    norm(fake_images)
    for i in range(fake_images.shape[0]):
        img=recover_image(fake_images[i])
        img.save(os.path.join(cfg.samples_dir,f"sample_{cfg.batch_size*k+i}.png"))

