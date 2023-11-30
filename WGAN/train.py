import comet_ml
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as vt


#import os
import yaml
from munch import DefaultMunch
from tqdm import trange
import random
import numpy as np
from wgan_gp import WGAN_GP


def set_seed(seed):
        torch.backends.cudnn.deterministic = True
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


cfg_path = "config.yml"
with open(cfg_path, "r") as f:
    print(f"Loading config file: {cfg_path}")
    cfg = yaml.safe_load(f)
cfg = DefaultMunch.fromDict(cfg)

set_seed(cfg.seed)

transforms = vt.Compose([vt.ToTensor(),vt.Normalize(0.5, 0.5),
    vt.Resize((cfg.imsize, cfg.imsize),antialias=True)])

dataset = ImageFolder(
    root=cfg.data_dir, transform=transforms
)
    
dataloader = DataLoader(
    dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=4,
    drop_last=True,
)
model=WGAN_GP(cfg)
start_epoch=model.starting_epoch
loop = trange(start_epoch,start_epoch+cfg.epochs, desc="Epoch: ", ncols=75)



experiment = comet_ml.Experiment(project_name=cfg.comet_project, workspace=cfg.comet_workspace,
                                         auto_metric_logging=False, auto_output_logging=False)
#experiment.set_name(cfg.experiment_name)
experiment.log_parameters(cfg)


for epoch in loop:
    loss_d,loss_g=model.train_epoch(dataloader)
    loop.set_postfix(loss_d=loss_d,loss_g=loss_g)
    metrics={'loss_d':loss_d,'loss_g':loss_g}
    experiment.log_metrics(metrics, epoch=epoch)
    if (epoch+1) % cfg.save_model_freq == 0:
        print(f"Saving model at epoch {epoch}")
        model.save_model(epoch)
        #fid=model.compute_fid(dataloader=dataloader)
        #experiment.log_metric("fid",fid,epoch=epoch)
    if(epoch+1) % cfg.save_image_freq == 0:
        img=model.save_images(epoch,32)
        experiment.log_image(img)
        print(f"Saving images at epoch {epoch}")

