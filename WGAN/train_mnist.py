import comet_ml
from comet_ml.integration.pytorch import log_model
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as vt
from torchvision.datasets import MNIST

#import os
import yaml
from munch import DefaultMunch
from tqdm import trange
import random
import numpy as np
from wgan_gp_mnist import WGAN_GP


def set_seed(seed):
        torch.backends.cudnn.deterministic = True
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


cfg_path = "config-mnist.yml"
with open(cfg_path, "r") as f:
    print(f"Loading config file: {cfg_path}")
    cfg = yaml.safe_load(f)
cfg = DefaultMunch.fromDict(cfg)

set_seed(cfg.seed)

transforms = vt.Compose([vt.ToTensor(),vt.Normalize(0.5, 0.5),
    vt.Resize((cfg.imsize, cfg.imsize),antialias=True)])

dataset =MNIST(root=cfg.data_dir, transform=transforms,download=True)
    
dataloader = DataLoader(
    dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=4,
    drop_last=True,
)
model=WGAN_GP(cfg)

loop = trange(cfg.epochs, desc="Epoch: ", ncols=75)



experiment = comet_ml.Experiment(project_name=cfg.comet_project, workspace=cfg.comet_workspace,
                                         auto_metric_logging=False, auto_output_logging=False)
experiment.set_name("mnist")
experiment.log_parameters(cfg)


for epoch in loop:
    loss_d,loss_g=model.train_epoch(dataloader)
    loop.set_postfix(loss_d=loss_d,loss_g=loss_g)
    metrics={'loss_d':loss_d,'loss_g':loss_g}
    experiment.log_metrics(metrics, epoch=epoch)
    if  epoch!=0 and epoch% cfg.save_model_freq == 0:
        model.save_model(epoch)
        model_checkpoint = {
           "epoch": epoch,
           "gen_state_dict": model.generator.state_dict(),
           "disc_state_dict": model.discrim.state_dict(),
           "optG_state_dict": model.optG.state_dict(),
           "optD_state_dict": model.optD.state_dict()
        }
        log_model(experiment, model_checkpoint,model_name="wgan_gp_mnist")
    if epoch!=0 and  epoch% cfg.save_image_freq == 0:
         img=model.save_images(epoch,32)
         experiment.log_image(img)

