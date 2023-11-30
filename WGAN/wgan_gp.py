import torch
import torch.nn as nn
from torch.optim import Adam
from torch import autograd
from tqdm import tqdm
from collections import defaultdict
import os
from PIL import Image
import numpy as np
from networks import Generator,Discriminator
from utils import init_weight,random_sample,norm
from torchvision.utils import make_grid
from torchmetrics.image.fid import FrechetInceptionDistance

class WGAN_GP():
    """
    WGAN_GP Wasserstein GAN. Uses gradient penalty instead of gradient clipping to enforce 1-Lipschitz continuity. 
    """

    def __init__(self, cfg):
       # super().__init__(cfg, writer)
        self.cfg = cfg
        self.d_iter_per_g = 1 if self.cfg.d_iter_per_g is None else self.cfg.d_iter_per_g
# see https://arxiv.org/abs/1511.06434
        # self.generator = Generator(
        #     z_dim=self.cfg.z_dim,
        #     out_ch=self.cfg.img_ch,norm_type=self.cfg.g_norm_type,
        #     final_activation=self.cfg.g_final_activation
        # )
       
        self.generator=Generator(cfg.imsize,cfg.img_ch,cfg.zdim,
                                 norm_type=cfg.norm_type.g,
                                 final_activation=self.cfg.final_activation.g)
        self.discrim = Discriminator(cfg.imsize,cfg.img_ch,norm_type=cfg.norm_type.d,
                                     final_activation=cfg.final_activation.d)
        self.initialize()
        self.set_optimizers()

    def initialize(self):
        dir_list=os.listdir(self.cfg.weights_dir)
    
            
        if self.cfg.resume and  dir_list:
            gen_files=[f for f in dir_list if f.startswith("gen")]
            dis_files=[f for f in dir_list if f.startswith("dis")]
            gen_files.sort()
            dis_files.sort()

            self.generator.load_state_dict(torch.load(self.cfg.weights_dir+"/"+gen_files[-1]))
            self.discrim.load_state_dict(torch.load(self.cfg.weights_dir+"/"+dis_files[-1]))
            print(f"loaded weights from {self.cfg.weights_dir}/{gen_files[-1]} and {self.cfg.weights_dir}/{dis_files[-1]}")
            self.starting_epoch=int(gen_files[-1].split("_")[1].split(".")[0])
            #self.starting_epoch=int(re.findall(r'\d+', gen_files[-1])[0])
        else:
            self.generator.apply(init_weight)
            self.discrim.apply(init_weight)
            self.starting_epoch=0
            
    def set_optimizers(self):
        self.generator = self.generator.to(self.cfg.device)
        self.discrim = self.discrim.to(self.cfg.device)

        self.optG = Adam(self.generator.parameters(), lr=self.cfg.lr.g)
        self.optD = Adam(self.discrim.parameters(), lr=self.cfg.lr.d)

    def generator_step(self, data):

        noise = random_sample(self.cfg.batch_size, self.cfg.zdim, self.cfg.device)
        fake_images = self.generator(noise)
        fake_logits = self.discrim(fake_images)
        g_loss = -fake_logits.mean().view(-1)
        self.optG.zero_grad()

        g_loss.backward()
        self.optG.step()

        self.metrics["G-loss"] += [g_loss.item()]

    def discriminator_step(self, data):
        
        real_images = data[0].float().to(self.cfg.device)
        noise = random_sample(self.cfg.batch_size, self.cfg.zdim, self.cfg.device)
        fake_images = self.generator(noise)
        
        real_logits = self.discrim(real_images)
        fake_logits = self.discrim(fake_images)

        gradient_penalty = self.cfg.w_gp * self._compute_gp(
            real_images, fake_images
        )

        loss_c = fake_logits.mean() - real_logits.mean()
        d_loss = loss_c + gradient_penalty

        self.optD.zero_grad()
        d_loss.backward()
        self.optD.step()

        self.metrics["D-loss"] += [d_loss.item()]
        self.metrics["GP"] += [gradient_penalty.item()]
    def train_epoch(self, dataloader):
        # use of defaultdict instead of regular dict
        # saves us the trouble of checking if a key exists
        # which it doesn't when we start appending to it
        self.metrics = defaultdict(list)
        
        loop = tqdm(dataloader, desc="Iteration: ",leave=False)

        for idx, data in enumerate(loop):
            self.discriminator_step(data)
            if idx % self.cfg.d_iter_per_g == 0:
                self.generator_step(data)
        
        return np.mean(self.metrics["D-loss"]),np.mean(self.metrics["G-loss"])

    def _compute_gp(self, real_data, fake_data):
        batch_size = real_data.size(0)
        eps = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
        eps = eps.expand_as(real_data)
        interpolation = eps * real_data + (1 - eps) * fake_data

        intp_logits = self.discrim(interpolation)
        grad_outputs = torch.ones_like(intp_logits)

        gradients = autograd.grad(
            outputs=intp_logits,
            inputs=interpolation,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, 1)
        return torch.mean((grad_norm - 1) ** 2)
    
    def generate_images(self, nsamples):
        self.generator.eval()
        with torch.no_grad():
            noise = random_sample(self.cfg.batch_size, self.cfg.zdim, self.cfg.device)[:nsamples]
            fake_images = self.generator(noise)
        return fake_images
    
    def save_images(self,epoch,nsamples=32):
    
        fake_images = self.generate_images(nsamples=nsamples)
        grid = make_grid(fake_images, nrow=4, normalize=True)
        img=self.recover_image(grid)
        img.save(os.path.join(self.cfg.images_dir,f"sample_{epoch}.png"))
        return img
    
    def recover_image(self,img):
        # PIL expects the image to be of shape (H,W,C)
        # in PyTorch it's (C,H,W)

        img=img.cpu().numpy().transpose(1, 2,0)*255
        return Image.fromarray(img.astype(np.uint8))
    
    def save_model(self,epoch):
        # if the directory doesn't exist, create it
        try:
            os.mkdir(self.cfg.weights_dir)
        except:
            pass
        torch.save(self.generator.state_dict(), os.path.join(self.cfg.weights_dir, f"generator_{epoch:03}.pth"))
        torch.save(self.discrim.state_dict(), os.path.join(self.cfg.weights_dir, f"discrim_{epoch:03}.pth"))

    def compute_fid(self,dataloader):
        #TODO: fix this because gpu running out of memory and cpu is too slow
        fid=FrechetInceptionDistance(feature=2048,normalize=True)
        fid.to(self.cfg.device)
        idx=0
        for data in tqdm(dataloader):
            print(idx)
            idx+=1
            real_images=data[0].float().to(self.cfg.device)
        #    real_images=data[0].float()
            noise = random_sample(self.cfg.batch_size,self.cfg.zdim,self.cfg.device)
           # g=self.generator.to('cpu')
            fake_images = self.generator(noise)
            norm(fake_images)
            norm(real_images)
            fid.update(fake_images,real=False)
            fid.update(real_images,real=True)
        return fid.compute()
            