import torch
from wgangp.utils import random_sample
import os
from PIL import Image
import numpy as np
import numpy as np
from torchvision.utils import make_grid

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser=ArgumentParser()
parser.add_argument('-s','--separate',action='store_true')
parser.add_argument('-c','--count',type=int,default=64)
parser.add_argument('-r','--rows',type=int,default=8)
args=parser.parse_args()
if args.rows<0 or args.count<0:
    print("invalid rows or count")
    exit()
if args.rows>args.count or args.count%args.rows!=0:
    print("invalid rows or count")
    exit()
print(args.separate,args.count,args.rows)
#exit()

def norm(img):
            low=float(img.min())
            high=float(img.max())
            img.sub_(low).div_(max(high - low, 1e-5))

def recover_image(tensor):
        tensor=tensor.cpu().numpy().transpose(1, 2,0)*255
        
        return Image.fromarray(tensor.astype(np.uint8))
dir_name="samples"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)    
from transformers import AutoModel
generator=AutoModel.from_pretrained("hikmatfarhat/WGANGP_generator",trust_remote_code=True)
generator=generator.to("cuda")
with torch.no_grad():
        noise=random_sample(args.count,128,"cuda")
        fake_images=generator(noise)
        if not args.separate:
          res=make_grid(fake_images,nrow=args.rows,padding=2,normalize=True)
          norm(res)
          img=recover_image(res)
          img.save(os.path.join("samples",f"grid.png"))
        else:
            noise = random_sample(args.count,128,"cuda")
            fake_images = generator(noise)  
            norm(fake_images)
            for i in range(args.count):
                img=recover_image(fake_images[i])
                img.save(os.path.join("samples",f"sample_{i}.png"))

