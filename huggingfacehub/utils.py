import torch
import numpy as np
from PIL import Image
def norm(img):
            low=float(img.min())
            high=float(img.max())
            img.sub_(low).div_(max(high - low, 1e-5))

def recover_image(tensor):
        tensor=tensor.cpu().numpy().transpose(1, 2,0)*255
        
        return Image.fromarray(tensor.astype(np.uint8))
