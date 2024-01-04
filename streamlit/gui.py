import streamlit as slt 
import math
import torch
from torchvision.utils import make_grid
from utils import norm,recover_image
# it seems streamlit reruns the whole script when an event occurs
from transformers import AutoModel
if "generator" not in slt.session_state:
    slt.session_state.generator=AutoModel.from_pretrained("hikmatfarhat/WGANGP_generator",trust_remote_code=True)
slt.markdown("# Choose the number of images to generate from the sidebar")
def display_images():
    with torch.no_grad():
        total=slt.session_state.size
        rows=int(math.sqrt(total))
        noise=torch.randn(total,128, 1, 1)
        generator=slt.session_state.generator
        fake_images=generator(noise)
        res=make_grid(fake_images,nrow=rows,padding=2,normalize=True)
        norm(res)
        img=recover_image(res)
        slt.session_state.image=img
        #slt.image(img)
if 'image' in slt.session_state:
    slt.image(slt.session_state.image)

slt.sidebar.selectbox("Select number of images",[16,32,64],key="size")

slt.sidebar.button("Generate images",on_click=display_images)#,args=[slt.session_state.total])
