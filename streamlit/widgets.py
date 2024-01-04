import streamlit as slt 
import math
import torch
from wgangp.utils import random_sample
import os
from PIL import Image
import numpy as np
import numpy as np
from torchvision.utils import make_grid
from utils import norm,recover_image
# it seems streamlit reruns the whole script when an event occurs
from transformers import AutoModel


def display_images():
    with torch.no_grad():
        #total=slt.session_state.total
        total=slt.session_state.size
        slt.session_state.rows=int(math.sqrt(total))
        rows=slt.session_state.rows
        noise=random_sample(total,128,"cuda")
        fake_images=generator(noise)
        res=make_grid(fake_images,nrow=rows,padding=2,normalize=True)
        norm(res)
        img=recover_image(res)
        #img.save(os.path.join("samples",f"grid.png"))
        slt.image(img)

generator=AutoModel.from_pretrained("hikmatfarhat/WGANGP_generator",trust_remote_code=True)
generator=generator.to("cuda")
def selected():
    pass
    #slt.write(f"you selected {slt.session_state.size}")
slt.sidebar.selectbox("Select number of images",[16,32,64],key="size",on_change=selected)

if 'total' not in slt.session_state:
    slt.session_state.total=16
rows=4

slt.sidebar.button("Generate images",on_click=display_images)#,args=[slt.session_state.total])
#slt.image("grid.png")

# def set_total(total):
#     slt.session_state.total=total
#     #slt.write(f"total={total}")
#     #slt.write(f"rows={int(math.sqrt(total))}")
#     slt.session_state.rows=int(math.sqrt(total))
# slt.sidebar.write("hello")



#slt.sidebar.selectbox("select here",["a","b","c"])
#slt.session_state.total=slt.slider("Number of images",16,64,step=4)
#slt.session_state.total=slt.slider("Number of images",16,64,step=4)
#slt.button("set 64",on_click=set_total,args=[64])
#slt.button("set 16",on_click=set_total,args=[16])
#slt.button("click me",on_click=slt.write,args=[slt.session_state.total])
# import streamlit as st

# if 'button' not in st.session_state:
#     st.session_state.button = False

# def click_button():
#     st.session_state.button = not st.session_state.button

# st.button('Click me', on_click=click_button)

# if st.session_state.button:
#     # The message and nested widget will remain on the page
#     st.write('Button is on!')
#     st.slider('Select a value')
# else:
#     st.write('Button is off!')