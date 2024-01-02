from wgangp.model import WGANGP_generator
from wgangp.config import Config
import yaml
from munch import DefaultMunch
from transformers import AutoConfig,AutoModel
import torch
## two lines below are necessary to 
## upload the python code to huggingface hub
## i.e, config.py and model.py
## also, they are essential to load the model using AutoModel.from_pretrained("hikmatfarhat/WGANGP_generator")
## Note that in Config class model_type = "WGANGP_generator", i.e. exactly the class name of the model
Config.register_for_auto_class()
WGANGP_generator.register_for_auto_class("AutoModel")

## not sure what the 2 lines below do
#AutoConfig.register("WGANGP_generator",Config)
#AutoModel.register(Config, WGANGP_generator)
cfg_path = "config.yml"
with open(cfg_path, "r") as f:
    print(f"Loading config file: {cfg_path}")
    cfg = yaml.safe_load(f)
cfg = DefaultMunch.fromDict(cfg)

config=Config(**cfg.toDict())

model=WGANGP_generator(config)
model.generator.load_state_dict(torch.load("/home/user/wgan-gp/chkpt/generator_499.pth"))
# save the config and the model to directory WGAN-GP. This mirrors what would be on huggingface hub
#config.save_pretrained("WGAN-GP")
#model.save_pretrained("WGAN-GP")


model.push_to_hub(cfg.name)
# No need to use the below
# because once "imported" from model.py they are automatically uploaded
# huggingface-cli upload repo-id dcgan.py
# huggingface-cli upload "repo-id utils.py 

# To use the pretrained model, use the following code
#from transformers import AutoModel
# example repo-id "hikmatfarhat/WGANGP_generator"
# see the generate_images.py
#generator=AutoModel.from_pretrained(repo-id,trust_remote_code=True)