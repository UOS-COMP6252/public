import huggingface_hub
from .config import Config
from transformers import PreTrainedModel
from .dcgan import Generator
# utils not used but importing it forces the upload to huggingface hub to include it
from .utils import *

class WGANGP_generator(PreTrainedModel):
    config_class = Config
    def __init__(self, config):
        super().__init__(config)
        
        self.generator=Generator(config.cfg["imsize"],config.cfg["img_ch"],config.cfg["zdim"],
           config.cfg["norm_type"]["g"],config.cfg["final_activation"]["g"])

    def forward(self, input):
        return self.generator(input)
