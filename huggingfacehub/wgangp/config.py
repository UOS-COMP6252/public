from transformers import PretrainedConfig
class Config(PretrainedConfig):
    model_type = "WGANGP_generator"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cfg=kwargs
