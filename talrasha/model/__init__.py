from .diffusion.diffusion_model import DiffusionModel as BasicDiffusion
from .vlb.vae import VanillaVAE
from .vlb.cat_vae import CategoricalVAE
from .gan.gan import VanillaGAN
from .various_generative.wae import WAEWithMMD
from .various_generative.neural_process import AttentiveNP

__all__ = [
    'BasicDiffusion',
    'VanillaVAE',
    'CategoricalVAE',
    'VanillaGAN',
    'WAEWithMMD',
    'AttentiveNP'
]
