"""
### mostly copy from fractalgen/models/fractalgen.py
"""
from fractalgen.models.ar import AR
import torch
import torch.nn as nn

class FractalGen(nn.Module):
    def __init__(
            self,
            action_step_list,
            embed_dim_list,
            num_blocks_list,
            num_heads_list,
            generator_type_list,
            label_drop_prob=0.1,
            class_num=1000,
            attn_dropout=0.1,
            proj_dropout=0.1,
            guiding_pixel=False,
            num_conds=1,
            r_weight=1.0,
            grad_checkpointing=False,
            fractal_level=0
    ):
        super().__init__()
        self.fractal_level = fractal_level
        self.num_fractal_levels = len(action_step_list)

        
