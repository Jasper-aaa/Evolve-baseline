import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.training_utils import EMAModel
from quest.algos.base import ChunkPolicy
class FractalPolicy(ChunkPolicy):
    def __init__(self, fractal_model, **kwargs):
        super().__init__(**kwargs)

        self.fratal_model = fractal_model.to(self.device)

    
    def compute_loss(self, data):
        data = self.preprocess_input(data, train_mode=True)
        cond = self.get_cond(data)
        loss = self.fratal_model(cond, data["actions"])
        info = {
            'loss': loss.item(),
        }
        return loss, info
    
    def sample_actions(self, data):
        data = self.preprocess_input(data, train_mode=False)
        cond = self.get_cond(data)
        actions = self.diffusion_model.get_action(cond)
        actions = actions.permute(1,0,2)
        return actions.detach().cpu().numpy()

    def get_cond(self, data):
        obs_emb = self.obs_encode(data)
        obs_emb = obs_emb.reshape(obs_emb.shape[0], -1)
        lang_emb = self.get_task_emb(data)
        cond = torch.cat([obs_emb, lang_emb], dim=-1)
        return cond
    

