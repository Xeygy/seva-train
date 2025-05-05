import torch
from torch import nn
from seva.model import *
from seva.modules.layers import timestep_embedding

class Discriminator_Seva(nn.Module):
    def __init__(self, seva_model: Seva):
        super().__init__()
        self.num_frames = seva_model.params.num_frames
        self.time_embed = seva_model.time_embed
        self.input_blocks = seva_model.input_blocks
        self.model_channels = seva_model.model_channels
        self.midle_block = seva_model.middle_block

        channel_list = [320, 640, 1280]
        self.heads = []
        for i, feat_c in enumerate(channel_list):
            head = nn.Sequential(
                nn.GroupNorm(32, feat_c, eps=1e-05, affine=True),
                nn.Conv2d(feat_c, feat_c // 4, kernel_size=4, stride=2, padding=2),
                nn.SiLU(),
                nn.Conv2d(feat_c // 4, 1, kernel_size=1, stride=1, padding=0)
            )
            setattr(self, f"disc_head_{i}", head)  # Optional: allows self.disc_head_0, etc.
            self.heads.append(head)


        self.last_layer_indices_at_each_level = [3, 6]


    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            y: torch.Tensor,
            dense_y: torch.Tensor,
            num_frames: int | None = None,
        ) -> torch.Tensor:
        
        num_frames = num_frames or self.num_frames
        t_emb = timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)
        hs = []
        h = x
        ## Run through input blocks
        for i,module in enumerate(self.input_blocks):
            h = module(
                h,
                emb=t_emb,
                context=y,
                dense_emb=dense_y,
                num_frames=num_frames,
            )
            if i in self.last_layer_indices_at_each_level:
                hs.append(h)

        ## Run through mid block
        h = self.midle_block(
            h,
            emb=t_emb,
            context=y,
            dense_emb=dense_y,
            num_frames=num_frames,
        )
        hs.append(h)
        scores = []
        for i in range(len(self.heads)):
            head_block = self.heads[i]
            output_from_encoded_block = hs[i]
            scores.append(head_block(output_from_encoded_block).reshape(output_from_encoded_block.shape[0], -1))
        
        scores = torch.cat(scores, 1)
    
        return scores