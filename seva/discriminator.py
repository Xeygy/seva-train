import torch
from torch import nn
from seva.model import *
from seva.modules.layers import timestep_embedding

class DownConv(nn.Module):
    def __init__(self, in_dim, discriminator_feature_dim):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_dim, discriminator_feature_dim, 4, 1, 1, bias=False),
                nn.SiLU(True),
                nn.Conv2d(discriminator_feature_dim, discriminator_feature_dim, 4, 1, 1, bias=False),
                nn.GroupNorm(4, discriminator_feature_dim),
                nn.SiLU(True),
                nn.Conv2d(discriminator_feature_dim, discriminator_feature_dim, 4, 1, 1, bias=False),
            )

    def forward(self, x):
        return self.conv(x)


class DiscriminatorLinearHead(nn.Module):
    def __init__(self, in_visual_channels, in_text_channels, in_time_channels, hidden_dim, num_heads):
        super().__init__()

        self.proj_visual = DownConv(in_visual_channels, hidden_dim)
        # self.proj_visual = nn.Linear(in_visual_channels, hidden_dim)
        self.proj_text = nn.Linear(in_text_channels, hidden_dim)
        self.proj_time = nn.Linear(in_time_channels, hidden_dim)

        self.q_ln = nn.LayerNorm(hidden_dim)
        self.k_ln = nn.LayerNorm(hidden_dim)
        # self.v_ln = nn.LeyerNorm(hidden_dim)

        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True) 
        self.head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, encoder_hidden_states, time_emb):
        time_emb = time_emb.unsqueeze(dim=1)

        # print(x.shape)
        visual = self.proj_visual(x)
        # print(visual.shape)
        visual = torch.flatten(visual, start_dim=2, end_dim=3).permute(0, 2, 1)

        text = self.proj_text(encoder_hidden_states)
        time = self.proj_time(time_emb)  

        query = self.q_ln(visual + time)
        key = self.k_ln(text)
        value = text

        attn_output, attn_output_weights = self.multihead_attn(query=query, key=key, value=value)

        x = attn_output + visual
        
        return self.head(x)

class Discriminator_Seva(nn.Module):
    def __init__(self, seva_model: Seva):
        super().__init__()
        self.num_frames = seva_model.params.num_frames
        self.time_embed = seva_model.time_embed
        self.input_blocks = seva_model.input_blocks
        self.model_channels = seva_model.model_channels

        self.disc_head_0 = DiscriminatorLinearHead(in_visual_channels=320, in_text_channels=1024, in_time_channels=1280, hidden_dim=768, num_heads=4)
        self.disc_head_1 = DiscriminatorLinearHead(in_visual_channels=640, in_text_channels=1024, in_time_channels=1280, hidden_dim=768, num_heads=4)
        self.disc_head_2 = DiscriminatorLinearHead(in_visual_channels=1280, in_text_channels=1024, in_time_channels=1280, hidden_dim=768, num_heads=4)
        self.disc_head_3 = DiscriminatorLinearHead(in_visual_channels=1280, in_text_channels=1024, in_time_channels=1280, hidden_dim=768, num_heads=4)

        self.heads = [
            self.disc_head_0, self.disc_head_1, self.disc_head_2, self.disc_head_3
        ]
        self.last_layer_indices_at_each_level = [3, 6, 9, 11]


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
        for module in self.input_blocks:
            h = module(
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
            output_from_encoded_block = hs[self.last_layer_indices_at_each_level[i]]
            scores.append(head_block(
                x = output_from_encoded_block,
                encoder_hidden_states = y,
                time_emb = t_emb   
            ))
            
    
        return torch.cat(scores, dim=1)