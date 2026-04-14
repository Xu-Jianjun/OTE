import math
from typing import Optional

import torch
from torch import nn as nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import transformer

from timm.models.vision_transformer import VisionTransformer, PatchEmbed


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu',
                 layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn_1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_c = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.activation = transformer._get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super().__setstate__(state)
    
    def forward(self, content, query_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None, content_key_padding_mask: Optional[Tensor] = None):
        content_norm = self.norm_c(content)
        content2, sa_weights = self.self_attn_1(content_norm, content_norm, content_norm, attn_mask=query_mask,key_padding_mask=tgt_key_padding_mask)
        content = content + self.dropout1(content2)
        content2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm1(content)))))
        content = content + self.dropout3(content2)
        return content,sa_weights
    

class PreDecoder(nn.Module):
    __constants__ = ['norm']
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu',
                 layer_norm_eps=1e-5):
        
    
    
    def forward(self,
        
    
    
    
    
        
        
        
        
        
    
        
            