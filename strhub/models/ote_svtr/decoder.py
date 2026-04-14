# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional

import torch
from torch import nn as nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import transformer

from timm.models.vision_transformer import VisionTransformer, PatchEmbed


class DecoderLayer(nn.Module):
    """A Transformer decoder layer supporting two-stream attention (XLNet)
       This implements a pre-LN decoder, as opposed to the post-LN default in PyTorch."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu',
                 layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.dropout3 = nn.Dropout(dropout)

        self.activation = transformer._get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.gelu
        super().__setstate__(state)

    def forward(self, content: Optional[Tensor], attn_mask: Optional[Tensor], tgt_key_padding_mask: Optional[Tensor]):

        content_norm = self.norm1(content)
        content2, ca_weights = self.self_attn(content_norm, content_norm, content_norm,attn_mask=attn_mask, key_padding_mask=tgt_key_padding_mask)
        content = content + self.dropout1(content)
        content2 = self.linear2(self.dropout2(self.activation(self.linear1(self.norm2(content)))))
        content = content + self.dropout3(content2)

        return content

class PositionDecoderLayer(nn.Module):
    def __init__(self, max_label_length,d_model,nhead,dropout,layer_norm_eps=1e-5):
        super().__init__()
        self.pos_queries = nn.Parameter(torch.Tensor(1, max_label_length, d_model))
        self.fc1 = nn.Linear(d_model,d_model)
        self.fc2 = nn.Linear(d_model,d_model)
        self.fc3 = nn.Linear(d_model,d_model)
        self.norm_p = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_c = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm = nn.LayerNorm(d_model,eps=layer_norm_eps)
        self.softmax = nn.Softmax(dim=-1)

        nn.init.trunc_normal_(self.pos_queries, std=.02)
        
    def forward(self,x_feat,L):
        B , N , D = x_feat.shape
        reading_order_embed = self.pos_queries[:,:L,:].tile([B,1,1])
        x_att = torch.tanh(self.fc1(x_feat) + reading_order_embed)
        x_att = self.softmax(self.fc2(x_att))
        x = self.fc3(x_att * x_feat)
        x = self.norm(x)
        return x
    
class Decoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, max_label_length,d_model, nhead, dropout, layer_norm_eps=1e-5):
        super().__init__()
        self.layers = transformer._get_clones(decoder_layer, num_layers)
        self.pos_queries = nn.Parameter(torch.Tensor(1, max_label_length, d_model))
        self.position_attn = PositionDecoderLayer(max_label_length=max_label_length,d_model=d_model,nhead=nhead,dropout=dropout,layer_norm_eps=layer_norm_eps)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model, layer_norm_eps)
        nn.init.trunc_normal_(self.pos_queries, std=.02)

    def forward(self, x_feat, L:Optional[int]=None, tgt_emb: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None,
                content_key_padding_mask: Optional[Tensor] = None):   
        x = self.position_attn(x_feat, L)
        if tgt_emb is not None:
            assert tgt_emb.shape[1] == x.shape[1] 
            x = x + tgt_emb+self.pos_queries[:,:L,:]
            # x = x + tgt_emb
        x = torch.cat([x_feat,x],dim=1)
        # print(x[0,:1,:])
        for i ,mod in enumerate(self.layers):
            x = mod(x, attn_mask, content_key_padding_mask)
        x = self.norm(x)
        return x[:,1:,:]
        
        
        
        
        #query位置编码
        #content内容
        #memory视觉特征


