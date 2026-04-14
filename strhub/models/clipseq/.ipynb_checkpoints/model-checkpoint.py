

from timm.models.vision_transformer import VisionTransformer, PatchEmbed
import torch
import torch.nn as nn
import math
from typing import Sequence, Any, Optional
from timm.models.helpers import named_apply
from timm.models.layers import trunc_normal_ 
from functools import partial
from strhub.models.stn import STN

def init_weights(module: nn.Module,
                 name: str = '',
                 exclude: Sequence[str] = ()):
    """Initialize the weights using the typical initialization schemes used in SOTA models."""
    if any(map(name.startswith, exclude)):
        return
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight,
                                mode='fan_out',
                                nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

class OTE_Encoder(VisionTransformer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed,class_token=False, is_tps = False):
        super().__init__(img_size, patch_size, in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                         num_classes=0, global_pool='', class_token=class_token)  # these disable the classifier head
        self.is_tps = is_tps
        print(f'tps:{is_tps}')
        if is_tps:
            self.stn = STN(in_channels=3, output_image_size = img_size)
            print('init stn')

    def forward(self, x):
        if self.is_tps:
            x = self.stn(x)
        # Return all tokens
        return self.forward_features(x)  
    

class OTE_Decoder(nn.Module):
    def __init__(self, max_label_length: int, embed_dim: int,
                 cls_token: bool) -> None:
        super().__init__()
        self.cls_token = cls_token
        self.max_label_length = max_label_length
        self.position_embedding_cpa = nn.Embedding(max_label_length, embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.fc3 = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.2)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x_feat):
        B, N, C = x_feat.shape
        if self.cls_token:
            x_feat = x_feat[:, :1, :]
        else:
            x_feat = F.adaptive_avg_pool1d(x_feat.transpose(2, 1),
                                           1).transpose(2, 1)
        reading_order = torch.arange(self.max_label_length).unsqueeze(0).tile(
            [B, 1]).to(x_feat.device)
        reading_order_embed = self.position_embedding_cpa(reading_order)
        x_att = torch.tanh(self.fc1(x_feat) + reading_order_embed)
        x_att = self.softmax(self.fc2(x_att))
        x = self.fc3(x_att * x_feat)
        x = self.norm(x)
        return x, x_feat
    
class OTE_CLIP(nn.Module):
    def __init__(self,
                 num_classes: int,
                 max_label_length: int,
                 img_size: Sequence[int],
                 patch_size: Sequence[int],
                 embed_dim: int,
                 enc_num_heads: int,
                 enc_mlp_ratio: int,
                 enc_depth: int,
                 dropout: float,
                 class_token=True,
                 is_tps = False,
                 **kwargs: Any) -> None:
        super().__init__()
        self.max_label_length = max_label_length
        
        self.encoder = OTE_Encoder(img_size,
                                  patch_size,
                                  embed_dim=embed_dim,
                                  depth=enc_depth,
                                  num_heads=enc_num_heads,
                                  mlp_ratio=enc_mlp_ratio,
                                  class_token=class_token,
                                  is_tps = is_tps)
        
        self.decoder = OTE_Decoder(max_label_length=max_label_length + 1,
                                   embed_dim=embed_dim,
                                   cls_token=class_token)
        self.head = nn.Sequential(
                          nn.Linear(embed_dim, 2 * embed_dim),
                          nn.LayerNorm(embed_dim * 2), nn.GELU(),
                          nn.Linear(embed_dim * 2, num_classes))
        
        named_apply(partial(init_weights, exclude=['encoder']), self)

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dropout = nn.Dropout(dropout)

    @torch.jit.ignore
    def no_weight_decay(self):
        param_names = {'text_embed.embedding.weight', 'pos_queries'}
        enc_param_names = {
            'encoder.' + n
            for n in self.encoder.no_weight_decay()
        }
        return param_names.union(enc_param_names)

    def encode(self, img):
        return self.encoder(img)

    def decode(self, x_feat):
        return self.decoder(x_feat)

    def forward(self, images, max_len=None):
        x_feat = self.encode(images)
        x_out, x_feat = self.decoder(x_feat)
        logits = self.head(x_out)
        return logits, x_feat
    

    