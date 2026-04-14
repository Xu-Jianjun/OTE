

from timm.models.vision_transformer import VisionTransformer, PatchEmbed
import torch
import torch.nn as nn
import math
from typing import Sequence, Any, Optional
from timm.models.helpers import named_apply
from timm.models.layers import trunc_normal_ 
from functools import partial
from strhub.models.stn import STN
from .decoder import DecoderLayer, Decoder


class TokenEmbedding(nn.Module):
    def __init__(self, charset_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens: torch.Tensor):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)
    

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
    
class OTE_CLIP(nn.Module):
    def __init__(self,
                 num_classes: int,
                 max_label_length: int,
                 img_size: Sequence[int],
                 patch_size: Sequence[int],
                 embed_dim: int,
                 vocab_size:int,
                 enc_num_heads: int,
                 enc_mlp_ratio: int,
                 enc_depth: int,
                 dropout: float,
                 dec_depth=1,
                 dec_num_heads=6,
                 dec_mlp_ratio=4,
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
        
        decoder_layer = DecoderLayer(d_model=embed_dim,
                                     nhead=dec_num_heads,
                                     dim_feedforward=embed_dim * dec_mlp_ratio,
                                     dropout=dropout)

        self.decoder = Decoder(decoder_layer,
                               num_layers=dec_depth,
                               max_label_length=max_label_length,
                               d_model=embed_dim,
                               nhead=dec_num_heads,
                               dropout=dropout
                               )
        
        self.text_embed = TokenEmbedding(vocab_size, embed_dim)
        
        # self.pos_queries = nn.Parameter(torch.Tensor(1, max_label_length + 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

        
        self.head = nn.Sequential(
                          nn.Linear(embed_dim, 2 * embed_dim),
                          nn.LayerNorm(embed_dim * 2), nn.GELU(),
                          nn.Linear(embed_dim * 2, num_classes))
        
        # self.head = nn.Sequential(
        #                   nn.Linear(embed_dim, num_classes)
        #                   )
        
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
    
    def decode(self, x_feat, tgt=None, attn_mask=None,tgt_padding_mask=None,max_len=None):
        if tgt is not None:
            B, L = tgt.shape
            tgt_emb = self.norm(self.text_embed(tgt))
        elif max_len is not None:
            L = max_len
            tgt_emb = None 
        else:
            L = self.max_label_length
            tgt_emb = None
        return self.decoder(x_feat, L=L, tgt_emb=tgt_emb, attn_mask = attn_mask,content_key_padding_mask = tgt_padding_mask)       
