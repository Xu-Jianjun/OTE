"""
Implementation of ViTSTR based on timm VisionTransformer.

TODO:
1) distilled deit backbone
2) base deit backbone

Copyright 2021 Rowel Atienza
"""

from timm.models.vision_transformer import VisionTransformer,LayerScale
import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, \
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_, resample_patch_embed, \
    resample_abs_pos_embed, RmsNorm, PatchDropout, use_fused_attn, SwiGLUPacked
from timm.models import named_apply,checkpoint_seq


def random_mask(tensor, mask_prob=0.5):
    """
    对输入的tensor在N维度上随机mask 50%的元素。

    参数:
    tensor (torch.Tensor): 输入的tensor，形状为 B x N x C。
    mask_prob (float): 被mask的元素比例，默认为0.5。

    返回:
    torch.Tensor: 被mask后的tensor。
    """
    B, N, C = tensor.shape

    # 为每个B生成一个N维的mask
    mask = torch.rand(B, N) < mask_prob

    # 将mask应用到整个C维度
    mask = mask.unsqueeze(-1).expand(-1, -1, C).to(tensor.device)

    # 应用mask
    tensor = tensor.masked_fill(mask, 0)  # 这里用0来代表被mask的元素
    

    return tensor,mask


class Block1(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
            word_length=25,
    ):
        super().__init__()
        self.norm1_1 = norm_layer(dim)
        self.norm1_2 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, bias=qkv_bias,batch_first=True)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.word_length = word_length
        
    def forward(self, x):
        
        B,N,C = x.shape
        attn_mask = torch.full((N, N), float('-Inf'), device=x.device, dtype=x.dtype)
        attn_mask[:-self.word_length,:-self.word_length] = 0
        attn_mask[-self.word_length:,-self.word_length:] = 0
        # print('attn_mask',attn_mask.shape)
        # import pdb;pdb.set_trace()
        x_vision, x_word = x[:,:-self.word_length,:],x[:,-self.word_length:,:] 
        
        x = torch.cat([self.norm1_1(x_vision),self.norm1_2(x_word)],dim=1)
        x = x + self.drop_path1(self.ls1(self.attn(x,x,x,attn_mask=attn_mask)[0]))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    
        return x
    
class Block2(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
            word_length=25,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, bias=qkv_bias,batch_first=True)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.word_length = word_length
        
    def forward(self, x):
        
        B,N,C = x.shape
        attn_mask = torch.full((N, N), float('-Inf'), device=x.device, dtype=x.dtype)
        attn_mask[:,:-self.word_length] = 0
        attn_mask[-self.word_length:,-self.word_length:] = torch.triu(torch.ones(self.word_length, self.word_length) * float('-inf'), diagonal=1)
        x = self.norm1(x)
        x = x + self.drop_path1(self.ls1(self.attn(x,x,x,attn_mask=attn_mask)[0]))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    
        return x
    
class ViTSTR(nn.Module):
    """
    ViTSTR is basically a ViT that uses DeiT weights.
    Modified head to support a sequence of characters prediction for STR.
    """
    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = False,
            no_embed_class: bool = False,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: str = '',
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[Callable] = None,
            act_layer: Optional[Callable] = None,
            block_fn: Callable = Block1,
            mlp_layer: Callable = Mlp,
            word_length: int = 25,
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()
        self.word_length = word_length  
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block1(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            ) if i % 2 == 0 else Block2(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            ) for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.pixel_proj = nn.Sequential(*[nn.Linear(embed_dim,embed_dim*4),nn.GELU(),nn.Linear(embed_dim*4,embed_dim)])
        self.word_embedding = nn.Parameter(torch.randn(1, word_length, embed_dim) * .02)
        self.modality_prompt = nn.Embedding(2, embed_dim)

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        if self.word_embedding is not None:
            trunc_normal_(self.word_embedding, std=.02)
        named_apply(init_weights_vit_timm, self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token','word_embedding'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def _intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
    ):
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_class_token: bool = False,
            norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """ Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward_features(self, x):
        x = self.patch_embed(x)
        # print("before pos_embed",x.shape)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            B,N,C = x.shape 
            # print(torch.tensor([0]*N+[1]*self.word_length,device=x.device))
            modality_prompt = self.modality_prompt(torch.tensor([0]*N+[1]*self.word_length,device=x.device).unsqueeze(0))
            x = torch.cat([x,self.word_embedding.expand(x.shape[0], -1, -1)],dim=1)+modality_prompt
            x = self.blocks(x)
        x = self.norm(x)
        return x

    

    def forward(self, x):
        x = self.forward_features(x)
        # x_vision = self.pixel_proj(x[:,:-self.word_length,:])
        x_word = self.head(x[:,-self.word_length:,:])
        return x_word
    
    def forward_with_mask(self, x, mask_ration):
        x_ori = self.patch_embed(x)
        # x ,mask = random_mask(x_ori,mask_ration)
        x = x_ori
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            B,N,C = x.shape 
            # print(torch.tensor([0]*N+[1]*self.word_length,device=x.device))
            modality_prompt = self.modality_prompt(torch.tensor([0]*N+[1]*self.word_length,device=x.device).unsqueeze(0))
            x = torch.cat([x,self.word_embedding.expand(x.shape[0], -1, -1)],dim=1)+ modality_prompt
            x = self.blocks(x)
        x = self.norm(x)
        x_vision = self.pixel_proj(x[:,:-self.word_length,:])
        x_word = self.head(x[:,-self.word_length:,:])
        # x_recog_loss = F.l1_loss(x_ori*mask, x_vision*mask, reduction='sum')/mask.sum()
        return None, x_vision, x_word
            
        
        

    
def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()
        
if __name__ == "__main__":
    x = torch.randn(1,3,224,224)
    model = ViTSTR()
    x = model(x)
