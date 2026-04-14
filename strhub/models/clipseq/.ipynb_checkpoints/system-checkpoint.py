from typing import Sequence, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F 
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from timm.models.layers import trunc_normal_ 
from dataclasses import dataclass
from nltk import edit_distance
import numpy as np

from strhub.models.base import CrossEntropySystem
from strhub.models.utils import init_weights
from strhub.models.stn import STN

from .clip import clip
from .model import OTE_CLIP as Model
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()



@dataclass
class BatchResult:
    num_samples: int
    correct: int
    ned: float
    confidence: float
    label_length: int
    loss: Tensor
    loss_numel: int
    

def load_clip_to_cpu(backbone_name, pretrain_dir):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, pretrain_dir)

    try:
        model = torch.jit.load(model_path, map_location='cpu').eval()
        stat_dict = None

    except RuntimeError:
        stat_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model(stat_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, backbone_name, pretrain_dir):
        super().__init__()
        clip_model = load_clip_to_cpu(backbone_name, pretrain_dir)
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding
        del clip_model

    def forward(self, text):
        text = clip.tokenize(text)
        # print(text.shape)
        x = self.token_embedding(text.to(self.text_projection.device)).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        # print(x.shape)
        return x
    
class OTE_Decoder(nn.Module):
    def __init__(self, max_label_length: int, embed_dim: int) -> None:
        super().__init__()
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
        # print(x_feat.shape)
        if len(x_feat.shape)==2:
            x_feat = x_feat.unsqueeze(1)
        B, N, C = x_feat.shape
        reading_order = torch.arange(self.max_label_length).unsqueeze(0).tile(
            [B, 1]).to(x_feat.device)
        reading_order_embed = self.position_embedding_cpa(reading_order)
        x_att = torch.tanh(self.fc1(x_feat) + reading_order_embed)
        x_att = self.softmax(self.fc2(x_att))
        x = self.fc3(x_att * x_feat)
        x = self.norm(x)
        return x
    
    
class CLIP_Seq(CrossEntropySystem):
    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,img_size: Sequence[int], patch_size: Sequence[int], embed_dim: int, 
                 enc_num_heads: int, enc_mlp_ratio: int, enc_depth: int, dropout: float, is_tps=False,
                 clip_name='ViT-B/32', pretrain_dir = 'clip', **kwargs: Any) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.save_hyperparameters()
        self.max_label_length = max_label_length
        self.clip_text_encoder = TextEncoder(clip_name, pretrain_dir)
        self.decoder = OTE_Decoder(max_label_length=max_label_length + 1, embed_dim=self.clip_text_encoder.transformer.width)
        self.head = nn.Linear(self.clip_text_encoder.transformer.width, len(self.tokenizer) - 2)
        self.model = Model(max_label_length = max_label_length+2, img_size=img_size, patch_size=patch_size, depth=12, mlp_ratio=4, 
                           qkv_bias=True, embed_dim=embed_dim, enc_num_heads=enc_num_heads, num_classes=len(self.tokenizer) - 2,
                           enc_mlp_ratio = enc_mlp_ratio, enc_depth = enc_depth, dropout = dropout, is_tps=is_tps, **kwargs)
        self.fc = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                nn.Linear(embed_dim, self.clip_text_encoder.transformer.width),                             nn.LayerNorm(self.clip_text_encoder.transformer.width)) 
        self.model.head.apply(init_weights)
        self.fc.apply(init_weights)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # for name, parameter in self.clip_text_encoder.named_parameters():
        #      parameter.requires_grad = True
        

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'model.' + n for n in self.model.no_weight_decay()}
    
    def forward(self, images: Tensor, max_length=None) -> Tensor:
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        logits_I, I_feat = self.model.forward(images)
        logits_I = logits_I[:,:max_length+1,:]
        return logits_I, I_feat
    
    

    def forward_logits_loss(self, images, labels):
        targets = self.tokenizer.encode(labels, self.device)
        targets = targets[:, 1:]  # Discard <bos>
        max_len = targets.shape[1] - 1
        logits, I_feat = self.forward(images, max_len)
        I_feat = self.fc(I_feat).squeeze()
        with torch.no_grad():
            T_feat = self.clip_text_encoder(labels)
        # print(I_feat.shape,T_feat.shape)
        I_feat = I_feat / I_feat.norm(dim=1, keepdim=True)
        T_feat = T_feat / T_feat.norm(dim=1, keepdim=True)
        similarity_matrix =  self.logit_scale * torch.matmul(I_feat, T_feat.T)
        i2t_loss = F.cross_entropy(similarity_matrix, torch.arange(len(T_feat)).to(I_feat.device))
        loss_ce = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.pad_id)
        # print(loss_ce.item(),i2t_loss.item())
        loss = loss_ce + i2t_loss/2
        self.log('loss', loss)
        self.log('loss_ce', loss_ce)
        self.log('loss_i2t', i2t_loss)
        loss_numel = (targets != self.pad_id).sum()
        return logits, loss, loss_numel
    
    def _eval_step(self, batch, validation: bool) -> Optional[STEP_OUTPUT]:
        images, labels = batch

        correct = 0
        total = 0
        ned = 0
        confidence = 0
        label_length = 0
        if validation:
            logits, loss, loss_numel = self.forward_logits_loss(images, labels)
        else:
            logits, _ = self.forward(images)
            loss = loss_numel = None 
        probs = logits.softmax(-1)
        print(probs.shape)
        preds, probs = self.tokenizer.decode(probs)
        for pred, prob, gt in zip(preds, probs, labels):
            confidence += prob.prod().item()
            pred = self.charset_adapter(pred)
            # Follow ICDAR 2019 definition of N.E.D.
            ned += edit_distance(pred, gt) / max(len(pred), len(gt))
            if pred == gt:
                correct += 1
            total += 1
            label_length += len(pred)
        return dict(output=BatchResult(total, correct, ned, confidence, label_length, loss, loss_numel))

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        loss = self.forward_logits_loss(images, labels)[1]
        return loss  
        
        