from typing import Sequence, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F 
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from strhub.models.base import CrossEntropySystem
from strhub.models.utils import init_weights
from .model import OTE as Model
from strhub.models.stn import STN

class CenterLoss(nn.Module):
    def __init__(self, class_num, feature_dim):
        super().__init__()
        self.class_num = class_num
        self.feat_dim = feature_dim
        self.center = nn.Parameter(torch.randn(self.class_num,self.feat_dim))
        
    def forward(self, x, labels):
        # labels = F.one_hot(labels, num_classes=self.class_num)
        # print(labels.shape,x.shape)
        # print(torch.max(labels),self.center.shape)
        # import pdb
        # pdb.set_trace()
        center_exp = self.center.index_select(dim=0, index=labels.long()) # [N, 2]
        count = torch.histc(labels.float(), bins=self.class_num, min=0, max=self.class_num-1) # [10]
        count_exp = count.index_select(dim=0, index=labels.long())+1 # [N]
        loss = torch.mean(torch.div(torch.sqrt(torch.sum(torch.pow(x - center_exp,2), dim=1)), count_exp)) # 求损失, 原公式
        # loss = torch.mean(torch.div(torch.sum(torch.pow(x - center_exp,2), dim=1), 2*count_exp)) # 求损失，略不同
        return loss 


class OTESeq(CrossEntropySystem):
    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], patch_size: Sequence[int], embed_dim: int, 
                 enc_num_heads: int, enc_mlp_ratio: int, enc_depth: int, dropout: float, is_tps=False, use_closs=False, **kwargs: Any) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.use_closs = use_closs
        print(self.bos_id ,self.eos_id ,self.pad_id)
        if self.use_closs:
            self.centerloss = CenterLoss(len(self.tokenizer), embed_dim)
        self.save_hyperparameters()
        self.max_label_length = max_label_length
        self.model = Model(max_label_length = max_label_length+2, img_size=img_size, patch_size=patch_size, depth=12, mlp_ratio=4, 
                           qkv_bias=True, embed_dim=embed_dim, enc_num_heads=enc_num_heads, num_classes=len(self.tokenizer) - 2,
                           enc_mlp_ratio = enc_mlp_ratio, enc_depth = enc_depth, dropout = dropout, is_tps=is_tps, **kwargs)
        self.model.head.apply(init_weights)
        
        
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'model.' + n for n in self.model.no_weight_decay()}
    
    def forward(self, images: Tensor, max_length=None) -> Tensor:
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        logits = self.model.forward(images)  # +2 tokens for [GO] and [s]
        logits = logits[:,:max_length+1,:]
        # Truncate to conform to other models. [GO] in ViTSTR is actually used as the padding (therefore, ignored).
        # First position corresponds to the class token, which is unused and ignored in the original work.
        return logits
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        loss = self.forward_logits_loss(images, labels)[1]
        return loss  
    
    def forward_logits_loss(self, images, labels):
        targets = self.tokenizer.encode(labels, self.device)
        targets = targets[:, 1:]  # Discard <bos>
        max_len = targets.shape[1] - 1  # exclude <eos> from count
        max_len = min(max_len, self.max_label_length)
        x_feat = self.model.forward_without_head(images)
        logits = self.model.head(x_feat)[:,:max_len+1,:]
        if self.use_closs:
            loss_ct = self.centerloss(x_feat[:,:max_len+1,:].flatten(end_dim=1),targets.flatten())
            loss_ce = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.pad_id)
            loss_numel = (targets != self.pad_id).sum()
            # print(loss_ce.item(),loss_ct.item())
            loss = loss_ct + loss_ce
            self.log('loss', loss)
            
        else:
            loss = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.pad_id)
            self.log('loss', loss)
            loss_numel = (targets != self.pad_id).sum()
        return logits, loss, loss_numel
    

    # def forward_logits_loss(self, images: Tensor, labels: List[str]) -> Tuple[Tensor, Tensor, int]:
    #     targets = self.tokenizer.encode(labels, self.device)
    #     targets = targets[:, 1:]  # Discard <bos>
    #     max_len = targets.shape[1] - 1  # exclude <eos> from count
    #     logits = self.forward(images, max_len)
    #     print(logits.shape,target.shape)
    #     loss = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.pad_id)
    #     loss_numel = (targets != self.pad_id).sum()
    #     return logits, loss, loss_numel
