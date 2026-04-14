from typing import Sequence, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F 
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from nltk import edit_distance
from dataclasses import dataclass

from strhub.models.base import CrossEntropySystem
from strhub.models.utils import init_weights
from .model import OTE_CLIP as Model
from strhub.models.stn import STN

from .clip import clip
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

# class ContrastiveCenterLoss(nn.Module):
#     def __init__(self, dim_hidden, num_classes, lambda_c=1.0):
#         super(ContrastiveCenterLoss, self).__init__()
#         self.dim_hidden = dim_hidden
#         self.num_classes = num_classes
#         self.lambda_c = lambda_c
#         self.centers = nn.Parameter(torch.randn(num_classes, dim_hidden))
        

#     # may not work due to flowing gradient. change center calculation to exp moving avg may work.
#     def forward(self, hidden,y):
#         batch_size = hidden.size()[0]
#         expanded_centers = self.centers.expand(batch_size, -1, -1)
#         expanded_hidden = hidden.expand(self.num_classes, -1, -1).transpose(1, 0)
#         distance_centers = (expanded_hidden - expanded_centers).pow(2).sum(dim=-1)
#         distances_same = distance_centers.gather(1, y.unsqueeze(1))
#         intra_distances = distances_same.sum()
#         inter_distances = distance_centers.sum().sub(intra_distances)
#         epsilon = 1e-6
#         loss = (self.lambda_c / 2.0 / batch_size) * intra_distances / \
#                (inter_distances + epsilon) / 0.1
#         return loss

# import torch.nn.functional as F

# def ContrastiveLoss(hidden, y, margin=1):
#     # 计算所有向量的范数
#     norms = hidden.norm(p=2, dim=1, keepdim=True)
#     norm_matrix = torch.mm(norms, norms.t())
    
#     # 计算点积
#     dot_product = torch.mm(hidden, hidden.t())
    
#     # 使用公式计算余弦相似度
#     cosine_similarity = dot_product / norm_matrix
    
#     pairwise_distances = 1.0 - cosine_similarity
    
#     label_matrix = y.unsqueeze(1) == y.unsqueeze(0)
    
#     positive_pairs = pairwise_distances * label_matrix.float()
#     positive_loss = positive_pairs.pow(2).sum()

#     negative_pairs = (margin - pairwise_distances) * (~label_matrix).float()
#     negative_loss = F.relu(negative_pairs).pow(2).sum()

#     loss = 0.5 * (positive_loss + negative_loss) / y.size(0)
#     return loss


# def ContrastiveLoss(hidden, y, margin=1):
#     pairwise_distances = 1.0 - F.cosine_similarity(hidden.unsqueeze(1), hidden.unsqueeze(0), dim=2)
#     # pairwise_distances = 1.0 - cosine_similarity
#     print(pairwise_distances.shape)
#     label_matrix = y.unsqueeze(1) == y.unsqueeze(0)
    
#     positive_pairs = pairwise_distances * label_matrix.float()
#     positive_loss = positive_pairs.pow(2).sum()

#     negative_pairs = (margin - pairwise_distances) * (~label_matrix).float()
#     negative_loss = F.relu(negative_pairs).pow(2).sum()
#     # print(negative_loss,positive_loss)
#     loss = 0.5 * (positive_loss + negative_loss) / y.size(0)
#     return loss

# def ContrastiveLoss(hidden, y, margin=1):
#     hidden = F.normalize(hidden,dim=-1)
#     pairwise_distances =1.0 - hidden @ hidden.T 
#     label_matrix = y.unsqueeze(1) == y.unsqueeze(0)
    
    
#     positive_pairs = pairwise_distances * label_matrix.float()
#     positive_loss = positive_pairs.pow(2).sum()

#     negative_pairs = (margin - pairwise_distances) * (~label_matrix).float()
#     negative_loss = F.relu(negative_pairs).pow(2).sum()
#     # print(negative_loss,positive_loss)
#     loss = 0.5 * (positive_loss + negative_loss) / y.size(0)
#     return loss
    
def ContrastiveLoss(hidden, y, margin=1):
    hidden = F.normalize(hidden,dim=-1)
    sim_martrix =hidden @ hidden.T 
    # sim_martrix = F.cosine_similarity(hidden.unsqueeze(1), hidden.unsqueeze(0), dim=-1)
    label_matrix = y.unsqueeze(1) == y.unsqueeze(0)
    loss = F.binary_cross_entropy_with_logits(sim_martrix,label_matrix.float())
    
    return loss

class ContrastiveLoss_center(nn.Module):
    def __init__(self, dim_hidden, num_classes, lambda_c=1.0):
        super(ContrastiveLoss_center, self).__init__()
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.centers = nn.Parameter(torch.randn(num_classes, dim_hidden))
    
    def forward(self,hidden,y):
        hidden = F.normalize(hidden,dim=-1)
        centers = F.normalize(self.centers,dim=-1)
        sim_martrix = hidden @ centers.T
        label_martrix = F.one_hot(y,num_classes = self.num_classes)
        loss = F.binary_cross_entropy_with_logits(sim_martrix,label_martrix.float())
        return loss
    
class OTESeq(CrossEntropySystem):
    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], patch_size: Sequence[int], embed_dim: int, 
                 enc_num_heads: int, enc_mlp_ratio: int, enc_depth: int, dropout: float, 
                 is_tps=False, clip_name='ViT-B/32', pretrain_dir = 'clip', class_token=True, is_ar=False, **kwargs: Any) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.save_hyperparameters()
        self.max_label_length = max_label_length
        self.model = Model(max_label_length = max_label_length+2, img_size=img_size, patch_size=patch_size, depth=12, mlp_ratio=4, 
                           qkv_bias=True, embed_dim=embed_dim, enc_num_heads=enc_num_heads, num_classes=len(self.tokenizer) - 2,
                           enc_mlp_ratio = enc_mlp_ratio, enc_depth = enc_depth, dropout = dropout, is_tps=is_tps, class_token=class_token, vocab_size = len(self.tokenizer), **kwargs)

        self.is_ar = is_ar
        # self.comtrastive_loss = ContrastiveCenterLoss(embed_dim,num_classes=len(self.tokenizer))
        print('is_ar:',is_ar)
        self.model.head.apply(init_weights)
        # self.fc.apply(init_weights)
        self.class_token = class_token
        # self.contrastive_loss = ContrastiveLoss_center(dim_hidden=embed_dim,num_classes=len(self.tokenizer))
        # self.contrastive_loss = None

    @staticmethod
    def _ensure_finite(name: str, tensor: Optional[Tensor]) -> None:
        if tensor is not None and not torch.isfinite(tensor).all():
            raise RuntimeError(f'Non-finite {name} detected during OTE training.')

    def _fail_on_non_finite_loss(self, loss: Tensor, **tensors: Tensor) -> None:
        if bool(torch.isfinite(loss).item()):
            return
        for name, tensor in tensors.items():
            self._ensure_finite(name, tensor)
        raise RuntimeError('Non-finite loss detected with finite intermediate tensors.')

    def get_query_mask(self,n,device):
        matrix = torch.full((n, n), float('-inf'),device=device)
        torch.diagonal(matrix).fill_(0)
        matrix[:, 0] = 0
        return matrix
        
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'model.' + n for n in self.model.no_weight_decay()}
    
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images, labels = batch
        targets = self.tokenizer.encode(labels, self.device)

        tgt_in = targets[:, :-1]
        tgt_out = targets[:, 1:]
        
        # tgt_padding_mask = (tgt_out == self.pad_id) | (tgt_out == self.eos_id)
        # tgt_padding_mask = torch.cat([torch.full((targets.shape[0],1),False,device=tgt_padding_mask.device),tgt_padding_mask], dim=1)
        # tgt_padding_mask = None
        
        B, L = tgt_in.shape
        
        n = (tgt_out != self.pad_id).sum().item()
        if self.is_ar:
            attn_mask = torch.triu(
                torch.full((L, L),
                        float('-inf'),
                        device=self._device), 1)
            # tgt_padding_mask = (tgt_out == self.pad_id)
            tgt_padding_mask = (tgt_out == self.pad_id) | (tgt_out == self.eos_id)
            # tgt_padding_mask = torch.cat([torch.full((targets.shape[0],1),False,device=tgt_padding_mask.device),tgt_padding_mask], dim=1)
        else:
            attn_mask = None
            tgt_in = None
            tgt_padding_mask = None
        
        query_mask = self.get_query_mask(L,self._device)
        # attn_mask = query_mask =None
        
        x_feat = self.model.encode(images)
        
        if self.class_token:
            x_feat = x_feat[:, :1, :]
        else:
            x_feat = F.adaptive_avg_pool1d(x_feat.transpose(2, 1),
                                           1).transpose(2, 1)

        # logits = self.model.decode(x_feat,tgt_in,query_mask,attn_mask,tgt_padding_mask)[:,1:,:]
        logits,hidden = self.model.decode(x_feat,tgt_in,attn_mask,tgt_padding_mask,max_len=L)
        
        logits = self.model.head(logits)

        # print(tgt_in[:2],logits.argmax(-1)[:2],tgt_out[:2])
        # x_feat = self.fc(x_feat).squeeze(-1)
        # with torch.no_grad():
        #     clip_feat = self.clip_text_encoder(labels)
        # loss_text = F.l1_loss(x_feat.flatten(), clip_feat.flatten())
        # loss_constrative = ContrastiveLoss(hidden.flatten(end_dim=1),tgt_out.flatten())
        # loss_constrative = self.contrastive_loss(hidden.flatten(end_dim=1),tgt_out.flatten())
        
        # loss_ce = F.cross_entropy(logits.flatten(end_dim=1), tgt_out.flatten(), ignore_index=self.pad_id)
        # print(loss_constrative)
        # loss = loss_ce + loss_constrative
        # self.log('loss_constrative', loss_constrative)
        # self.log('loss_ce',loss_ce)
        loss = F.cross_entropy(logits.flatten(end_dim=1), tgt_out.flatten(), ignore_index=self.pad_id)
        self._fail_on_non_finite_loss(
            loss,
            images=images,
            encoder_features=x_feat,
            decoder_hidden=hidden,
            logits=logits,
        )
        
        self.log('loss', loss, batch_size=images.shape[0], sync_dist=True)

        loss_numel = (targets != self.pad_id).sum()

        return loss  

    def forward(self, images,max_len=None):
        
        testing = max_len is None
        # print('------testing:{testing}----------')
        max_length = self.max_label_length if max_len is None else min(max_len, self.max_label_length)
        bs = images.shape[0]
        num_steps = max_length 
        x_feat = self.model.encode(images)
        if self.class_token:
            x_feat = x_feat[:, :1, :]
        else:
            x_feat = F.adaptive_avg_pool1d(x_feat.transpose(2, 1),
                                           1).transpose(2, 1)
        if self.is_ar:
            attn_mask = torch.triu(
                torch.full((num_steps,num_steps),
                        float('-inf'),
                        device=self._device), 1)
            
            # query_mask = self.get_query_mask(num_steps+1,self._device) 
            
            tgt_in = torch.full((bs, num_steps), self.pad_id, dtype=torch.long, device=self._device)
            tgt_in[:, 0] = self.bos_id
            
            logits = []
            
            for i in range(num_steps):
                j = i + 1 
                tgt_out,_ = self.model.decode(x_feat,tgt_in[:,:j],attn_mask[:j,:j])
                p_i = self.model.head(tgt_out)[:,-1:,:]
                logits.append(p_i)
                if j < num_steps:
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)
                    if testing and (tgt_in == self.eos_id).any(dim=-1).all():
                        break
            logits = torch.cat(logits,dim=1)
        else: 
            attn_mask = None
            tgt_in = None
            # query_mask = self.get_query_mask(num_steps,self._device)
            # query_mask 
            # print(max_length,query_mask.shape)
            logits,_ = self.model.decode(x_feat,tgt_in,attn_mask,max_len=max_length)
            logits = self.model.head(logits)
        return logits , x_feat
        

    def forward_logits_loss(self, images, labels):
        targets = self.tokenizer.encode(labels, self.device)
        targets = targets[:, 1:]  # Discard <bos>
        max_len = targets.shape[1]  # exclude <eos> from count
        logits, x_feat = self.forward(images, max_len)
        
        
        # x_feat = self.fc(x_feat).squeeze(-1)
        # with torch.no_grad():
        #     clip_feat = self.clip_text_encoder(labels)
        # print(x_feat.shape,clip_feat.shape)
        # loss_text = F.l1_loss(x_feat.flatten(), clip_feat.flatten())
        loss_ce = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.pad_id)
        loss = loss_ce
        self._fail_on_non_finite_loss(
            loss,
            images=images,
            encoder_features=x_feat,
            logits=logits,
        )
        # print(loss_ce.item(),loss_text.item())
        self.log('loss', loss, batch_size=images.shape[0], sync_dist=True)
        # self.log('loss_ce', loss_ce)
        # self.log('loss_text', loss_text)
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
            # At test-time, we shouldn't specify a max_label_length because the test-time charset used
            # might be different from the train-time charset. max_label_length in eval_logits_loss() is computed
            # based on the transformed label, which could be wrong if the actual gt label contains characters existing
            # in the train-time charset but not in the test-time charset. For example, "aishahaleyes.blogspot.com"
            # is exactly 25 characters, but if processed by CharsetAdapter for the 36-char set, it becomes 23 characters
            # long only, which sets max_label_length = 23. This will cause the model prediction to be truncated.
            logits, _ = self.forward(images)
            loss = loss_numel = None  # Only used for validation; not needed at test-time.

        probs = logits.softmax(-1)
        preds, probs = self.tokenizer.decode(probs)
        for pred, prob, gt in zip(preds, probs, labels):
            confidence += prob.prod().item()
            pred = self.charset_adapter(pred)
            # Follow ICDAR 2019 definition of N.E.D.
            ned += edit_distance(pred, gt) / max(len(pred), len(gt))
            with open('result_v2.txt','a') as f:
                f.write(f'{pred}  {gt} \n')
            if pred == gt:
                correct += 1
            # else:
            #     with open('result.txt','a') as f:
            #         f.write(f'{pred}  {gt} \n')
            total += 1
            label_length += len(pred)
        return dict(output=BatchResult(total, correct, ned, confidence, label_length, loss, loss_numel))
