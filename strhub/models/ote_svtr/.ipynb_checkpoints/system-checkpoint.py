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
    def __init__(self, backbone_name, pretrain_dir,output_dim):
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
        return x


class OTE_CLIP_SVTR_PCA(CrossEntropySystem):
    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float, num_heads,
                 img_size: Sequence[int], embed_dim, local_mixer,depth,patch_merging,mlp_ratio, dropout, 
                 mixer=['Local'] * 8 + ['Global'] * 12,
                 is_tps=False, clip_name='ViT-B/32', pretrain_dir = 'clip', class_token=False, is_ar=False, 
                 **kwargs: Any) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.save_hyperparameters()
        self.max_label_length = max_label_length
        self.model = Model(num_classes=len(self.tokenizer) - 2, 
                           max_label_length = max_label_length+2, 
                           img_size=img_size, 
                           embed_dim = embed_dim,
                           vocab_size = len(self.tokenizer),
                           local_mixer = local_mixer,
                           depth=depth, 
                           mixer=mixer,
                           num_heads = num_heads,
                           patch_merging=patch_merging,
                           dropout = dropout,
                           mlp_ratio=mlp_ratio,
                           is_tps=is_tps, 
                           class_token=class_token, **kwargs)

        self.is_ar = is_ar
        print('is_ar:',is_ar)
        self.model.head.apply(init_weights)
        # self.fc.apply(init_weights)
        self.class_token = class_token
    
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
                torch.full((L+1, L+1),
                        float('-inf'),
                        device=self._device), 1)
            # tgt_padding_mask = (tgt_out == self.pad_id)
            tgt_padding_mask = (tgt_out == self.pad_id) | (tgt_out == self.eos_id)
            tgt_padding_mask = torch.cat([torch.full((targets.shape[0],1),False,device=tgt_padding_mask.device),tgt_padding_mask], dim=1)
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
        logits = self.model.decode(x_feat,tgt_in,attn_mask,tgt_padding_mask,max_len=L)
        logits = self.model.head(logits)

        # print(tgt_in[:2],logits.argmax(-1)[:2],tgt_out[:2])
        # x_feat = self.fc(x_feat).squeeze(-1)
        # with torch.no_grad():
        #     clip_feat = self.clip_text_encoder(labels)
        # loss_text = F.l1_loss(x_feat.flatten(), clip_feat.flatten())
        loss = F.cross_entropy(logits.flatten(end_dim=1), tgt_out.flatten(), ignore_index=self.pad_id)
        # loss = loss_ce + loss_text

        self.log('loss', loss)

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
                torch.full((num_steps+1,num_steps+1),
                        float('-inf'),
                        device=self._device), 1)
            
            # query_mask = self.get_query_mask(num_steps+1,self._device) 
            # tgt_padding_mask = (tgt_in == self.pad_id) 
            
            
            tgt_in = torch.full((bs, num_steps), self.pad_id, dtype=torch.long, device=self._device)
            tgt_in[:, 0] = self.bos_id
            
            logits = []
            
            for i in range(num_steps):
                j = i + 1 
                tgt_out = self.model.decode(x_feat,tgt_in[:,:j],attn_mask[:j+1,:j+1])
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
            # print(max_length,query_mask.shape)
            logits = self.model.decode(x_feat,tgt_in,attn_mask,max_len=max_length)
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
        # print(loss_ce.item(),loss_text.item())
        self.log('loss', loss)
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
            if pred == gt:
                correct += 1
            else:
                with open('result.txt','a') as f:
                    f.write(f'{pred}  {gt} \n')
            total += 1
            label_length += len(pred)
        return dict(output=BatchResult(total, correct, ned, confidence, label_length, loss, loss_numel))
