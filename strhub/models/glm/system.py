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

from typing import Sequence, Any, Optional

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
import torch.nn.functional as F

from strhub.models.base import CrossEntropySystem
from strhub.models.utils import init_weights
from .model import ViTSTR as Model
from typing import Optional, Tuple, List
import math
import numpy as np

class GLMSeq(CrossEntropySystem):

    def __init__(self, charset_train: str, charset_test: str, max_label_length: int,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float,
                 img_size: Sequence[int], patch_size: Sequence[int], embed_dim: int, num_heads: int,
                 **kwargs: Any) -> None:
        super().__init__(charset_train, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.save_hyperparameters()
        self.max_label_length = max_label_length
        # We don't predict <bos> nor <pad>
        self.model = Model(img_size=img_size, patch_size=patch_size, depth=12, mlp_ratio=4, qkv_bias=True,
                           embed_dim=embed_dim, num_heads=num_heads, num_classes=len(self.tokenizer) - 2,word_length=max_label_length)
        # Non-zero weight init for the head
        self.model.head.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'model.' + n for n in self.model.no_weight_decay()}

    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        max_length = self.max_label_length if max_length is None else min(max_length, self.max_label_length)
        logits = self.model.forward(images)  #
        logits = logits[:, :max_length, :]
        return logits
    
    def forward_logits_loss(self, images: Tensor, labels: List[str],ratio=0) -> Tuple[Tensor, Tensor, int]:
        targets = self.tokenizer.encode(labels, self.device)
        targets = targets[:, 1:]  # Discard <bos>
        max_len = targets.shape[1]   # exclude <eos> from count
        # print(targets[0])
        _, x_vision, logits = self.model.forward_with_mask(images, mask_ration=ratio)
        # print(loss_recons)
        logits = logits[:, :max_len, :]
        loss = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.pad_id)
        loss_numel = (targets != self.pad_id).sum()
        # self.log('loss_recons', loss_recons)
        return logits, loss, loss_numel
    
    def get_ratio(self, current_step, total_steps, a=0.75, b=0.1):
            mid_point = total_steps / 2
            amplitude = (a - b) / 2
            mid_value = (a + b) / 2

            ratio = mid_value + amplitude * np.cos(np.pi * (current_step - mid_point) / mid_point)
            return ratio

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        global_step = self.global_step
        total_steps = len(self.trainer.train_dataloader)*self.trainer.max_epochs
        # ratio=self.get_ratio(global_step,total_steps)
        ratio = 0
        # print(global_step,total_steps,self.trainer.num_training_batches)
        # print(batch_idx)
        # print(self.trainer.__dict__)
        images, labels = batch
        loss = self.forward_logits_loss(images, labels,ratio=ratio)[1]
        self.log('loss', loss)
        self.log('ratio', ratio)
        return loss
    
if __name__ =="__name__":
    pass
