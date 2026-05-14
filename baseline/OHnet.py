from argparse import ArgumentParser, BooleanOptionalAction
import torch
from torch.nn import Module
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from .base_model import DeepHashBase


import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time

from copy import deepcopy
import os

class Queue(nn.Module):
    """this is for memory bank
        """
    def __init__(self, length, dim):
        super(Queue, self).__init__()
        self.dict = nn.Parameter(torch.randn(length, dim), requires_grad=False) # 4096 x 512

    def update_queue(self, inputs):
        n = inputs.size(0)
        update_dict = torch.cat([inputs, self.dict[:-n]], dim=0) # 
        self.dict.data.copy_(update_dict)
        
        
class Encoder(nn.Module):
    def __init__(self, backbone_with_encoder:Module, middle_dim, c_dim=512):
        super(Encoder, self).__init__()
        self.backbone_with_encoder = backbone_with_encoder # (backbone) + (fc, b)
        self.layer_c = nn.Linear(middle_dim, c_dim)

    def forward(self, inputs):
        out = self.backbone_with_encoder(inputs)
        
        feature = out['encoder_block_outputs'][0] # [batch_size, 4096]
        
        b = torch.sigmoid(out['continuous_code']) # z_b = [batch_size, 16]
        b = self.grad_through(b)
        
        c = self.layer_c(feature) # layer_c = [4096, 512] z_c = [batch_size, 512]
        c = F.normalize(c, p=2, dim=-1)
        
        return b, c

    def grad_through(self, b):
        device = b.device
        binary_hash = torch.where(b > 0.5, torch.tensor(1., device=device), torch.tensor(0., device=device))
        return b + (binary_hash - b).detach() 

# Example usage:
# encoder = Encoder(middle_dim=512, b_dim=128, c_dim=256)
# input_tensor = torch.randn((1, 3, 224, 224))  # Assuming input shape (batch_size, channels, height, width)
# output_b, output_c = encoder(input_tensor)


class OH(nn.Module):

    def __init__(self, backbone_with_encoder: Module, middle_dim, b_dim, c_dim=512, m=0.999, T=0.2, queue_length=4096):
        super(OH, self).__init__()
        self.middle_dim = middle_dim
        self.c_dim = c_dim
        self.b_dim = b_dim
        self.m = m
        self.T = T

        # create the encoders
        self.encoder_q = Encoder(backbone_with_encoder, middle_dim, c_dim)
        self.encoder_k = Encoder(deepcopy(backbone_with_encoder), middle_dim, c_dim)

        # create the queue
        self.P_Queue = Queue(queue_length, c_dim)
        self.B_Queue = Queue(queue_length, b_dim)
        self.C_Queue = Queue(queue_length, c_dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)

        self.softmax = nn.Softmax(dim=-1)

    def update_queue(self, p, b, c):
        self.P_Queue.update_queue(p)
        self.B_Queue.update_queue(b)
        self.C_Queue.update_queue(c)

    # self-attention
    def contrastive(self, b, c):
        sim = torch.matmul(b, self.B_Queue.dict.t()) + torch.matmul(1 - b, 1 - self.B_Queue.dict.t())
        p = F.normalize(torch.matmul(self.softmax(sim / self.b_dim / self.T), self.C_Queue.dict), p=2, dim=-1) + 0.5 * c
        p = F.normalize(p, p=2, dim=-1)
        return p

    def call_key_encoder(self, inputs):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            update_k = param_k.data * self.m + param_q.data * (1. - self.m)
            param_k.data.copy_(update_k)

        # idx_shuffle = torch.randperm(inputs.size(0))
        # inputs = inputs[idx_shuffle]

        b, c = self.encoder_k(inputs)

        # _, idx_unshuffle = torch.sort(idx_shuffle)
        # b = b[idx_unshuffle]
        # c = c[idx_unshuffle]

        k = self.contrastive(b, c) # r_hat

        return k.detach(), b.detach(), c.detach()

    def call_query_encoder(self, inputs):
        b, c = self.encoder_q(inputs)
        q = self.contrastive(b, c) # r
        return q, b, c

    def forward(self, im_q, im_k):
        
        q, b_query, c_query = self.call_query_encoder(im_q)
        k, b_key, c_key = self.call_key_encoder(im_k)

        l_pos = torch.einsum('nc,nc->n', q, k).unsqueeze(-1) # positive similarity
        l_neg = torch.einsum('nc,kc->nk', q, self.P_Queue.dict) # 
        logits1 = torch.cat([l_pos, l_neg], dim=-1)
        logits1 /= self.T
        labels1 = torch.zeros(logits1.size(0)).long()

        # dequeue and enqueue
        self.update_queue(k, b_key, c_key)

        q2, b_query2, c_query2 = self.call_query_encoder(im_k)
        q2 = q2.detach()
        logits2 = torch.einsum('nc,kc->nk', q, q2)
        logits2 /= self.T
        labels2 = torch.arange(logits2.size(0)).long()

        return logits1, labels1, logits2, labels2

# Example usage:
# ohnet = OHNet(input_shape=(3, 224, 224), middle_dim=512, b_dim=128, c_dim=256)
# input_tensor_q = torch.randn((1, 3, 224, 224))
# input_tensor_k = torch.randn((1, 3, 224, 224))
# logits1, labels1, logits2, labels2 = ohnet(input_tensor_q, input_tensor_k)


class OHnet(DeepHashBase):
    def _get_default_config_dict(self) -> dict:
        
        config = {
            "dm" : 512, #1024
            "L" : 4096,
            "mu" : 1/3,
            "batch_size" : 64,
            "max_epoch" : 200,
            "learning_rate" : 1e-5,
            "optimizer_name" : "adam",
            "lr_scheduler" : "none",
            
        }
        return config
    
    def _get_config_dict_for_dataset(self, default_config: dict, dataset: str) -> dict:
        return default_config
    
    def _get_fixed_config_dict(self) -> dict:
        
        fixed_config = {
            "gen_code_method" : "sign_sigmoid_m0.5",
            "dataset_return_index" : True,
            "dataset_return_paired_aug_img" : True,
            "finetune" : False,
            "transform": "Flip+Rot+RandCrop",
            "encoder_type": "linear",
            "encoder_layers": "layer=2+hidden=2048",
            
        }
        return fixed_config

    def _add_model_specific_args_into_parser(self, parser: ArgumentParser) -> ArgumentParser:
        
        parser.add_argument("--dm", default=512, type=int, help="Dimension of continuous head [%(default)s]")
        parser.add_argument("--L", default=4096, type=int, help="Length of memory bank [%(default)s]")
        parser.add_argument("--mu", default=1/3, type=float, help="hyper-parameter in loss [%(default)f]")
        
        return parser
    
    def _train_model(self, backbone_with_encoder: Module, optimizer: Optimizer, scheduler: _LRScheduler, trainset: Dataset, testset: Dataset, dbset: Dataset, model_dir: str, result_dir: str, device: str, batch_size: int, eval_period: int, config: dict) -> None:
        
        # DataLoader
        train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        
        model = OH(backbone_with_encoder, middle_dim=2048, b_dim=config['bit'], c_dim=config["dm"], m=0.999, T=0.2, queue_length=config['L']).to(device)
        
        model.train()
        
        for param in model.parameters():
            param.requires_grad = False
            
        for param in model.encoder_q.backbone_with_encoder.encoder_layers.parameters():
            param.requires_grad = True
        
        for param in model.encoder_q.layer_c.parameters():
            param.requires_grad = True
        
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(config["max_epoch"]):
            
            for batch in train_loader:
                
                # if batch['idx'] >= 50000 / batch_size : break

                logits1, labels1, logits2, labels2 = model(batch['img_tr1'].to(device), batch['img_tr2'].to(device))
                
                loss = criterion(logits1, labels1.to(device)) + criterion(logits2, labels2.to(device))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self._insert_iter_loss_sum({"loss": loss.item()}, current_batch_size= batch['img_tr1'].shape[0])

            self._compute_loss_per_epoch()
            
            model_name = f"{epoch:03d}"
            self._save_train_model_log(model_name, "epoch", True)
        

            if (epoch + 1) % eval_period == 0:
                model.eval()
                self.start_eval_process(model_name, "epoch")
                self._save_train_model_params(model_name, "epoch")



