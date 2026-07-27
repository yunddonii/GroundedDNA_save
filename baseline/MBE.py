"""Legacy, unregistered Bi-half prototype; do not use for comparisons.

This historical file predates the cached-feature runner and does not implement
the AAAI release proxy gradient or inference contract.  The audited baseline is
``baseline/BiHalf.py`` and its only registered CLI key is ``--method bihalf``.
The prototype is retained solely to avoid deleting prior repository work.
"""

import argparse
import torch
from torch import nn
from torch.autograd import Function
from torch.nn import Module
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from argparse import ArgumentParser, BooleanOptionalAction

from lib.quantizations import rank_and_assign, convert_b2bq_with_seq_r_constrants
from .base_model import DeepHashBase


class BiHalf(Function):
    @staticmethod
    def forward(ctx, U, grad_tanh):
        # Yunqiang for half and half (optimal transport)
        B = rank_and_assign(U, axis=0)
        ctx.grad_tanh = grad_tanh
        return B

    @staticmethod
    def backward(ctx, dL_dB):
        if ctx.grad_tanh:
            dL_dU = torch.tanh(dL_dB) 
        else:
            dL_dU = dL_dB

        return dL_dU, None, None, None
    
    
class BiHalfModule(nn.Module):
    def __init__(self, grad_tanh=False, **kwargs) -> None:
        super().__init__()
        self.grad_tanh = grad_tanh
    
    def forward(self, continuous_code):
        return BiHalf.apply(continuous_code, self.grad_tanh)


class Model(nn.Module):
    def __init__(self, backbone_with_encoder:Module, grad_tanh=False) -> None:
        super().__init__()
        self.backbone_with_encoder = backbone_with_encoder
        self.bihalf = BiHalfModule(grad_tanh)
        
    def forward(self, img):
        output = self.backbone_with_encoder(img)
        target_emb = output['backbone_last_output']
        continuous = output['continuous_code']
        binary = self.bihalf(continuous)
        return target_emb, continuous, binary 
    

def compute_target_sim(x, mode="pairwise_cosine"):
    N = x.shape[0] 
    if mode == "pairwise_cosine":
        o_sim = F.cosine_similarity(x[: N // 2], x[N // 2 :  N // 2 * 2])
    else:
        raise NotImplementedError(f"Similarity Computing mode: {mode} is not defined")
    return o_sim
    
def compute_recon_sim(b, mode="pairwise_cosine"):
    # pairwise_cosine / pairwise_cosine_detach / pairwise_cosine_confident_detach
    N = b.shape[0]
    b1 = b[: N // 2]
    b2 = b[N // 2 : N // 2 * 2]
    
    if mode == "pairwise_cosine":
        recon_sim = F.cosine_similarity(b1, b2,)
    else:
        raise NotImplementedError(f"Similarity Computing mode: {mode} is not defined")

    return recon_sim
    
def compute_qt_loss(u, b):
    return F.mse_loss(u,b.detach())  # sum( ( u - b)^2 ) / (batch_size * encode_len)
            
def compute_sim_recon_loss(x, b):
    model_sim = compute_recon_sim(b=b)
    target_sim = compute_target_sim(x=x)
    return F.mse_loss(model_sim, target_sim) 



class MBE(DeepHashBase):
    def _get_default_config_dict(self) -> dict:
        default = {
                    "max_epoch": 200,
                    "optimizer_name": "adam",
                    "lr_scheduler": 'none',
                    "learning_rate": 0.0001,
                    "batch_size": 32, 
                    "grad_tanh": False,
                    "alpha":  3,
                }
        return default
    
    def _get_config_dict_for_dataset(self, default_config: dict, dataset: str) -> dict:
        return default_config

    def _get_fixed_config_dict(self) -> dict:
        fixed ={
                    "gen_code_method"   : "sign",
                    "finetune": False,
                    "dataset_return_index" : False,
                    "dataset_return_paired_aug_img" : False,
                }

        return fixed
    
    def _add_model_specific_args_into_parser(self, parser: ArgumentParser) -> ArgumentParser:
        # Hyper Parameters
        parser.add_argument("--grad_tanh", default=False, type=str,  action=BooleanOptionalAction,)
        parser.add_argument("--alpha", default=3, type=float, help="(default: %(default)s)")
        return parser
    
    def _train_model(self, backbone_with_encoder: Module, optimizer: Optimizer, scheduler: _LRScheduler, trainset: Dataset, testset: Dataset, dbset: Dataset, model_dir: str, result_dir: str, device: str, batch_size: int, eval_period: int, config: dict) -> None:
        
        # Data Loader
        train_loader = torch.utils.data.DataLoader( dataset=trainset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
        
        model = Model(backbone_with_encoder).to(device)

        model.train()
        
        for epoch in range(config["max_epoch"]):
            for batch in train_loader:
                image = batch['img'].to(device)
                                
                target_emb, continuous_code, binary_code = model(image)

                recon_loss = compute_sim_recon_loss(target_emb, binary_code)
                
                qt_loss = compute_qt_loss(continuous_code, binary_code)
                
                loss = recon_loss + config['alpha'] * qt_loss 
                
                self._insert_iter_loss_sum({"recon_loss": recon_loss.item(), 'qt_loss': qt_loss.item()}, current_batch_size= image.shape[0])
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            scheduler.step()
                
            self._compute_loss_per_epoch()
            
            model_name = f"{epoch:03d}"
            self._save_train_model_log(model_name, "epoch", True)
        

            if (epoch + 1) % eval_period == 0:
                self.start_eval_process(model_name, "epoch")
                self._save_train_model_params(model_name, "epoch")
  
