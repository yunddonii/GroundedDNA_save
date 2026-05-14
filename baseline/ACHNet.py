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

class ModelWithClassifier(nn.Module):
    def __init__(self, backbone_with_encoder:Module, code_length:int, num_classes:int) -> None:
        super().__init__()
        self.backbone_with_encoder = backbone_with_encoder
        self.classifier = nn.Linear(code_length, num_classes)
        
    def forward(self, img):
        output = self.backbone_with_encoder(img)
        continuous_code = output['continuous_code']
        classifier_result = self.classifier(continuous_code)
        
        return continuous_code, classifier_result
    
    


def cen_con_loss(hashcode, center, label, t=1.0):
    cos_sim = F.cosine_similarity(hashcode.unsqueeze(1), center.unsqueeze(0), dim=2)

    positives = (torch.exp(cos_sim * t) * label)
    denominator = torch.exp(cos_sim * t) * (1 - label)
    loss = -torch.log(torch.sum(positives, dim=1) / torch.sum(denominator, dim=1))
    loss = torch.mean(loss)

    return loss

def generate_hash_center(model, dataloader, device, code_length, n_class):
    """
    Generate hash_center
    Args
        dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.
    Returns
        code(torch.Tensor): hash_center.
    """
    model.eval()
    with torch.no_grad():
        hash_center = torch.zeros([n_class, code_length], device=device)
        counter = torch.zeros([n_class], device=device)
        for batch in dataloader:
            data, targets = batch['img'].to(device), batch['label'].to(device)
            hash_code, _ = model(data)
            index = targets.argmax(-1)

            for center_idx in index.unique():
                data_idx, = torch.where(index == center_idx)
                hash_center[center_idx] +=  hash_code[data_idx, :].sum(dim=0)
                counter[center_idx] += len(data_idx) 

        for k in range(n_class):
            hash_center[k] = hash_center[k] / counter[k]
            
            
    torch.cuda.empty_cache()
    
    return hash_center


class ACHNet(DeepHashBase):
    def _get_default_config_dict(self) -> dict:
        config = {
                    "lamb"           : 0.2,
                    "batch_size"    : 16,
                    "max_epoch"     : 200,         # T_in
                    "learning_rate" : 1e-4,
                    "optimizer_name": "adam",
                    "lr_scheduler"  : "none",
                }
        return config
    
    def _get_config_dict_for_dataset(self, default_config: dict, dataset: str) -> dict:
        return default_config
    
    def _get_fixed_config_dict(self) -> dict:
        config = {
                "finetune"          : False,
                "gen_code_method"   : "sign",
                "encoder_type": "cafe",
                }
        return config

    def _add_model_specific_args_into_parser(self, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument("--lamb", type=float, default=0.2,  help="balance between CE loss and contrasive loss. (default: %(default)s)")
        return parser
    
    def _train_model(self, backbone_with_encoder: Module, optimizer: Optimizer, scheduler: _LRScheduler, trainset: Dataset, testset: Dataset, dbset: Dataset, model_dir: str, result_dir: str, device: str, batch_size: int, eval_period: int, config: dict) -> None:
        
        # Data Loader
        train_loader = torch.utils.data.DataLoader( dataset=trainset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
        
        
        if self.is_multi_label_class():
            raise NotImplementedError("ACHNet doesn't support Multi-label Dataset")
        
        n_class = self._get_n_class()
        code_length = config["bit"]

        model = ModelWithClassifier(backbone_with_encoder,  code_length, n_class).to(device)
        
        max_iter = config["max_epoch"]
        lamb = 0.5

        model.train()
        
        for it in range(max_iter):
            generate_hash_center_start = time.time()
            hash_center = generate_hash_center(model, train_loader, device, code_length, n_class)
            generate_hash_center_end = time.time()
            print('iter[{}/{}] generate_hash_center time {:.3f}'.format(it, max_iter, generate_hash_center_end - generate_hash_center_start))

            model.train()

            for batch in train_loader:

                data, targets = batch['img'].to(device), batch['label'].to(device)

                optimizer.zero_grad()

                hash_center = hash_center.to('cuda')
                
                hashcodes, classifier_result = model(data)

                loss_ce = F.cross_entropy(classifier_result, targets.argmax(dim=-1))
                loss_con = cen_con_loss(hashcodes, hash_center, targets)
                
                loss = lamb * loss_ce + loss_con

                loss.backward()
                optimizer.step()
                self._insert_iter_loss_sum({"ce_loss": loss_ce.item(), "con_loss": loss_con.item()}, current_batch_size= data.shape[0])

            # update step
            scheduler.step()
            self._compute_loss_per_epoch()
            
            model_name = f"{it:03d}"
            self._save_train_model_log(model_name, "epoch", True)
        

            if (it + 1) % eval_period == 0:
                self.start_eval_process(model_name, "epoch")
                self._save_train_model_params(model_name, "epoch")

