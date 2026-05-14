from argparse import ArgumentParser, BooleanOptionalAction
import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from .base_model import DeepHashBase
from tqdm import tqdm

import os

os.environ["CUDA_LAUNCH_BLOCKING"]= "1"


## for generating hash centers

from scipy.special import comb, perm  #calculate combination
from itertools import combinations
from scipy.linalg import hadamard  # direct import  hadamrd matrix from scipy
import torch
import numpy as np


import random
from scipy.special import comb, perm  #calculate combination
from itertools import combinations

def gen_hash_center_single_label(num_class, num_bit):
    
    # construction of a hadamard matrix
    # each row / col can be used as a target for a class of images in hashing
    print("generate hash center for single label ------------------------")
    k = num_bit
    check_power_of2 = np.log2(k).is_integer()
    
    if (num_class <= k) and check_power_of2:
        ha = hadamard(k) # hadamard matrix
        hash_targets = torch.from_numpy(ha[:num_class]).float()
        
    elif (num_class > k) and (num_class <= 2*k) and check_power_of2:
        ha = hadamard(k) # hadamard matrix
        ha_2 = np.concatenate((ha, -ha), 0) # can be used as targets for 2*d hash bit
        hash_targets = torch.from_numpy(ha_2[:num_class]).float()
        
    else:
        
        # generating hash targets by sampling from Bernouli distribution
        
        a = np.arange(0, k).tolist() # for sampling 0.5 * K
        b = np.arange(0, num_class).tolist() # for calculating the combinations of 51 num_class
        
        for _ in tqdm(range(10000)):
            hash_targets = torch.zeros([num_class, k])
            
            for i in range(num_class):
                
                ones = torch.ones(k)
                sa = random.sample(a, round(k/2))
                ones[sa] = -1
                hash_targets[i] = ones
            
            com_num = int(comb(num_class, 2))
            c = np.zeros(com_num)
            
            for i in range(com_num):
                i_1 = list(combinations(b, 2))[i][0]
                i_2 = list(combinations(b, 2))[i][1]
                
                TF = torch.sum(hash_targets[i_1] != hash_targets[i_2])
                c[i] = TF
            
            if min(c) >= 20 and np.mean(c) >= 32:
                break
        
    print(f"hash centers shape : {hash_targets.shape}\n")
    
    
    return hash_targets



def gen_hash_center_multi_labels(labels, Hash_center, random_center):
    """label.shape: [B, n_class], Hash_center.shape: [n_class, n_bit].

    NOTE (modified for GroundedDNA): handle rows with NO active label
    (e.g. Flickr25k rows whose 24 concept tags are all zero) by falling back
    to a random {-1, +1} center. The original baseline crashed on those.
    """
    out_centers = []
    rc01 = random_center.clone()
    rc01[rc01 == 0] = -1                                  # {-1, +1}
    for label in labels:
        one_labels = (label == 1).nonzero().squeeze(1)
        if one_labels.numel() == 0:                       # no active labels -> random
            cm = rc01.clone()
        else:
            cm = torch.mean(Hash_center[one_labels], dim=0)
            cm = torch.where(cm > 0, torch.ones_like(cm),
                  torch.where(cm < 0, -torch.ones_like(cm), rc01))
        out_centers.append(cm.view(1, -1))
    return torch.cat(out_centers, dim=0)


def pairwise_loss(outputs1, outputs2, label1, label2, sigmoid_param=1, data_imbalance=1):
    similarity = torch.autograd.Variable(torch.mm(label1.data.float(), label2.data.float().t()) > 0).float()
    dot_product = sigmoid_param * torch.mm(outputs1, outputs2.t())
    exp_product = torch.exp(dot_product)

    exp_loss = (torch.log(1 + exp_product) - similarity * dot_product)
    loss = torch.mean(exp_loss)

    return loss

def adjust_learning_rate(optimizer, epoch, current_lr, multi_lr):
    """Sets the learning rate to the initial LR decayed by 0.7 every 10 epochs.

    NOTE: original CSQ used a 2-group (backbone vs head) optimizer with
    `multi_lr` scaling for the backbone group. Under the GroundedDNA project
    the backbone is frozen cached features, so we collapse to a single group
    and ignore `multi_lr`.
    """
    lr = current_lr * (0.7 ** (epoch // 10))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr

class CSQModel(nn.Module):
    def __init__(self, backbone_with_encoder) -> None:
        super().__init__()
        ## resnet
        ## resent 뒤에 3 fc + tanh
        self.backbone_with_encoder = backbone_with_encoder
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.backbone_with_encoder(x)
        out = x['continuous_code'] 
        out = self.tanh(out)
        
        return out
    
class CSQ(DeepHashBase):
    def _get_default_config_dict(self) -> dict:
        
        config = {
            "batch_size" : 64,
            "max_epoch" : 100,
            "learning_rate" : 0.001,
            "optimizer_name" : "adam",
            "lr_scheduler" : "none",
            "num_iteration" : 60,
            "num_workers" : 4,
            "multi_lr" : 0.01,
            "lambda0" : 1,
            "lambda1" : 0.2,
            "lambda2" : 0.05,
            # "batch_size_hash" : 40,
        }
        
        return config
    
    def _get_config_dict_for_dataset(self, default_config: dict, dataset: str) -> dict:
        return default_config
    
    def _get_fixed_config_dict(self) -> dict:
        # NOTE: original baseline finetuned the CNN backbone end-to-end with a
        # 3-layer encoder head. Under the GroundedDNA project the backbone is
        # the FROZEN cached SigLIP2 features, so we override `finetune=False`
        # and let `encoder_layers='none'` (single linear head, matches DPSH/HashNet).
        fixed_config = {
            "gen_code_method" : "sign",
            "dataset_return_index" : False,
            "dataset_return_paired_aug_img" : False,
            "finetune" : False,
            "transform" : "default",
            "encoder_type" : "linear",
            "encoder_layers" : "none",
        }
        return fixed_config
    
    def _add_model_specific_args_into_parser(self, parser: ArgumentParser) -> ArgumentParser:
        
        parser.add_argument("--num_iteration", default=60, type=int, help="number of iteration")
        parser.add_argument("--num_workers", default=8, type=int, help="number of data loader workers")
        parser.add_argument("--multi_lr", default=0.01, type=float, help="multiplier for leargning rate")
        parser.add_argument("--lambda0", default=1, type=float, help="hyper-parameters 0")
        parser.add_argument("--lambda1", default=0.2, type=float, help="hyper-parameters 1")
        parser.add_argument("--lambda2", default=0.05, type=float, help="hyper-parameters 2")
        # parser.add_argument("--batch_size_hash", default=40, type=int, help="the batch size for training (batch_size must be even in this project)")
        
        return parser
    
    def _train_model(self, backbone_with_encoder: Module, optimizer: Optimizer, scheduler: _LRScheduler, trainset: Dataset, testset: Dataset, dbset: Dataset, model_dir: str, result_dir: str, device: str, batch_size: int, eval_period: int, config: dict) -> None:
        
        # Dataset-specific knobs. Our active datasets: CIFAR10 (single-class)
        # and Flickr25k (multi-label). The legacy dict added ImageNet100/MSCOCO/
        # NUSWIDE; keep them for compat. `total_epoch` is unused here -- we
        # use `config['max_epoch']`.
        is_multi_label = self.is_multi_label_class()
        if config["dataset"] == "ImageNet100":
            data_imbalance = 100; two_loss_epoch = -1; total_epoch = 90
        elif config["dataset"] == "CIFAR10":
            data_imbalance = 1;   two_loss_epoch = -1; total_epoch = 90
        elif config["dataset"] == "MSCOCO":
            data_imbalance = 1;   two_loss_epoch = -1; total_epoch = 90
        elif config["dataset"] == "NUSWIDE":
            data_imbalance = 5;   two_loss_epoch = -1; total_epoch = 90
        elif config["dataset"] == "Flickr25k":
            # Multi-label like NUSWIDE; per-image label cardinality is small.
            data_imbalance = 1;   two_loss_epoch = -1; total_epoch = 90
        else:
            data_imbalance = 1;   two_loss_epoch = -1; total_epoch = 90
            
        num_class = self._get_n_class()

        Hash_center = gen_hash_center_single_label(num_class, config["bit"]).to(device)
        random_center = torch.randint_like(Hash_center[0], 2).to(device)
        
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
        
        model = CSQModel(backbone_with_encoder).to(device)
        # model = torch.nn.DataParallel(model).to(device)
        
        criterion = nn.BCELoss().to(device)
        
        # Backbone (cached SigLIP2) has no trainable params; only the encoder
        # head trains. Drop the original two-group multi_lr setup; use a single
        # Adam group (the parent already built `optimizer` similarly, but we
        # rebuild here to match the original CSQ Adam betas).
        param_list = [
            {'params' : model.backbone_with_encoder.encoder_layers.parameters()}
        ]
        optimizer = torch.optim.Adam(param_list, lr=config["learning_rate"], betas=(0.9, 0.999))
        
        for epoch in range(config["max_epoch"]):
            lr = adjust_learning_rate(optimizer, epoch, config["learning_rate"], config["multi_lr"])
            
            model.train()
            iter_num=0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                image = batch["img"].to(device)
                label = batch["label"].to(device)
                
                
                if is_multi_label:
                    # Flickr25k / MSCOCO / NUSWIDE: per-image label cardinality > 1.
                    _hash_center = gen_hash_center_multi_labels(label, Hash_center, random_center)
                else:
                    # CIFAR10 / ImageNet100: one-hot.
                    hash_label = (label == 1).nonzero()[:, 1]
                    _hash_center = Hash_center[hash_label]
                
                _hash_center = _hash_center.to(device)
                
                out = model(image)
                
                center_loss = criterion(0.5 * (out + 1), 0.5 * (_hash_center + 1))
                Q_loss = torch.mean((torch.abs(out) - 1.0) ** 2)
                
                if epoch <= two_loss_epoch:
                    loss = config["lambda0"] * center_loss + config["lambda2"] * Q_loss
                    
                else:
                    if len(label) < config["batch_size"] : 
                        # if the last batch is not completed, set similarity_loss=0
                        similarity_loss = 0
                    else:
                        output1 = out.narrow(0, 0, int(0.5 * len(out)))
                        output2 = out.narrow(0, int(0.5 * len(out)), int(0.5 * len(out)))
                        
                        label1 = label[:int(0.5 * len(label))].to(device) # [1/2 * batch_size, num_class]
                        label2 = label[int(0.5 * len(label)):int(len(label))].to(device) # [1/2 * batch_size, num_class]
                        
                        similarity_loss = pairwise_loss(output1, output2, label1, label2, 
                                                        sigmoid_param=10./config["bit"], 
                                                        data_imbalance=data_imbalance)
                        
                    loss = config["lambda0"] * center_loss + config["lambda1"] * similarity_loss + config["lambda2"] * Q_loss
                
            
                loss.backward()
                optimizer.step()
                iter_num+=1
                self._insert_iter_loss_sum({"loss" : loss.item()}, current_batch_size=batch['img'].shape[0])
                
            if (epoch+1) % 30 == 0: print(f"learning rate in {epoch}-th epoch = {lr}")
            self._compute_loss_per_epoch()
            
            model_name = f"{epoch:03d}"
            self._save_train_model_log(model_name, "epoch", True)
            
            if (epoch + 1) % eval_period == 0:
                model.eval()
                self.start_eval_process(model_name, "epoch")
                self._save_train_model_params(model_name, "epoch")
            
        
        
        
    