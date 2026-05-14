from argparse import ArgumentParser
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time

from torch.nn import Module
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from .base_model import DeepHashBase

from tqdm import tqdm
from copy import deepcopy
import random

def AdjustLearningRate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def Logtrick(x):
    lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, torch.FloatTensor([0.]).cuda())
    return lt

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def generate_similarity_matrix(train_loader, num_train, model, dim_feature, num_class, k_nn, o_nn, alpha, device):
    
    print("=" * 10 , "start generating similarity matrix", "="*10)
    
    x_feature = torch.FloatTensor(num_train, dim_feature).to(device)
    x_label = torch.FloatTensor(num_train, num_class).to(device)
    
    # num of nearest neighbors 
    k = int(num_train * k_nn)
    o = int(num_train * o_nn * 1.5)
    
    model.eval()
    
    with torch.no_grad():
    # extract features from pre-trained CNN
        for batch in train_loader:
            img, label = batch['img'].to(device), batch['label'].to(device)
            
            features, _ = model(img)
            
            x_feature[batch['idx'], :] = features
            x_label[batch['idx'], :] = label.float()
        
    # neighbor graph 
    dim = x_feature.shape[0] # num of training data
    normal = torch.sqrt((x_feature.pow(2)).sum(1)).view(-1, 1)
    normal_feature = x_feature / (normal.expand(dim, dim_feature))
    sim1 = normal_feature.mm(normal_feature.t())
    final_sim = sim1 * 1.0
    
    sim = sim1 - torch.eye(dim).float().to(device)
    
    # top = torch.rand((1, dim)).float()
    # for i in range(dim):
    #     top[0, :] = sim[i, :]
    #     top20 = top.sort()[1][0]
    #     zero = torch.zeros(dim).float()
    #     zero[top20[-k:]] = 1.0
    #     sim[i, :] = top[0, :] * zero

    top = torch.rand((dim, dim)).float()
    top = deepcopy(sim)
    
    top20 = top.sort(1)[1]
    zero = torch.zeros((dim, dim)).float().to(device)
    for i in range(dim): zero[i, top20[i, -k:]] = 1.0
    sim = deepcopy(top * zero)
    
    A = (sim > 0.0001).float()
    A = (A * A.t()) * sim
    
    sum_row = A.sum(1)
    aa = dim - (sum_row > 0).sum()
    kk = sum_row.sort()[1]
    
    res_ind = list(range(dim))
    
    for ind in range(aa):
        res_ind.remove(kk[ind])
    
    res_ind = random.sample(res_ind, dim-aa)
    
    ind_to_new_id = {}
    
    for i in range(dim - aa): 
        ind_to_new_id[i] = res_ind[i]
    
    res_ind = (torch.from_numpy(np.asarray(res_ind))).long()
    
    sim = sim[res_ind][:, res_ind]
    
    # sim20 = {}
    
    dim = dim - aa # (sum_row > 0).sum()
    top = torch.rand((dim, dim)).float()
    
    top = deepcopy(sim)
    top20 = top.sort(1)[1]
    zero = torch.zeros((dim, dim)).float().to(device)
    for i in range(dim): zero[i, top20[i, -k:]] = 1.0
    knn_list = top20[:, -k:]
    sim20 = deepcopy(knn_list)
    
    sim = deepcopy(top * zero)

    G = (sim > 0.0001).float()
    
    G = (G * G.t()) * sim
    D = G.sum(1)
    
    ## Generate hat_G 
    help_D = torch.diag(D.pow(-0.5))
    
    G = G.mm(help_D)
    _G = help_D.mm(G)
    
    manifold_sim = (1 - alpha) * torch.inverse(torch.eye(dim).float().to(device) - alpha * _G)
    
    manifold20 = {}
    for i in range(dim):
        top[i, :] = manifold_sim[i, :]
        top20 = top[i].sort()[1]
        k = top20[-o:].cpu().tolist()
        manifold20[i] = k
        
    sim20_ = {}
    for ii in range(len(sim20)): sim20_[ii] = sim20[ii].cpu().tolist()
    
    for i in range(len(sim20)):
        aa = len(manifold20[i])
        zz = deepcopy(manifold20[i])
        ddd = []
        for k in range(aa):
            if zz[k] in sim20_[i]:
                sim20_[i].remove(zz[k])
                manifold20[i].remove(zz[k])
                ddd.append(ind_to_new_id[zz[k]])
        j = ind_to_new_id[i]
        for l in ddd:
            final_sim[j, l] = 1.0
        for l in sim20_[i]:
            final_sim[j, ind_to_new_id[l]] = 0.0
    
    f1 = (final_sim > 0.999).float()
    f1 = ((f1 + f1.t()) > 0.999).float()
    f2 = (final_sim < 0.0001).float()
    f2 = ((f2 + f2.t()) > 0.999).float()
    final_sim = final_sim * (1. - f2)
    final_sim = final_sim * (1. - f1) + f1
    final_sim = 2 * final_sim - 1.0
    
    print("=" * 10 , "end of generating similarity matrix", "="*10)
    
    return final_sim
    
class HashNet(nn.Module):
    def __init__(self, backbone_with_encoder:Module) -> None:
        super().__init__()

        self.backbone_with_encoder = backbone_with_encoder
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        
        out = self.backbone_with_encoder(x)
        y = self.tanh(out['continuous_code'])
        
        return out['backbone_last_output'], y
    
    
class LogCoshLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, result_from_model, code_length, batch_S, batch_size):
        
        hat_b = result_from_model
        sim_bin = torch.mm(hat_b, hat_b.t()) / float(code_length)
        cosh_result = torch.cosh((sim_bin - batch_S))
        loss = torch.log(cosh_result).sum() / (batch_size * batch_size)
        
        return loss
        
    
class MLS3RDUH(DeepHashBase):
    def _get_default_config_dict(self) -> dict:
        
        config = {
            "batch_size" : 128,
            "max_epoch" : 150,
            "learning_rate" : 0.04,
            "optimizer_name" : "sgd",
            "sgd_momentum" : 0.9,
            "sgd_weight_decay" : 10 ** (-5),
            "lr_scheduler" : "none",
            "k_nn" : 0.06,
            "o_nn" : 0.06, 
            "alpha" : 0.99,
        }
        return config
        
    def _get_config_dict_for_dataset(self, default_config: dict, dataset: str) -> dict:
        
        return default_config
    
    def _get_fixed_config_dict(self) -> dict:
        
        fixed_config = {
            "gen_code_method" : "sign",
            "dataset_return_index" : True,
            "dataset_return_paired_aug_img" : False,
            "finetune" : True,
            "transform" : "CenterCrop",
            "encoder_type" : "linear",
            "encoder_layers" : "layer=1",
            
        }
        return fixed_config
    
    def _add_model_specific_args_into_parser(self, parser: ArgumentParser) -> ArgumentParser:
        
        parser.add_argument("--k_nn", type=float, default=0.06, help="the rate of top k in k- nearest neighbors : k = (k_nn) * (dataset size)([default: %(defalut)f])")
        parser.add_argument("--o_nn", type=float, default=0.06, help="the rate of top o in manifold similarity : o = (o_nn) * (dataset size) ([default: %(defalut)f])")
        parser.add_argument("--alpha", type=float, default=0.99, help="hyper-parameter in random walk [default : %(default)f]")
        
        return parser
    
    def _train_model(self, backbone_with_encoder: Module, optimizer: Optimizer, scheduler: _LRScheduler, trainset: Dataset, testset: Dataset, dbset: Dataset, model_dir: str, result_dir: str, device: str, batch_size: int, eval_period: int, config: dict) -> None:
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5, last_epoch=-1)
        train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
        criterion = LogCoshLoss().to(device)
        
        model = HashNet(backbone_with_encoder).to(device)
        
        num_class = self._get_n_class()
        
        if config["backbone"] == "VGG16":
            dim_feature = 4096
        elif config["backbone"] == "ViT":
            dim_feature = 768
        
        similarity = generate_similarity_matrix(train_loader, len(trainset), model, dim_feature, num_class, config["k_nn"], config["o_nn"], config["alpha"], device) 
        
        for epoch in range(config["max_epoch"]):
            
            model.train()
            
            for batch in train_loader:
                
                data = batch["img"].to(device)
                # label = batch["label"].to(device)
                idx = batch["idx"].to(device)
                
                S = similarity[idx][:, idx].to(device)
                
                _, out = model(data)
                
                loss = criterion(out, config["bit"], S, idx.shape[0])
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                self._insert_iter_loss_sum({"loss": loss.item()}, current_batch_size= batch['img'].shape[0])
            
            scheduler.step()
            self._compute_loss_per_epoch()
            
            model_name = f"{epoch:03d}"
            self._save_train_model_log(model_name, "epoch", True)
            
            if (epoch + 1) % eval_period == 0:
                
                model.eval()
                self.start_eval_process(model_name, "epoch")
                self._save_train_model_params(model_name, "epoch")
                