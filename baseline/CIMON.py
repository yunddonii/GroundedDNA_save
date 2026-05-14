from argparse import ArgumentParser, BooleanOptionalAction
import torch
from torch.nn import Module
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from .base_model import DeepHashBase


import torch.optim as optim
import numpy as np
import os
import torchvision.transforms as transforms
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from loguru import logger
import torch.nn.functional as F
# from model_loader import load_model
# from evaluate import mean_average_precision
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from torch.nn import Parameter

import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time

from tqdm import tqdm

from copy import deepcopy

def extract_features(model, dataset, batch_size, device):
    """
    Extract features.
    """
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model.eval()
    model.set_extract_features(True)
    features_1 = []
    features_2 = []
    with torch.no_grad():

        for batch in tqdm(train_loader, desc="extract features : "):
            # logger.debug('[Batch:{}/{}]'.format(i+1, N))
            batch["img_tr1"] = batch["img_tr1"].to(device)
            batch["img_tr2"] = batch["img_tr2"].to(device)
            
            data1 = model(batch["img_tr1"]).cpu()
            data2 = model(batch["img_tr2"]).cpu()
            
            features_1.append(data1.view(batch["img_tr1"].shape[0], -1))
            features_2.append(data2.view(batch["img_tr2"].shape[0], -1))
            
    features_1 = torch.cat(features_1, dim=0)
    features_2 = torch.cat(features_2, dim=0)

    model.set_extract_features(False)
    model.train()
    
    return features_1, features_2


def generate_similarity_weight_matrix(features, threshold, num_clusters):
    """
    Generate similarity and confidence matrix.

    Args
        features(torch.Tensor): Features.
        alpha, beta(float): Hyper-parameters.

    Returns
        S(torch.Tensor): Similarity matrix.
    """
    # Cosine similarity
    cos_dist = squareform(pdist(features.numpy(), 'cosine')) # [5000, 5000]
    features = features.numpy()

    # Construct similarity matrix
    S = (cos_dist <= threshold) * 1.0 + (cos_dist > threshold) * -1.0
    
    # weight according to similarity

    # find the up and down extreme
    # Find maximum count of cosine distance
    # 가장 많은 개수의 cosine distance 크기를 찾는 과정 
    # cosine distance는 0과 2사이의 값을 갖지만 효율적으로 0과 1사이만 check한 것으로 보임
    # 따라서 cosine은 0.01 간격으로 샘플링된다고 가정
    max_cnt, max_cos = 0, 0
    interval = 1. / 100
    cur = 0
    for i in range(200):
        cur_cnt = np.sum((cos_dist > cur) & (cos_dist < cur + interval))
        if max_cnt < cur_cnt:
            max_cnt = cur_cnt
            max_cos = cur
        cur += interval



    # Split features into two parts
    flat_cos_dist = cos_dist.reshape((-1, 1))
    left = flat_cos_dist[np.where(flat_cos_dist <= max_cos)[0]]
    right = flat_cos_dist[np.where(flat_cos_dist > max_cos)[0]]

    # Reconstruct gaussian distribution
    left = np.concatenate([left, 2 * max_cos - left])
    right = np.concatenate([2 * max_cos - right, right])

    # Model data using gaussian distribution
    left_mean, left_std = norm.fit(left)
    right_mean, right_std = norm.fit(right)

    # dist_up = right_mean + beta * right_std
    
    # dist_down = left_mean - alpha * left_std

    def weight_norm(x, threshold):
        weight_f = (x>threshold)* (norm.cdf((x-right_mean)/right_std)-norm.cdf((threshold-right_mean)/right_std))/(1-norm.cdf((threshold-right_mean)/right_std)) + \
         (x<=threshold) * (norm.cdf((threshold-left_mean)/left_std)-norm.cdf((x-left_mean)/left_std) )/ (norm.cdf((threshold-left_mean)/left_std))
        return weight_f
    
    weight_1 = np.clip(weight_norm(cos_dist, threshold), 0, 1)


    # weight according to clustering
    features_norm = (features.T/ np.linalg.norm(features,axis=1)).T
    sp_cluster = SpectralClustering(n_clusters=num_clusters, random_state=0, assign_labels="discretize").fit(features_norm)
    A = sp_cluster.labels_[np.newaxis, :] # label vector
    # kmeans = KMeans(n_clusters=Classes, random_state=0, init='k-means++').fit(features_norm)
    # A = kmeans.labels_[np.newaxis, :] #label vector
    weight_2 =  ((((A - A.T) == 0)-1/2)*2* S +1)
    W = weight_1 * weight_2
    return torch.FloatTensor(S), torch.FloatTensor(W)



class SEM_CON_Loss(nn.Module):
    def __init__(self):
        super(SEM_CON_Loss, self).__init__()

    def forward(self, H, W, S):
        loss = (W * S.abs() * (H - S).pow(2)).sum() / (H.shape[0] ** 2)
        return loss
    
    
class CimonModel(nn.Module):
    
    def __init__(self, backbone_with_encoder:Module) -> None:
        super().__init__()
        
        self.backbone_with_encoder = backbone_with_encoder
        self.tanh = torch.nn.Tanh()
        self.extract_features = False
        
    def set_extract_features(self, flag:bool) -> None:
        """Extract features

        Args:
            flag (bool): true, if one needs extract features.

        Returns:
            None
        """
        
        self.extract_features = flag
        
    def forward(self, x):
        
        if self.extract_features:
            out = self.backbone_with_encoder(x)
            return out['backbone_last_output']
        else:
            out = self.backbone_with_encoder(x)
            v = self.tanh(out['continuous_code'])
            return v

    
class CIMON(DeepHashBase):
    def _get_default_config_dict(self) -> dict:
        
        config = {
            "batch_size" : 24,
            "max_epoch" : 150,
            "learning_rate": 1e-3,
            "optimizer_name" : "sgd",
            "sgd_momentum" : 0.9,
            "sgd_weight_decay" : 1e-5,
            "lr_scheduler" : "none",
            # "alpha" : 2,
            # "beta" : 2,
            "threshold" : 0.1,
            "eta" : 0.3,
            "temperature" : 0.5,
            "num_clusters" : 70, # spectal clustering
        }
        return config
    
    def _get_config_dict_for_dataset(self, default_config: dict, dataset: str) -> dict:
        return default_config
    
    def _get_fixed_config_dict(self) -> dict:
        
        fixed_config = {
            "gen_code_method" : "sign",
            "dataset_return_index" : True,
            "dataset_return_paired_aug_img" : True,
            "finetune" : True,
            "transform" : "CIB",
            "encoder_type" : "linear",
            "encoder_layers" : "layer=1",
            
        }
        return fixed_config
    
    def _add_model_specific_args_into_parser(self, parser: ArgumentParser) -> ArgumentParser:
        
        # parser.add_argument('--alpha', default=2, type=float,
        #                 help='Hyper-parameter in SSDH.[%(default)f]')
        # parser.add_argument('--beta', default=2, type=float,
        #                     help='Hyper-parameter in SSDH.[%(default)f]')
        parser.add_argument('--threshold', default=0.1, type=float,
                            help='Hyper-parameter.[%(default)f]')
        parser.add_argument('--num_clusters', default=70, type=int,
                            help='Number of clusters in Spectral Clusting[%(default)d]')
        parser.add_argument('--eta', default=0.3, type=float,
                            help='Hyper-parameter.[%(default)f]')
        parser.add_argument('--temperature', default=0.5, type=float,
                            help='Hyper-parameter in SimCLR .[%(default)f]')
        
        return parser
    
    def _train_model(self, backbone_with_encoder: Module, optimizer: Optimizer, scheduler: _LRScheduler, trainset: Dataset, testset: Dataset, dbset: Dataset, model_dir: str, result_dir: str, device: str, batch_size: int, eval_period: int, config: dict) -> None:
        
        model = CimonModel(backbone_with_encoder).to(device)
        
        train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        
        criterion = SEM_CON_Loss()
        # criterion_aug = nn.MSELoss()
     
        f1, f2 = extract_features(model, trainset, config["batch_size"], device=device)
        S_1, W_1 = generate_similarity_weight_matrix(f1, config["threshold"], num_clusters=config["num_clusters"])
        S_1 = S_1.to(device); W_1 = W_1.to(device)
        S_2, W_2 = generate_similarity_weight_matrix(f2, config["threshold"], num_clusters=config["num_clusters"])
        S_2 = S_2.to(device); W_2 = W_2.to(device)
        
        # Train
        for epoch in range(config["max_epoch"]):
            
            model.train()

            for i, batch in enumerate(train_loader):
                
                data = batch["img_tr1"].to(device)
                data_aug = batch["img_tr2"].to(device)
                
                optimizer.zero_grad()
                
                v = model(data)
                v_aug = model(data_aug)
                
                norm_v = F.normalize(v)
                norm_v_aug = F.normalize(v_aug)
                
                out = torch.cat([norm_v, norm_v_aug], dim=0) # top batch size => (1) || bottom batch size => (2)
                
                Z = torch.exp(torch.mm(out, out.t().contiguous())/ config["temperature"]) # denominators 
                
                # positive pair remove
                mask = (torch.ones_like(Z) - torch.eye(2 * data.shape[0], device=device)).bool() 
                Z = Z.masked_select(mask).view(2 * data.shape[0], -1) # [batch_size, (batch_size-1)]
                
                pos_sim = torch.exp(torch.sum(norm_v * norm_v_aug, dim=-1) / config["temperature"]) # numerators (1) x (2)
                pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
                
                nce_loss = (- torch.log(pos_sim / Z.sum(dim=-1))).mean() # l_cc
                
                H = v @ v.t() / config["bit"] # [batch_size, batch_size]
                
                H_aug = v_aug @ v_aug.t() / config["bit"] # [batch_size, batch_size]
                
                targets_1 = S_1[batch["idx"], :][:, batch["idx"]]
                targets_2 = S_2[batch["idx"], :][:, batch["idx"]]
                weights_1 = W_1[batch["idx"], :][:, batch["idx"]]
                weights_2 = W_2[batch["idx"], :][:, batch["idx"]]
                
                loss =  (criterion(H, weights_2, targets_2)+  criterion(H_aug, weights_1, targets_1)) + (criterion(H_aug, weights_2, targets_2) \
                    + criterion(H, weights_1, targets_1)) + config["eta"] * nce_loss 
            
                loss.backward()
                optimizer.step()
                self._insert_iter_loss_sum({"loss": loss.item()}, current_batch_size= batch['img'].shape[0])
                
            self._compute_loss_per_epoch()
            
            model_name = f"{epoch:03d}"
            self._save_train_model_log(model_name, "epoch", True)
        

            if (epoch + 1) % eval_period == 0:
                model.eval()
                self.start_eval_process(model_name, "epoch")
                self._save_train_model_params(model_name, "epoch")
                
                
                
                

    