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

def Soft_Quantization(X, C, N_books, tau_q):
    L_word = int(C.size()[1]/N_books)
    x = T.split(X, L_word, dim=1)
    c = T.split(C, L_word, dim=1)
    for i in range(N_books):
        soft_c = F.softmax(squared_distances(x[i], c[i]) * (-tau_q), dim=-1)
        if i==0:
            Z = soft_c @ c[i]
        else:
            Z = T.cat((Z, soft_c @ c[i]), dim=1)
    return Z

def squared_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    return T.sum(diff * diff, -1)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet_Baseline(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet_Baseline, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc_out = nn.Linear(512*block.expansion, 512)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc_out(out))
        return out
    
    


class CQCLoss(nn.Module):

    def __init__(self, device, batch_size, tau_cqc):
        super(CQCLoss, self).__init__()
        self.batch_size = batch_size
        self.tau_cqc = tau_cqc
        self.device = device
        self.COSSIM = nn.CosineSimilarity(dim=-1)
        self.CE = nn.CrossEntropyLoss(reduction="sum")
        self.get_corr_mask = self._get_correlated_mask().type(torch.bool)

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    def forward(self, Xa, Xb, Za, Zb):

        XaZb = torch.cat([Xa, Zb], dim=0)
        XbZa = torch.cat([Xb, Za], dim=0)

        Cossim_ab = self.COSSIM(XaZb.unsqueeze(1), XaZb.unsqueeze(0))
        Rab = torch.diag(Cossim_ab, self.batch_size)
        Lab = torch.diag(Cossim_ab, -self.batch_size)
        Pos_ab = torch.cat([Rab, Lab]).view(2 * self.batch_size, 1)
        Neg_ab = Cossim_ab[self.get_corr_mask].view(2 * self.batch_size, -1)

        Cossim_ba = self.COSSIM(XbZa.unsqueeze(1), XbZa.unsqueeze(0))
        Rba = torch.diag(Cossim_ba, self.batch_size)
        Lba = torch.diag(Cossim_ba, -self.batch_size)    
        Pos_ba = torch.cat([Rba, Lba]).view(2 * self.batch_size, 1)
        Neg_ba = Cossim_ba[self.get_corr_mask].view(2 * self.batch_size, -1)


        logits_ab = torch.cat((Pos_ab, Neg_ab), dim=1)
        logits_ab /= self.tau_cqc

        logits_ba = torch.cat((Pos_ba, Neg_ba), dim=1)
        logits_ba /= self.tau_cqc

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        
        loss = self.CE(logits_ab, labels) + self.CE(logits_ba, labels)
        return loss / (2 * self.batch_size)
    
    
class SPQ(DeepHashBase):
    def _get_default_config_dict(self) -> dict:
        
        config = {
            "batch_size" : 256,
            "max_epoch" : 5000,
            "learning_rate" : 3e-4,
            "weight_decay" : 10e-6,
            "M_books" : 8,
            "K_words" : 16,
            "D_M_word" : 16,
            "tau_q" : 5.0,
            "tqu_cqc" : 0.5,
             
        }
        return config
    
    def _get_config_dict_for_dataset(self, default_config: dict, dataset: str) -> dict:
        return default_config
    
    def _get_fixed_config_dict(self) -> dict:
        
        fixed_config = {
            "gen_code_method" : 
        }
        return fixed_config