from argparse import ArgumentParser

from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset

from argparse import ArgumentParser

import torch
from torch import nn
import torch.nn.functional as F
from .base_model import DeepHashBase

# code https://github.com/kamwoh/orthohash/blob/main/main.py


def get_hd(a, b):
    return 0.5 * (a.size(0) - a @ b.t()) / a.size(0)

def get_imbalance_mask(sigmoid_logits, labels, n_class, threshold=0.7, imbalance_scale=-1):
    if imbalance_scale == -1:
        imbalance_scale = 1 / n_class

    mask = torch.ones_like(sigmoid_logits) * imbalance_scale

    # wan to activate the output
    mask[labels == 1] = 1

    # if predicted wrong, and not the same as labels, minimize it
    correct = (sigmoid_logits >= threshold) == (labels == 1)
    mask[~correct] = 1

    multiclass_acc = correct.float().mean()

    # the rest maintain "imbalance_scale"
    return mask, multiclass_acc


def get_codebook(n_class, n_bit, maxtries=10000, initdist=0.61, mindist=0.2, reducedist=0.01):
    """
    brute force to find centroid with furthest distance
    :param n_class:
    :param n_bit:
    :param maxtries:
    :param initdist:
    :param mindist:
    :param reducedist:
    :return:
    """
    codebook = torch.zeros(n_class, n_bit)
    i = 0
    count = 0
    currdist = initdist
    while i < n_class:
        print(i, end='\r')
        c = torch.randn(n_bit).sign()
        nobreak = True
        for j in range(i):
            if get_hd(c, codebook[j]) < currdist:
                i -= 1
                nobreak = False
                break
        if nobreak:
            codebook[i] = c
        else:
            count += 1

        if count >= maxtries:
            count = 0
            currdist -= reducedist
            print('reduce', currdist, i)
            if currdist < mindist:
                raise ValueError('cannot find')

        i += 1
    codebook = codebook[torch.randperm(n_class)]
    return codebook



class OrthoHashLoss(nn.Module):  # m_type: cos/arc
    def __init__(self, s=8, m=0.2, m_type='cos', multiclass=False, multiclass_loss='label_smoothing', onehot=True,**kwargs):
        super(OrthoHashLoss, self).__init__()
        self.losses = {}
        self.s = s
        self.m = m
        self.m_type = m_type
        self.multiclass = multiclass
        self.onehot = onehot

        self.multiclass_loss = multiclass_loss
        assert multiclass_loss in ['bce', 'imbalance', 'label_smoothing']

    def compute_margin_logits(self, logits, labels):
        if self.m_type == 'cos':
            if self.multiclass:
                y_onehot = labels * self.m
                margin_logits = self.s * (logits - y_onehot)
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
                margin_logits = self.s * (logits - y_onehot)
                
        elif self.m_type == 'arc':
            if self.multiclass:
                y_onehot = labels * self.m
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.s * logits
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.s * logits
        
        else:
            raise NotImplementedError

        return margin_logits

    def forward(self, logits, labels):
        if self.multiclass:
            if not self.onehot:
                labels = F.one_hot(labels, logits.size(1))
            labels = labels.float()

            useful_sample_idx = labels.sum(-1) != 0
            logits = logits[useful_sample_idx]
            labels = labels[useful_sample_idx]
            
            margin_logits = self.compute_margin_logits(logits, labels)

            if self.multiclass_loss in ['bce', 'imbalance']:
                loss_ce = F.binary_cross_entropy_with_logits(margin_logits[useful_sample_idx], labels[useful_sample_idx], reduction='none')
                if self.multiclass_loss == 'imbalance':
                    imbalance_mask, multiclass_acc = get_imbalance_mask(torch.sigmoid(margin_logits), labels,
                                                                        labels.size(1))
                    loss_ce = loss_ce * imbalance_mask
                    loss_ce = loss_ce.sum() / (imbalance_mask.sum() + 1e-7)
                    self.losses['multiclass_acc'] = multiclass_acc
                else:
                    loss_ce = loss_ce.mean()
                    
            elif self.multiclass_loss in ['label_smoothing']:
                log_logits = F.log_softmax(margin_logits, dim=1)
                labels_scaled = labels / labels.sum(dim=1, keepdim=True)
                loss_ce = - (labels_scaled * log_logits).sum(dim=1)
                loss_ce = loss_ce.mean()
            else:
                raise NotImplementedError(f'unknown method: {self.multiclass_loss}')
        else:
            if self.onehot:
                labels = labels.argmax(1)

            margin_logits = self.compute_margin_logits(logits, labels)
            loss_ce = F.cross_entropy(margin_logits, labels)
            
        self.losses['ce'] = loss_ce
        
        return loss_ce
    
    
class CosSim(nn.Module):
    def __init__(self, n_feat, n_class, codebook=None, learn_cent=True):
        super(CosSim, self).__init__()
        self.n_feat = n_feat
        self.n_class = n_class
        self.learn_cent = learn_cent

        if codebook is None:  # if no centroids, by default just usual weight
            codebook = torch.randn(n_class, n_feat)

        self.centroids = nn.Parameter(codebook.clone())
        if not learn_cent:
            self.centroids.requires_grad_(False)

    def forward(self, x):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True)
        n_feat = torch.div(x, norms)

        norms_c = torch.norm(self.centroids, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centroids, norms_c)
        
        logits = torch.matmul(n_feat, torch.transpose(ncenters, 0, 1))

        return logits

    def extra_repr(self) -> str:
        return 'in_features={}, n_class={}, learn_centroid={}'.format(
            self.n_feat, self.n_class, self.learn_cent
        )



class ModelWithCodebook(Module):
    def __init__(self, backbone_with_encoder:Module, n_class:int, n_bit:int, codebook:torch.Tensor=None) -> None:
        super(ModelWithCodebook, self).__init__()
        self.backbone_with_encoder = backbone_with_encoder
        self.classifier = CosSim(n_bit, n_class, codebook, learn_cent=False)
        self.extrabit = 0
        
    def forward(self, img):
        output = self.backbone_with_encoder(img)
        
        continuous_code = output['continuous_code']
        classified_output = self.classifier(continuous_code)
        
        return classified_output, continuous_code


class OrthoHash(DeepHashBase):
    
    def _get_default_config_dict(self) -> dict:
        DEFAULT_CONFIG={
                "batch_size"        : 64,
                "optimizer_name"    :  'adam',
                "lr_scheduler"       : 'explr',
                "learning_rate"     : 1e-4,
                "batch_norm"        : True,
                "margin"            : 0.2,
                "margin-type"       : 'cos', # cos, arc 
                "scale"             : 'root(bit)',
                "codebook-method"   : "B", 
                "multiclass-loss"   : "label_smoothing"
            }
        
        return DEFAULT_CONFIG
        
    
    def _get_config_dict_for_dataset(self, default_config: dict, dataset: str) -> dict:
        return default_config
    
    
    
    def _get_fixed_config_dict(self) -> dict:
    
        FIXED_CONFIG={
                    "gen_code_method"               : "sign",
                    "dataset_return_index"          : False,
                    "dataset_return_paired_aug_img" : False,
                    "finetune"                      : False,
                }

        return FIXED_CONFIG ## codebook gen
    
    def _add_model_specific_args_into_parser(self, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--margin', default=0.2, type=float, help='ortho margin default: %(default)s)')
        parser.add_argument('--margin-type', default='cos', choices=['cos', 'arc'], help='margin type default: %(default)s)')
        parser.add_argument('--scale', default='root(bit)', type=str, help='scale for cossim default: %(default)s)')
        parser.add_argument('--codebook-method', default='B', choices=['N', 'B', 'O'], help='N = sign of gaussian; '
                                                                                            'B = bernoulli; '
                                                                                            'O = optimize default: (%(default)s)')
        parser.add_argument('--multiclass-loss', default='label_smoothing',
                            choices=['bce', 'imbalance', 'label_smoothing'], help='multiclass loss types default: %(default)s)')
        
        return parser
    
    def _train_model(self, backbone_with_encoder: Module, optimizer: Optimizer, scheduler: _LRScheduler, trainset: Dataset, testset: Dataset, dbset: Dataset, model_dir: str, result_dir: str, device: str, batch_size: int, eval_period: int, config: dict) -> None:
        
        n_class = self._get_n_class()  
        n_bit = config['bit']
        is_one_hot_emb = True
        
        # argparse turns --codebook-method into config['codebook_method'].
        cb_method = config.get('codebook_method', config.get('codebook-method', 'B'))
        if cb_method == 'N':  # normal
            codebook = torch.randn(n_class, n_bit)
        elif cb_method == 'B':  # bernoulli
            prob = torch.ones(n_class, n_bit) * 0.5
            codebook = torch.bernoulli(prob) * 2. - 1.
        elif cb_method == 'O': # optim
            codebook = get_codebook(n_class, n_bit)
        else:
            raise NotImplementedError
        
        if config['scale'] == 'root(bit)':
            scale = float(config['bit']) ** (1/2)
        else:
            scale = float(config['scale'])
        
        
        codebook = codebook.sign().to(device)
        
        # onehot 판단하는거 넣기
        is_multi_label=  self.is_multi_label_class()
        
        
        model = ModelWithCodebook( backbone_with_encoder, n_class, n_bit, codebook).to(device)
        
        train_loader = torch.utils.data.DataLoader( dataset=trainset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
        
        model.train()

        m_type = config.get('margin_type', config.get('margin-type', 'cos'))
        mc_loss = config.get('multiclass_loss', config.get('multiclass-loss', 'label_smoothing'))
        criterion = OrthoHashLoss(scale, config['margin'], m_type, is_multi_label, mc_loss, is_one_hot_emb)
        
        for epoch in range(config['max_epoch']):
            for batch in train_loader:
                data, labels = batch['img'].to(device), batch['label'].to(device)
                logits, codes = model(data)

                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_dict = {}
                loss_dict['loss_total'] = loss.item()
                loss_dict['loss_ce'] = criterion.losses['ce'].item()
                
                self._insert_iter_loss_sum(loss_dict, current_batch_size= data.shape[0])
            
            scheduler.step()
            
            model_name = f"{epoch:03d}"
            self._compute_loss_per_epoch()
            self._save_train_model_log(model_name, "epoch", True)

            if (epoch + 1) % eval_period == 0:
                self.start_eval_process(model_name, "epoch")
                self._save_train_model_params(model_name, "epoch", {'codebook': codebook})

