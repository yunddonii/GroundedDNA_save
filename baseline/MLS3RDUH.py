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


# The IJCAI paper and the authors' public implementation disagree in three
# places which materially affect a reproduction.  Keep the choices named here
# instead of hiding them behind numeric literals:
#
# * paper:  o = 0.06 N, Xavier hash projection, SGD momentum 0.9;
# * release (commit below): o = 0.06 N * 1.5, torch Linear defaults, and the
#   SGD constructor omits momentum (therefore uses 0.0).
#
# The registered ``mls3rduh`` runner deliberately follows the paper choices.
# The release policy remains callable through ``mls3rduh_neighbor_counts`` and
# ``initialize_mls3rduh_hash_head`` for an explicitly named diagnostic; it is
# never selected silently.
PAPER_SOURCE_PROFILE = "ijcai2020-paper-cache-v1"
RELEASE_SOURCE_PROFILE = "authors-release-5c9a99f-cache-v1"
PAPER_NEIGHBOR_POLICY = "paper_o_equals_o_nn_times_N"
RELEASE_NEIGHBOR_POLICY = "release_o_equals_1.5_times_o_nn_times_N"
PAPER_HASH_INIT_POLICY = "xavier_uniform_weight_zero_bias"
RELEASE_HASH_INIT_POLICY = "torch_nn_linear_default"
OFFICIAL_RELEASE_COMMIT = "5c9a99f23aea415e933c9b34e0297e64d96ff7c6"
PAPER_DOI = "10.24963/ijcai.2020/479"


def mls3rduh_neighbor_counts(num_train, k_nn, o_nn,
                             policy=PAPER_NEIGHBOR_POLICY):
    """Resolve the paper/release k and o definitions without ambiguity."""
    if policy == PAPER_NEIGHBOR_POLICY:
        o_multiplier = 1.0
    elif policy == RELEASE_NEIGHBOR_POLICY:
        o_multiplier = 1.5
    else:
        raise ValueError(f"unknown MLS3RDUH neighbor policy: {policy!r}")
    k = int(num_train * k_nn)
    o = int(num_train * o_nn * o_multiplier)
    return k, o


def _hash_projection(backbone_with_encoder):
    """Return the final linear hash projection of the cached-feature head."""
    encoder = getattr(backbone_with_encoder, "encoder_layers", None)
    if encoder is None:
        raise TypeError("MLS3RDUH model has no encoder_layers module")
    projections = [module for module in encoder.modules()
                   if isinstance(module, nn.Linear)]
    if not projections:
        raise TypeError("MLS3RDUH encoder has no linear hash projection")
    return projections[-1]


def initialize_mls3rduh_hash_head(
        backbone_with_encoder, policy=PAPER_HASH_INIT_POLICY):
    """Apply the selected source initialization to the final hash layer.

    ``RELEASE_HASH_INIT_POLICY`` is intentionally a no-op: a freshly-created
    ``nn.Linear`` already has the public release's PyTorch-default parameters.
    The paper policy explicitly applies Glorot/Xavier uniform weights and a
    zero bias before the optimizer is constructed.
    """
    projection = _hash_projection(backbone_with_encoder)
    if policy == PAPER_HASH_INIT_POLICY:
        nn.init.xavier_uniform_(projection.weight)
        if projection.bias is not None:
            nn.init.zeros_(projection.bias)
    elif policy != RELEASE_HASH_INIT_POLICY:
        raise ValueError(f"unknown MLS3RDUH hash initialization: {policy!r}")
    return projection


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


def generate_similarity_matrix(train_loader, num_train, model, dim_feature,
                               num_class, k_nn, o_nn, alpha, device,
                               neighbor_policy=PAPER_NEIGHBOR_POLICY):
    
    print("=" * 10 , "start generating similarity matrix", "="*10)
    
    x_feature = torch.FloatTensor(num_train, dim_feature).to(device)
    x_label = torch.FloatTensor(num_train, num_class).to(device)
    
    # Number of nearest neighbors.  The canonical path follows the paper's
    # k=0.06N, o=0.06N definition.  The public release's 1.5 multiplier is
    # available only through the explicitly named release policy above.
    k, o = mls3rduh_neighbor_counts(
        num_train, k_nn, o_nn, policy=neighbor_policy)
    
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
            # Immutable source-boundary metadata.  These scalar strings are
            # retained by both config.json and checkpoint config payloads.
            "implementation_variant" : PAPER_SOURCE_PROFILE,
            "release_implementation_variant" : RELEASE_SOURCE_PROFILE,
            "source_paper_doi" : PAPER_DOI,
            "source_code_commit" : OFFICIAL_RELEASE_COMMIT,
            "mls3rduh_neighbor_policy" : PAPER_NEIGHBOR_POLICY,
            "mls3rduh_effective_o_formula" : "int(N * o_nn)",
            "mls3rduh_release_effective_o_formula" : "int(N * o_nn * 1.5)",
            "mls3rduh_hash_head_initialization" : PAPER_HASH_INIT_POLICY,
            "mls3rduh_release_hash_head_initialization" : RELEASE_HASH_INIT_POLICY,
            "mls3rduh_optimizer_choice" : "paper_sgd_momentum_0.9",
            "mls3rduh_release_optimizer_choice" : "release_sgd_momentum_0.0",
            "mls3rduh_common_similarity" : "mutual_knn_random_walk_plus_2cosine_minus_1",
            "mls3rduh_common_objective" : "log_cosh_of_tanh_code_inner_product",
            "sgd_momentum" : 0.9,
            
        }
        return fixed_config

    def _build_model_from_config(self, d_in: int, config: dict) -> Module:
        """Build the cache head and initialize its hash layer per the paper."""
        model = super()._build_model_from_config(d_in, config)
        initialize_mls3rduh_hash_head(
            model,
            policy=config.get(
                "mls3rduh_hash_head_initialization",
                PAPER_HASH_INIT_POLICY,
            ),
        )
        return model
    
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
        elif config["backbone"] in ("SigLIP2", "siglip2", "SigLIP2-cached"):
            # Auto-detect from the cached visual_global D_proj (SigLIP2=768,
            # CLIP=512). Falls back to 768 if attribute missing.
            try:
                dim_feature = int(trainset.visual_global.shape[1])
            except Exception:
                dim_feature = 768
        else:
            raise ValueError(f"[MLS3RDUH] unknown backbone {config['backbone']!r} -- "
                             "add a dim_feature mapping above.")
        
        similarity = generate_similarity_matrix(
            train_loader, len(trainset), model, dim_feature, num_class,
            config["k_nn"], config["o_nn"], config["alpha"], device,
            neighbor_policy=config.get(
                "mls3rduh_neighbor_policy", PAPER_NEIGHBOR_POLICY),
        )
        
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
