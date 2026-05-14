from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.autograd import Function

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from torch.nn import Module

from .base_model import DeepHashBase


class hash(Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def hash_layer(input):
    return hash.apply(input)



class NtXentLoss(nn.Module):
    def __init__(self,  temperature):
        super(NtXentLoss, self).__init__()
        self.temperature = temperature
        self.similarityF = nn.CosineSimilarity(dim = 2)
        self.criterion = nn.CrossEntropyLoss(reduction = 'sum')
    

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size 
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
    

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarityF(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, batch_size )
        sim_j_i = torch.diag(sim, -batch_size )
        
        mask = self.mask_correlated_samples(batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).view(N, 1)
        negative_samples = sim[mask].view(N, -1)

        labels = torch.zeros(N, device=z_i.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
    
def compute_kl(prob, prob_v):
    prob_v = prob_v.detach()

    kl = prob * (torch.log(prob + 1e-8) - torch.log(prob_v + 1e-8)) + (1 - prob) * (torch.log(1 - prob + 1e-8 ) - torch.log(1 - prob_v + 1e-8))
    kl = torch.mean(torch.sum(kl, axis = 1))
    
    return kl


class CIBHash(DeepHashBase):
    def _get_default_config_dict(self) -> dict:
        config = {
                    "batch_size"    : 64,
                    "max_epoch"     : 100,         
                    "learning_rate" : 1e-3,
                    "optimizer_name": "adam",
                    "lr_scheduler"  : "explr",
                    "temperature"   : 0.3,
                    "weight"        : 0.001,
                    }
        return config
    
    
    def _get_config_dict_for_dataset(self, default_config: dict, dataset: str) -> dict:
        return default_config
    
    
    def _get_fixed_config_dict(self) -> dict:
        fixed_config ={
                    "gen_code_method"   : "sign_sigmoid_m0.5",
                    "finetune": False,
                    "dataset_return_index" : False,
                    "dataset_return_paired_aug_img" : True,
                    "transform"                     : "CIB"
                }
        return fixed_config
    
    
    def _add_model_specific_args_into_parser(self, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument("--temperature", default = 0.3, type = float,
                            help = "Temperature [%(default)d]",)
        parser.add_argument("--weight", default = 0.001, type=float,
                            help='weight of I(x,z) [%(default)f]')
        return parser
    
    
    def _train_model(self, backbone_with_encoder: Module, optimizer: Optimizer, scheduler: _LRScheduler, trainset: Dataset, testset: Dataset, dbset: Dataset, model_dir: str, result_dir: str, device: str, batch_size: int, eval_period: int, config: dict) -> None:
        
        # Data Loader
        train_loader = torch.utils.data.DataLoader( dataset=trainset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
        
        model = backbone_with_encoder.to(device)

        model.train()
        
        criterion = NtXentLoss(config['temperature'] )
        weight = config['weight']
        
        
        for epoch in range(config["max_epoch"]):
            
            for batch in train_loader:
            
                output_1 = model(batch['img_tr1'].to(device))
                output_2 = model(batch['img_tr2'].to(device)) #####################################################################
                
                prob_1 = torch.sigmoid(output_1['continuous_code'])
                z_i = hash_layer(prob_1 - 0.5)

                prob_2 = torch.sigmoid(output_2['continuous_code'])
                z_j = hash_layer(prob_2 - 0.5)

                kl_loss = (compute_kl(prob_1, prob_2) + compute_kl(prob_2, prob_1)) / 2
                
                contra_loss = criterion(z_i, z_j)
                loss = contra_loss + weight * kl_loss

                loss_dict = {'loss': loss.item(), 'contra_loss': contra_loss.item(), 'kl_loss': kl_loss.item()}
                
                self._insert_iter_loss_sum(loss_dict, current_batch_size= batch['img'].shape[0])
                
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
