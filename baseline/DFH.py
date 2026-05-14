
from argparse import ArgumentParser

from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset


import torch
from torch.nn.functional import one_hot
from .base_model import DeepHashBase



# DFH(BMVC2019)
# paper [Push for Quantization: Deep Fisher Hashing](https://arxiv.org/abs/1909.00206)
# code 1 [Push-for-Quantization-Deep-Fisher-Hashing](https://github.com/liyunqianggyn/Push-for-Quantization-Deep-Fisher-Hashing)
# code 2 https://github.com/swuxyj/DeepHash-pytorch/blob/master/DFH.py



class DFHLoss(torch.nn.Module):
    def __init__(self, n_train, n_class, mu, m, eta,vul, nta,  device, bit):
        super(DFHLoss, self).__init__()
        self.U = torch.zeros(bit, n_train).float().to(device)
        self.Y = torch.zeros(n_class, n_train).float().to(device)

        # Relax_center
        self.V = torch.zeros(bit, n_class).to(device)

        # Center
        self.C = self.V.sign().to(device)

        T = 2 * torch.eye(self.Y.size(0)) - torch.ones(self.Y.size(0))
        TK = self.V.size(0) * T
        self.TK = torch.FloatTensor(torch.autograd.Variable(TK, requires_grad=False)).to(device)
        self.mu = mu
        self.m = m
        self.eta = eta
        self.vul = vul
        self.nta = nta
        
        # self.theta = theta


    def forward(self, u, y, ind):

        self.U[:, ind] = u.t().data
        self.Y[:, ind] = y.t()

        b = (self.mu * self.C @ y.t() + u.t()).sign()

        self.Center_gradient(torch.autograd.Variable(self.V, requires_grad=True),
                             torch.autograd.Variable(y, requires_grad=False),
                             torch.autograd.Variable(b, requires_grad=False))

        s = (y @ self.Y > 0).float()
        inner_product = u @ self.U * 0.5
        inner_product = inner_product.clamp(min=-100, max=50)
        metric_loss = ((1 - s) * torch.log(1 + torch.exp(self.m + inner_product))
                       + s * torch.log(1 + torch.exp(self.m - inner_product))).mean()
        
        # metric_loss = (torch.log(1 + torch.exp(self.theta)) - S * self.theta).mean()  # Without Margin
        
        quantization_loss = (b - u.t()).pow(2).mean()
        loss = metric_loss + self.eta * quantization_loss
        return loss

    def Center_gradient(self, V, batchy, batchb):
        alpha = 0.03
        for i in range(200):
            intra_loss = (V @ batchy.t() - batchb).pow(2).mean()
            inter_loss = (V.t() @ V - self.TK).pow(2).mean()
            quantization_loss = (V - V.sign()).pow(2).mean()

            loss = intra_loss + self.vul * inter_loss + self.nta * quantization_loss

            loss.backward()

            if i in (149, 179):
                alpha = alpha * 0.1

            V.data = V.data - alpha * V.grad.data

            V.grad.data.zero_()
        self.V = V
        self.C = self.V.sign()



class DFH(DeepHashBase):
    def _get_default_config_dict(self) -> dict:
        config = {
        "dfh_m": 3,
        "dfh_mu": 0.1,
        "dfh_vul": 1,
        "dfh_nta": 1,
        "dfh_eta": 0.5,
        "max_epoch": 150,
        "optimizer_name": 'adam',
        "lr_scheduler": "none"
    }
        return config
    
    def _get_fixed_config_dict(self) -> dict:
        config = {
                    "gen_code_method"               : "sign",
                    "dataset_return_index"          : True,
                    "dataset_return_paired_aug_img" : False,
                    "finetune"                      : False,
        }
        
        return config
    
    def _get_config_dict_for_dataset(self, default_config: dict, dataset: str) -> dict:
        return default_config
    
    def _add_model_specific_args_into_parser(self, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument("--dfh_m", default= 3, type=float, help="(default: %(default)s)")
        parser.add_argument("--dfh_mu", default= 0.1, type=float, help="(default: %(default)s)")
        parser.add_argument("--dfh_vul", default= 1, type=float, help="(default: %(default)s)")
        parser.add_argument("--dfh_nta", default= 1, type=float, help="(default: %(default)s)")
        parser.add_argument("--dfh_eta", default= 0.5, type=float, help="(default: %(default)s)")
        
        return parser
    
    
    def _train_model(self, backbone_with_encoder: Module, optimizer: Optimizer, scheduler: _LRScheduler, trainset: Dataset, testset: Dataset, dbset: Dataset, model_dir: str, result_dir: str, device: str, batch_size: int, eval_period: int, config: dict) -> None:

        train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,  num_workers=8, shuffle=True)
        n_train = len(trainset)
        n_class = self._get_n_class()
        net = backbone_with_encoder.to(device)
        
        criterion = DFHLoss( n_train, n_class,
                                mu = config['dfh_mu'], 
                                m = config['dfh_m'], 
                                eta = config['dfh_eta'],
                                vul = config['dfh_vul'], 
                                nta = config['dfh_nta'], 
                                device = device,  
                                bit = config['bit'])

        for epoch in range(config['max_epoch']):
            net.train()

            for batch in train_loader:
                image = batch['img'].to(device)
                label = batch['label'].to(device)
 
                label = label.float()
                label /= label.sum(-1).reshape(-1, 1) #sum to 1 (Label Smoothing)
                
                ind = batch['idx'].to(device)

                optimizer.zero_grad()
                u = net(image)

                loss = criterion(u['continuous_code'], label, ind)

                loss.backward()
                optimizer.step()

                self._insert_iter_loss_sum({'loss': loss.item()},image.shape[0])
            
            scheduler.step()
            
            self._compute_loss_per_epoch()
            self._save_train_model_log(f"{epoch:03d}", "Epoch", enable_print=True)

            if (epoch + 1) % eval_period == 0:
                self.start_eval_process(f"{epoch:03d}", "Epoch")
                self._save_train_model_params(f"{epoch:03d}", "Epoch")


