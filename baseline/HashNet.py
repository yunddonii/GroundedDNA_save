

from argparse import ArgumentParser

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from torch.nn import Module
from torch.nn.functional import one_hot
import torch

from .base_model import DeepHashBase

# HashNet(ICCV2017)
# paper [HashNet: Deep Learning to Hash by Continuation](http://openaccess.thecvf.com/content_ICCV_2017/papers/Cao_HashNet_Deep_Learning_ICCV_2017_paper.pdf)
# code1 [HashNet caffe and pytorch](https://github.com/thuml/HashNet)
# code2 https://github.com/swuxyj/DeepHash-pytorch/blob/master/HashNet.py


class HashNetLoss(Module):
    def __init__(self):
        super(HashNetLoss, self).__init__()
        self.scale = 1

    def forward(self, u, y, alpha):
        u = torch.tanh(self.scale * u)
        S = (y @ y.t() > 0).float()
        sigmoid_alpha = alpha
        dot_product = sigmoid_alpha * u @ u.t()
        mask_positive = S > 0
        mask_negative = (1 - S).bool()
        
        neg_log_probe = dot_product + torch.log(1 + torch.exp(-dot_product)) -  S * dot_product
        S1 = torch.sum(mask_positive.float())
        S0 = torch.sum(mask_negative.float())
        S = S0 + S1

        neg_log_probe[mask_positive] = neg_log_probe[mask_positive] * S / S1
        neg_log_probe[mask_negative] = neg_log_probe[mask_negative] * S / S0

        loss = torch.sum(neg_log_probe) / S
        return loss
    

class HashNet(DeepHashBase):
    def _get_default_config_dict(self) -> dict:
        config = {
            "alpha"             : 0.1,
            "step_continuation" : 20,
            "max_epoch"         : 150,
            "optimizer_name"    : "adam",
            "lr_scheduler"      : "none"
            }
        return config
    
    def _get_config_dict_for_dataset(self, default_config: dict, dataset: str) -> dict:
        return default_config
    
    def _get_fixed_config_dict(self) -> dict:
        config = {
            "finetune"          : False,
            "gen_code_method"   : "sign"
        }
        return config
    
    def _add_model_specific_args_into_parser(self, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument("--alpha", default=0.1, help="(default: %(default)s)")
        parser.add_argument("--step_continuation", default=20, help="(default: %(default)s)")
        return parser
    
    def _train_model(self, backbone_with_encoder: Module, optimizer: Optimizer, scheduler: _LRScheduler, trainset: Dataset, testset: Dataset, dbset: Dataset, model_dir: str, result_dir: str, device: str, batch_size: int, eval_period: int, config: dict) -> None:
        
        train_loader = torch.utils.data.DataLoader( dataset=trainset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
        
        net = backbone_with_encoder.to(device)
        n_class = self._get_n_class()

        criterion = HashNetLoss()

        for epoch in range(config["max_epoch"]):
            criterion.scale = (epoch // config["step_continuation"] + 1) ** 0.5
            net.train()

            for batch in train_loader:

                image = batch['img'].to(device)
                label = batch['label'].to(device)

                optimizer.zero_grad()
                u = net(image)

                loss = criterion(u['continuous_code'], label.float(), config['alpha'])
                
                self._insert_iter_loss_sum({"loss": loss.item()}, current_batch_size= image.shape[0])

                loss.backward()
                optimizer.step()
            
            scheduler.step()
                
            self._compute_loss_per_epoch()
            
            model_name = f"{epoch:03d}"
            self._save_train_model_log(model_name, "epoch", True)
        

            if (epoch + 1) % eval_period == 0:
                self.start_eval_process(model_name, "epoch")
                self._save_train_model_params(model_name, "epoch")
