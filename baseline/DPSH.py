from argparse import ArgumentParser, BooleanOptionalAction
import torch
from torch.nn import Module
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from .base_model import DeepHashBase





def DPSH_loss(output, target, eta, enable_mask=False):
    """ 
    target : label, class 정보 받아오기
    """
    u = output #[batch_size 48]
    target = target.float()

    S = (target @ target.t() > 0).float() # [batch_size batch_size]
    Theta = 0.5 * (u @ u.T)     # [batch_size feat] x [feat batch_size] = [batch_size batch_size]

    if enable_mask:
        n_sample = S.shape[0]
        mask = torch.eye(n_sample, device=output.device).bool()
        mask = ~mask
        S = torch.masked_select(S, mask)
        Theta = torch.masked_select(Theta, mask)

    likelihood_loss =  binary_cross_entropy_with_logits(Theta, S)
    quantization_loss = eta * (torch.sign(u) - u).pow(2).mean()
    
    return likelihood_loss + quantization_loss

class DPSH(DeepHashBase):
    def _get_default_config_dict(self) -> dict:
        config = {
                    "eta"           : 10,
                    "batch_size"    : 16,
                    "max_epoch"     : 200,         # T_in
                    "learning_rate" : 1e-4,
                    "optimizer_name": "adam",
                    "lr_scheduler"  : "none",
                    "enable_mask"   : False
                    # "momentum"      : 0.99,
                    # "weight_decay"  : 1e-5,
                    # "sche_gamma"    : 0.9,
                }
        return config
    
    def _get_config_dict_for_dataset(self, default_config: dict, dataset: str) -> dict:
        return default_config
    
    def _get_fixed_config_dict(self) -> dict:
        config = {
                "finetune"          : False,
                "gen_code_method"   : "sign",
                }
        return config

    def _add_model_specific_args_into_parser(self, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument("--eta", type=float, default=10, help="(default: %(default)s)")
        parser.add_argument("--enable_mask", action=BooleanOptionalAction, default=False )
        return parser
    
    def _train_model(self, backbone_with_encoder: Module, optimizer: Optimizer, scheduler: _LRScheduler, trainset: Dataset, testset: Dataset, dbset: Dataset, model_dir: str, result_dir: str, device: str, batch_size: int, eval_period: int, config: dict) -> None:
        
        # Data Loader
        train_loader = torch.utils.data.DataLoader( dataset=trainset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
        
        model = backbone_with_encoder.to(device)
        
        n_class = self._get_n_class()

        model.train()
        
        for epoch in range(config["max_epoch"]):
            for batch in train_loader:
                image = batch['img'].to(device)
                label = batch['label'].to(device)
                
                if label.ndim == 1:
                    label = one_hot(label.long(), num_classes=n_class).float()
                
                output = model(image)
                
                loss = DPSH_loss(output['continuous_code'], label, config['eta'], config['enable_mask'])
                
                self._insert_iter_loss_sum({"loss": loss.item()}, current_batch_size= image.shape[0])
                
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

