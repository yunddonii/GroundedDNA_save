import numpy as np

import torch
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
import torch.nn.functional as F 
from argparse import ArgumentParser
from .base_model import DeepHashBase



class ADSH(DeepHashBase):
    def __init__(self) -> None:
        super().__init__()
        
    def _get_default_config_dict(self) -> dict:
        DEFAULT_CONFIG={
                "adsh_gamma"    : 200,
                "num_samples"   : 2000,
                "batch_size"    : 64,
                "max_iter"      : 150,       # T_out
                "max_epoch"     : 3,         # T_in
                "learning_rate" : 1e-4,
                "optimizer_name": "sgd",
                "sgd_weight_decay": 1e-5,
                "sgd_momentum"  : 0.99,
                "lr_scheduler"  : "explr",
                "explr_gamma"   : 0.9, 
            }
        
        return DEFAULT_CONFIG
        
    def _get_config_dict_for_dataset(self, default_config: dict, dataset: str) -> dict:
        # 각 데이터셋을 위한 하이퍼파라미터로 configuration을 변경
        
        # if dataset == "CIFAR10":
        #     default_config['max_iter'] = 5000000000
        
        return default_config
    
    
    def _get_fixed_config_dict(self) -> dict:
        FIXED_CONFIG={
                    "gen_code_method"               : "sign",
                    "dataset_return_index"          : True,
                    "dataset_return_paired_aug_img" : False,
                    "finetune"                      : False,
                }

        return FIXED_CONFIG

    def _add_model_specific_args_into_parser(self, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument("--adsh_gamma", default=200, type=int, help="(default: %(default)s)")
        parser.add_argument("--num_samples", default=2000, type=int, help="(default: %(default)s)")
        parser.add_argument("--max_iter", default=150, type=int, help="(default: %(default)s)")
        return parser



    def get_train_loader(self, num_samples, batch_size, trainset, num_workers=8):
        """ 
        train dataset에서 특정 개수(num_samples:2000)만큼 슬라이스해서 query set을 만드는 method \n
        (query set을 포함한) 나머지 train dataset은 database set이 된다
        """
        # indice = list(range(len(trainset)))
        indice = np.random.permutation(range(len(trainset)))
        # np.random.shuffle(indice)
        idx_omega = indice[:num_samples]
        
        q_sampler = torch.utils.data.SubsetRandomSampler(idx_omega)
        db_sampler = torch.utils.data.SubsetRandomSampler(indice)
        
        # test_data
        q_train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, sampler=q_sampler, num_workers=num_workers)
        db_train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, sampler=db_sampler, num_workers=num_workers)

        return idx_omega, q_train_loader, db_train_loader

    def get_test_loader(self, batch_size,  queryset, databaseset, num_workers=8):
        # query_data
        query_loader = torch.utils.data.DataLoader(dataset=queryset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        # database_data
        database_loader = torch.utils.data.DataLoader(dataset=databaseset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        return query_loader, database_loader

    def clac_sim_label(self, q_label, db_labels):
        """ 
        q_label.shape = (batch size, 1) -> (batch size, 10) \n
        db_labels.shape = (n, 1) -> (n, 10) \n
        m : batch_size \n
        n : # of sample \n
        """
        q_label = q_label.to(torch.float32)
        db_labels = db_labels.to(torch.float32)
        s = (torch.matmul(q_label, db_labels.t()) > 0).float() # s.shape = (m, n)
        s = torch.where(s == 1, s, torch.full_like(s, -1))
        
        ratio = s.sum()/(1-s).sum()
        s = s * (1+ratio) - ratio
        # s = s.view(q_label.shape[0], -1)
        
        return s



    def ADSH_loss(self, tanh_u, v, S, num_train, idx, bit, gamma):
        """
        tanh_u : hash code of models -> (batch_size, c) \n
        v : database points -> (n, c) \n
        S : similarity information -> (batch_size, n) \n
        idx_omega -> m \n
        bit -> c
        """
        
        
        likelihood = tanh_u @ v.t() - bit * S
        likelihood = (likelihood ** 2).sum()
        quantization = v[idx, :] - tanh_u
        quantization = gamma * (quantization ** 2).sum()
        
        # return (likelihood + quantization) / (num_train * tanh_u.shape[0])
        return  (likelihood + quantization) / ( num_train * tanh_u.shape[0]) #
    
    
    
    
    def _train_model(self, backbone_with_encoder: Module, optimizer: Optimizer, scheduler: _LRScheduler, trainset: Dataset, testset: Dataset, dbset: Dataset, model_dir: str, result_dir: str, device: str, batch_size: int, eval_period: int, config: dict) -> None:
        
        # Init Model
        n_class = self._get_n_class()
        model = backbone_with_encoder.to(device)
        

        __, database_loader = self.get_test_loader(batch_size, testset, dbset)
        

        V = torch.zeros((len(dbset), config["bit"])).to(device) # n x c
        
        max_outer_iter = config["max_iter"] 
        max_epoch = config["max_epoch"]
        
        for out_itr in range(max_outer_iter): # 150번 돎
            
            backbone_with_encoder.train()
            
            idx_omega, query_train_loader, db_train_loader = self.get_train_loader(config['num_samples'], batch_size, trainset)
            # idx_omega = torch.LongTensor(idx_omega)
            # idx_omega의 길이 : m (query sample 개수)
            # idx_omega 의 data type : list
            
            
            ## database label 가져오기 -> similarity label만들기 위함
            db_label_list = torch.zeros(len(dbset), n_class).to(torch.long) # n
            for batch in database_loader:
                db_idx, db_label = batch['idx'], batch['label']
                db_idx = torch.LongTensor(db_idx)
                db_label_list[db_idx] = db_label

            query_label = db_label_list[idx_omega, :] # m 
            
            S = self.clac_sim_label(query_label, db_label_list).float().to(device) # m x n
            db_label_list = db_label_list.to(device)
            U = torch.zeros((len(dbset), config["bit"])).to(device) # n x c
        
            for epoch in range(max_epoch): 
                num_data = 0
                for batch in query_train_loader:
                    image = batch['img'].to(device)
                    q_label = batch['label'].to(device)
                    
                    output = model(image)
                    hash_code = torch.tanh(output['continuous_code']) # batch size x c
                    
                    U[batch['idx']] = hash_code.data # U => n x c

                    sim_label = self.clac_sim_label(q_label, db_label_list).float() # batch size x n
                    
                    model.zero_grad()
                    
                    loss = self.ADSH_loss(hash_code, V, sim_label, len(trainset), batch['idx'], config['bit'], config['adsh_gamma'])
                    
                    self._insert_iter_loss_sum({'loss': loss.item() }, image.shape[0])
                    num_data += num_data
                    
                    loss.backward()
                    optimizer.step()
                    
                scheduler.step()
                    
                self._compute_loss_per_epoch()
                self._save_train_model_log(f"{out_itr:04d}_{epoch:03d}", "Iter_Epoch", enable_print=True)
                
                
            U_ = U[idx_omega] # U_ => m x c
            Q = - 2 * config["bit"] * (S.t() @ U_) - 2 * config["adsh_gamma"] * U # n x c
            
            # update V*_k
            for k in range(config["bit"]):
                ex_k = [ii for ii in range(config["bit"]) if ii != k]
                V_hat = V[:,ex_k]

                U_k = U_[:,k] # m x 1
                U_hat = U_[:, ex_k]
                Q_k = Q[:, k] # n x 1
                
                # k-th col of V(db) update
                V[:, k] = - (2 * ((V_hat @ U_hat.t()) @ U_k) + Q_k).sign()

            
            etc_info = {}
            etc_info['database_codes'] = V.cpu().numpy() 

            if  ((out_itr + 1) % eval_period == 0): #(out_itr == 0) or
                model_id = f"itr{out_itr:04d}_e{epoch:03d}"
                id_type = "Iter_Epoch"
                self.start_eval_process(model_id, id_type)
                self._save_train_model_params( model_id, id_type, etc_info = etc_info) 


