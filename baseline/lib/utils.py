import numpy as np
from copy import deepcopy
import pandas as pd
import os
import random, os
import numpy as np
import torch

class Logger():
    def __init__(self, result_dir:str, config={}) -> None:
        self.train_loss_result_path = os.path.join(result_dir, f"train_loss.csv")
        self.test_loss_result_path = os.path.join(result_dir, f"test_loss.csv")
        self.database_loss_result_path = os.path.join(result_dir, f"database_loss.csv")

        self.rank_result_path = os.path.join(result_dir, f"rank_eval.csv")
        self.seq_r_result_path = os.path.join(result_dir, f"seq_r_eval.csv")
        self.gc_result_path = os.path.join(result_dir, f"gc_eval.csv")
        
        self.result_dir = result_dir
        
        if type(config) != dict: 
            self.config = vars(config)
        else:
            self.config = config
        
        self.loss_info = {}
        self.is_avg_loss = False
        self.n_samples = 0
    
    
    def save_eval_model_log(self, model_id:int|str, eval_score:dict, id_type:str="epoch",seq_r_info:dict=None, gc_info:dict=None):
        self.config['id_type'] = id_type
        self.config['model_id'] = model_id
        
        save_dict_into_csv(eval_score, id=model_id, id_type=id_type, path=self.rank_result_path, config= self.config)
        if seq_r_info != None:
            save_dict_into_csv(seq_r_info, id=model_id, id_type=id_type, path=self.seq_r_result_path,config= self.config)
        if gc_info != None:
            save_dict_into_csv(gc_info, id=model_id, id_type=id_type, path=self.gc_result_path,config= self.config)
        
    def insert_iter_loss_sum(self, loss_sum_info:dict, current_batch_size:int):
        # {[loss_name]: [loss_sum]}
        loss_metric_names = self.loss_info.keys()
        for key, value in loss_sum_info.items():
            if key in loss_metric_names:
                self.loss_info[key] += value * current_batch_size
            else:
                self.loss_info[key] = value * current_batch_size
        
        self.n_samples += current_batch_size
    
    def compute_loss_per_epoch(self):
        assert self.is_avg_loss == False
        for key, loss in self.loss_info.items():
            self.loss_info[key] = loss / self.n_samples
        self.is_avg_loss = True
        self.n_samples = 0
            
    def save_train_model_log(self, model_id, id_type:str="epoch", enable_print:bool = False, print_message:str=""):
        assert self.is_avg_loss == True
        if enable_print and (print_message == ""):
            print_message = f"Train {id_type}: {model_id}\t"
            
        for key, loss in self.loss_info.items():
            print_message += f"{key}:\t {loss}\t"
            
        if enable_print:
            print(print_message)

        save_dict_into_csv(self.loss_info, id=model_id, id_type= id_type, path=self.train_loss_result_path, config=self.config)
        self.loss_info = {}
        self.is_avg_loss = False


def fix_random_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def save_dict_into_csv(dict_info, path:str, id=int|str, id_type="epoch", config = {},):
    dict_info = {**dict_info, **config}
    dict_info[id_type] = id
    df_info = pd.DataFrame( pd.Series(dict_info)).T
    column_names = df_info.columns
    column_names = [id_type] + list(column_names[:-1])
    df_info = df_info[column_names]

    if os.path.exists(path):
        saved_df = pd.read_csv(path, dtype=str, index_col=None)
        new_df = pd.concat([saved_df, df_info], sort=False)
    else:
        new_df = df_info
    
    directory=os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    new_df.to_csv(path, index=False)
    

def adjust_learning_rate(optimizer, epoch, init_learning_rate, epoch_lr_decrease):
    lr = init_learning_rate * (0.1 ** (epoch // epoch_lr_decrease))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def sampling_from_dataset(train_dataset, test_dataset, database_dataset, n_class, n_query_per_class, n_database_per_class, n_train_per_class): # convert_train_test=False
    # train, test data를 모두 합침
    
    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate(( np.array(train_dataset.targets), np.array(test_dataset.targets)))

    first = True

    for label in range(n_class):
        # 레이블이 "label"에 해당하는 데이터의 인덱스를 찾음
        index = np.where(L == label)[0]

        # 인덱스를 랜덤으로 섞음
        N = index.shape[0]
        np.random.seed(0)
        perm = np.random.permutation(N)
        index = index[perm]

        # 레이블이 "label"에 해당하는 데이터 개수만큼 선택 -> testset, database set, train set에 추가함
        # Test
        data = X[index[0:n_query_per_class]]
        labels = L[index[0:n_query_per_class]]
        if first:
            test_L = labels
            test_data = data
        else:
            test_L = np.concatenate((test_L, labels))
            test_data = np.concatenate((test_data, data))

        # 나머지 데이터에서 레이블이 "label"에 해당하는 데이터를 선택 -> 
        # Database
        data = X[index[n_query_per_class:n_database_per_class]]
        labels = L[index[n_query_per_class:n_database_per_class]]

        if first:
            dataset_L = labels
            data_set = data
        else:
            dataset_L = np.concatenate((dataset_L, labels))
            data_set = np.concatenate((data_set, data))

        # Train
        data = X[index[n_query_per_class:n_train_per_class]]
        labels = L[index[n_query_per_class:n_train_per_class]]
        if first:
            train_L = labels
            train_data = data
        else:
            train_L = np.concatenate((train_L, labels))
            train_data = np.concatenate((train_data, data))

        first = False

    train_dataset.data = deepcopy(train_data)
    train_dataset.targets = deepcopy(train_L.astype(np.int64))
    test_dataset.data = deepcopy(test_data)
    test_dataset.targets = deepcopy((test_L).astype(np.int64))
    database_dataset.data = deepcopy(data_set)
    database_dataset.targets = deepcopy((dataset_L).astype(np.int64))
