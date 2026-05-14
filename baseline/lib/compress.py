from tqdm import tqdm
import numpy as np
import torch
import math
import itertools
from copy import deepcopy
from .quantizations import convert_b2q, convert_u2b_with_gcconstrant, convert_b2bq_with_seq_r_constrants

def convert_c2b(continuous_code:np.ndarray, method:str):
    if method == 'sign':
        return np.sign(continuous_code)
    elif method == 'sign_m0.5':
        return np.sign( continuous_code- 0.5)
    elif method == 'sign_sigmoid_m0.5':
        return np.sign( 1/(1+np.exp(np.clip(-continuous_code, -500, +500))) - 0.5)
    else:
        raise NotImplementedError(f"Unknown option:{method}")

def convert_all_type_codes(query_binary_codes,  retrieval_binary_codes,query_continuous_codes, retrieval_continuous_codes, max_seq_r):
    query_codes = {}
    retrieval_codes = {}
    
    query_codes['B'] = query_binary_codes
    retrieval_codes['B'] = retrieval_binary_codes
    
    query_codes['C'] = query_continuous_codes
    retrieval_codes['C'] = retrieval_continuous_codes
    
    query_codes['Q'] = convert_b2q(query_codes['B'])
    retrieval_codes['Q'] = convert_b2q(retrieval_codes['B'])

    query_codes['B_gc'] = convert_u2b_with_gcconstrant(query_codes['C'])
    retrieval_codes['B_gc'] = convert_u2b_with_gcconstrant(retrieval_codes['C'])

    query_codes['Q_gc'] = convert_b2q(query_codes['B_gc'])
    retrieval_codes['Q_gc'] = convert_b2q(retrieval_codes['B_gc'])

    query_codes['B_gc_seq_r'], query_codes['Q_gc_seq_r'] = convert_b2bq_with_seq_r_constrants(query_codes['B_gc'], max_seq_r=max_seq_r)
    query_codes['B_gc_seq_r'], query_codes['Q_gc_seq_r'] = query_codes['B_gc_seq_r'], query_codes['Q_gc_seq_r']
    
    retrieval_codes['B_gc_seq_r'], retrieval_codes['Q_gc_seq_r'] = convert_b2bq_with_seq_r_constrants(retrieval_codes['B_gc'], max_seq_r=max_seq_r)
    retrieval_codes['B_gc_seq_r'], retrieval_codes['Q_gc_seq_r'] = retrieval_codes['B_gc_seq_r'], retrieval_codes['Q_gc_seq_r']
    
    for code_type in query_codes.keys():
        if 'B'in code_type:
            query_codes[code_type] = (query_codes[code_type] == 1).astype(np.uint8)
            retrieval_codes[code_type] = (retrieval_codes[code_type] == 1).astype(np.uint8)
            
        elif 'Q' in code_type:
            query_codes[code_type] = query_codes[code_type].astype(np.uint8)
            retrieval_codes[code_type] = retrieval_codes[code_type].astype(np.uint8)
        
        elif 'C' in code_type:
            continue
        
        else:
            raise NotImplementedError(f"Unrecognized code type: {code_type}")
    
    return query_codes, retrieval_codes

def convert_bi2dec(codes, base):
    if codes.ndim == 1:
        codes = codes.reshape(1, -1)
    elif codes.ndim == 2:
        pass
    else:
        raise ValueError(f"Not Supported Shape:{codes.shape}")  
    
    
    if (codes.min() == -1) and (base == 2):
        codes = (codes == +1).astype(np.uint8)

    codes = codes.astype(np.uint8)
        
    
    n_samples, n_bit = codes.shape
    
    max_size = base ** n_bit
    
    if max_size <= 2**8:
        dtype = np.uint8
    elif max_size <= 2**16:
        dtype = np.uint16
    elif max_size <= 2**32:
        dtype = np.uint32
    elif max_size <=2**64:
        dtype = np.uint64
    elif max_size <=2**128:
        dtype = np.uint128
    else:
        raise ValueError(f"Not Supported: {n_bit}-bit")
             
    decimals = np.zeros(shape=(n_samples), dtype=dtype) 
    for bit in range(n_bit):
        decimals += codes[:, n_bit - bit - 1] * (base ** bit)
        
    return decimals

def convert_dec2bi(decimals, base, n_bit):
    assert decimals.ndim == 1
    n_samples, = decimals.shape

    codes = np.zeros(shape=(n_samples, n_bit), dtype=np.uint8) 
    quotient = decimals
    for bit in range(n_bit):
        quotient, remainder = np.divmod(quotient , base)
        codes[:, n_bit - bit - 1] = remainder
        
    return codes

def make_hash_table(codes, base):
    decimal = convert_bi2dec(codes, base)
    hash_table = {}
    for decimal_hash_code in np.unique(decimal):
        hash_table[decimal_hash_code] = np.where(decimal == decimal_hash_code)[0]
    return hash_table



def retrieve_items_with_distance(hash_table, query_code, base, distance = 0):
    n_bit = query_code.shape[-1]
    
    n_hashes = math.comb(n_bit, distance) # n_bit일때 거리가 distance인 샘플들의 개수: C_{distance}^{n_bit} 

    new_query_codes = []
    
    n_change_bit = distance
    
    if n_hashes == 1:
        new_query_codes.append(query_code)
    else:
        # 거리가 distance인 샘플들의 코드 # 쿼터너리도 지원되게 만들어야함
        for i, change_idx in enumerate(itertools.combinations(range(n_bit), n_change_bit)):
            change_idx = np.array(change_idx)
            original_values = query_code[change_idx]
            
            list_for_iter = []
            for value in original_values:
                values_for_change = list(range(base))
                values_for_change.remove(value)
                list_for_iter.append(values_for_change)
                
            for change_values in itertools.product(*list_for_iter):
                temp_code = deepcopy(query_code)
                np.put(temp_code, ind=change_idx, v=change_values)
                new_query_codes.append(temp_code )
    

    new_query_codes = np.stack(new_query_codes)

    query_decimals = convert_bi2dec(new_query_codes, base)
    
    retrieved_sample_idx = []
    retrieved_sample_key = []
    
    for query_decimal in query_decimals:
        try:
            retrieved = hash_table[query_decimal]
            retrieved_sample_idx.append(retrieved)
            retrieved_sample_key.append(query_decimal.repeat(len(retrieved)))
        except KeyError as e:
            pass
    
    if len(retrieved_sample_idx) != 0:
        retrieved_sample_idx = np.concatenate(retrieved_sample_idx, axis=0)
        retrieved_sample_key = np.concatenate(retrieved_sample_key, axis=0)
        
    else:
        retrieved_sample_idx = np.zeros(shape=(0),dtype=np.int8)
        retrieved_sample_key = np.zeros(shape=(0),dtype=np.int8)
        
    return retrieved_sample_idx, retrieved_sample_key


def retrieve_items_less_than_distance(hash_table, query_code, base, distance = 0):
    retrieved_sample_idx = []
    retrieved_sample_key = []
    for d in range(distance+1):
        idx, decimal = retrieve_items_with_distance(hash_table, query_code, base, distance=d)
        retrieved_sample_key.append(decimal)
        retrieved_sample_idx.append(idx)
    
    if len(retrieved_sample_idx) != 0:
        retrieved_sample_idx = np.concatenate(retrieved_sample_idx, axis=0)
        retrieved_sample_key = np.concatenate(retrieved_sample_key, axis=0)
        
    else:
        retrieved_sample_idx = np.zeros(shape=(0),dtype=np.int8)
        retrieved_sample_key = np.zeros(shape=(0),dtype=np.int8)
        
    return retrieved_sample_idx, retrieved_sample_key 





def compress_images(query_loader, db_loader, model, gen_code_method, device, classes=10, return_continuous_code=False, return_cnn_feat=False):

    dataloaders = {'query': query_loader, 'retrieval': db_loader}
    
    code_info_dict = {}
    label_info_dict = {}
    
    for key in ['query', 'retrieval']:
        code_info_dict[key] = {'B': None}
        label_info_dict[key] = None
        
        if return_continuous_code:
            code_info_dict[key]['C'] = None
            
        if return_cnn_feat:
            raise NotImplementedError()
            # code_info_dict[key]['F'] = None
        
    
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        for key, dataloader in dataloaders.items():
            code_info = code_info_dict[key]
            for i, batch in tqdm( enumerate(dataloader), desc=f"Convert {key} to Binary", total=len(dataloader), leave=False):
                img = batch['img'].to(device)
                label = batch['label']
                
                info = model(img)
                
                continuous_code = info['continuous_code'].cpu().data.numpy()
                binary_code = convert_c2b(continuous_code, method=gen_code_method).astype(np.int8)
                featuremap = info['featuremap'].cpu().data.numpy()
                
                
                if i == 0:
                    n_bit = binary_code.shape[-1]
                    n_data = len(dataloader.dataset)
                    code_info['B'] = np.zeros((n_data, n_bit), dtype=np.int8)
                    label_info_dict[key] = np.zeros((n_data, classes), dtype=np.int8)
                    
                    if return_continuous_code:
                        dim = continuous_code.shape[-1]
                        code_info['C'] = np.zeros((n_data, dim), dtype=np.float16)
                
                    if return_cnn_feat:
                        dim = featuremap.shape[-1]
                        code_info['F'] = np.zeros((n_data, dim), dtype=np.float16)
                    
                    idx = 0

                batch_size = len(img)
                code_info['B'][idx: idx+batch_size]  = binary_code.astype(np.int8)
                
                if label.ndim == 1:
                    label_info_dict[key][idx: idx+batch_size] = np.eye(classes.astype(np.int32), dtype=np.int8)[label]
                else:
                    label_info_dict[key][idx: idx+batch_size] = np.array(label,dtype=np.int8)
                
                if return_continuous_code:
                    code_info['C'][idx: idx+batch_size] = continuous_code.astype(np.float16)
                
                if return_cnn_feat:
                    code_info['F'][idx: idx+batch_size] = featuremap.astype(np.float16)
                
                idx += batch_size
                
    
    return code_info_dict['query'], code_info_dict['retrieval'], label_info_dict['query'], label_info_dict['retrieval']

