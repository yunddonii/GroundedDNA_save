import numpy as np

from sklearn.metrics import ndcg_score as compute_ndcg
from sklearn.metrics import DistanceMetric, pairwise
from tqdm import tqdm
from math import comb # We have to import this function to calculate combination operation 

from .quantizations import compute_seq_r, compute_gc_content
from .compress import make_hash_table, retrieve_items_less_than_distance, retrieve_items_with_distance

def compute_apt(binary_affinity, similarity, n_bit):
    Np = np.sum(binary_affinity)

    if Np== 0:
        return 0.0

    nD = np.bincount(similarity , minlength=n_bit+1)
    nDp = np.bincount(similarity[binary_affinity==1] , minlength=n_bit+1)
    
    assert (n_bit + 1) == nD.shape[0]
    assert (n_bit + 1) == nDp.shape[0]

    NDp = np.cumsum(nDp)
    ND = np.cumsum(nD)

    Np0 = np.zeros_like(NDp)
    N0 = np.zeros_like(ND)
    Np0[1:] = NDp[:-1]
    N0[1:] = ND[:-1]

    # 1. exact solution for no ties
    APt_i = nDp * NDp / (ND + np.finfo(float).eps)
    APt_i[ND == 0] = 0

    # 2. update where there are ties
    for l in np.where((nD > 1) & (nDp > 0))[0]:
        mult = (nDp[l] - 1) / (nD[l] - 1)
        nume = Np0[l] + np.arange(nD[l]) * mult + 1
        deno = N0[l] + np.arange(1, nD[l]+1)
        APt_i[l] = np.mean(nume / deno) * nDp[l]
    
    return np.sum(APt_i) / Np


def calculate_ap(binary_affinity, hamming_distance, topk=-1):
    idx_for_sort = np.argsort(hamming_distance)
    binary_affinity = binary_affinity[idx_for_sort]
    
    ap = calculate_ap_from_sorted_affinity(binary_affinity, topk=topk)

    return ap

def calculate_ap_from_sorted_affinity(sorted_binary_affinity, topk=-1):
    if topk == -1:
        topk = len(sorted_binary_affinity)+1
    
    bi_aff_topk = sorted_binary_affinity[:topk]
        
    # tsum number of items with same label
    tsum = int(np.sum(bi_aff_topk))
    if tsum == 0:
        return 0.0
        
    count = np.linspace(1, tsum, tsum)  # [1,2, tsum]
    tindex = np.asarray(np.where(bi_aff_topk == 1)) + 1.0
    ap = np.mean(count / (tindex))

    return ap


def calculate_bestworst_ap(binary_affinity, hamming_distance, topk, n_bit):
    sorted_idx = np.argsort(hamming_distance)
        
    sorted_bi_aff = binary_affinity[sorted_idx]
    sorted_dist = hamming_distance[sorted_idx]

    best_order_dist_list = []
    best_order_aff_list = []
    worst_order_dist_list = []
    worst_order_aff_list = []

    for b in range(0, n_bit+1):
        pos_sample_dist = sorted_dist[ (sorted_bi_aff == 1) & (sorted_dist== b)]
        neg_sample_dist = sorted_dist[(sorted_bi_aff == 0) & (sorted_dist== b)]
        pos_sample_bi_aff = np.ones_like(pos_sample_dist)
        neg_sample_bi_aff = np.zeros_like(neg_sample_dist)

        best_order_dist = np.concatenate((pos_sample_dist, neg_sample_dist), axis=0)
        worst_order_dist = np.concatenate((neg_sample_dist, pos_sample_dist), axis=0)

        best_order_aff = np.concatenate((pos_sample_bi_aff, neg_sample_bi_aff), axis=0)
        best_order_dist_list.append(best_order_dist)
        best_order_aff_list.append(best_order_aff)
        
        worst_order_aff = np.concatenate((neg_sample_bi_aff, pos_sample_bi_aff), axis=0)
        worst_order_dist_list.append(worst_order_dist)
        worst_order_aff_list.append(worst_order_aff)

    best_order_dist = np.concatenate( best_order_dist_list, axis=0)
    best_order_aff = np.concatenate(best_order_aff_list, axis=0)
    
    worst_order_dist = np.concatenate( worst_order_dist_list, axis=0)
    worst_order_aff = np.concatenate(worst_order_aff_list, axis=0)

    
    best_ap_topk = calculate_ap_from_sorted_affinity(best_order_aff, topk=topk)
    worst_ap_topk = calculate_ap_from_sorted_affinity(worst_order_aff, topk=topk)

    
    return best_ap_topk, worst_ap_topk



def calculate_ap_r(query_code, db_hash_table, binary_affinity, continuous_distance, eval_max_radius, base):
    # Paper: Deep Cauchy Hashing - link( http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-cauchy-hashing-cvpr18.pdf )
    # 1) Pruning, to return data points within Hamming radius 2
    # 2) Scanning, to re-rank the returned data points in ascending order of their distances to each query using "continuous codes"
    
    retrieved_sample_idx, __ = retrieve_items_less_than_distance(db_hash_table, query_code, base, eval_max_radius)
    
    idx_for_sort = np.argsort(continuous_distance[retrieved_sample_idx])
    
    bi_aff_ = binary_affinity[retrieved_sample_idx][idx_for_sort]
    
    ap_r = calculate_ap_from_sorted_affinity( bi_aff_ )
    
    return ap_r

def calculate_RAMAP(query_code, db_hash_table, binary_affinity, eval_max_radius, n_bit, base):
    samples_by_dist = np.zeros(shape=(eval_max_radius+1))
    tps_by_dist     = np.zeros(shape=(eval_max_radius+1))
    comb_by_dist    = np.zeros(shape=(eval_max_radius+1))
    
    for r in range(eval_max_radius+1):
        retrieved_sample_idx, __ = retrieve_items_with_distance(db_hash_table, query_code, base, r)
        samples_by_dist[r] = len(retrieved_sample_idx)
        tps_by_dist[r] = np.sum( binary_affinity[retrieved_sample_idx])
        comb_by_dist[r] = comb(n_bit, r) * (base - 1)**r
        
    
    cum_samples_by_dist = np.cumsum(samples_by_dist)
    cum_tp_by_dist      = np.cumsum(tps_by_dist)
    cum_comb_by_dist    = np.cumsum(comb_by_dist)
    
    weight_   = 1.0 / cum_comb_by_dist
    precision = cum_tp_by_dist / (cum_samples_by_dist + np.finfo(float).eps)
    ramap     = np.mean((precision * weight_))
    
    return ramap

def calculate_LGAP(query_code, db_hash_table, binary_affinity, eval_max_radius, n_bit, base):
    samples_by_dist     = np.zeros(shape=(eval_max_radius+1))
    tps_by_dist         = np.zeros(shape=(eval_max_radius+1))
    max_samples_by_dist = np.zeros(shape=(eval_max_radius+1))
    comb_by_dist        = np.zeros(shape=(eval_max_radius+1))

    
    for r in range(eval_max_radius+1):
        retrieved_sample_idx, retrieved_sample_key = retrieve_items_with_distance(db_hash_table, query_code, base, r)
        samples_by_dist[r] = len(retrieved_sample_idx)
        tps_by_dist[r] = np.sum( binary_affinity[retrieved_sample_idx])
        comb_by_dist[r] = comb(n_bit, r) * (base - 1)**r
        
        __, counts  = np.unique(retrieved_sample_key, return_counts=True)
        max_samples_by_dist[r] = np.max(counts) if len(counts) != 0 else 0


    cum_samples_by_dist = np.cumsum(samples_by_dist) 
    cum_hash_codes_by_dist = np.cumsum(comb_by_dist) 
    cum_tp_by_dist = np.cumsum(tps_by_dist)
    
    max_cum_samples_by_dist = np.maximum.accumulate(max_samples_by_dist)
    
    penalty = cum_samples_by_dist / (  max_cum_samples_by_dist * cum_hash_codes_by_dist + np.finfo(float).eps)
    
    precision = cum_tp_by_dist / (cum_samples_by_dist + np.finfo(float).eps)
    
    lgap = np.mean( (precision * penalty))
    
    return lgap


def calculate_hamming_distance(query_codes, retrieval_codes):
    hamming_distances = {}
    hamming = DistanceMetric.get_metric("hamming")

    for code_type in query_codes.keys():
        if code_type == "C":
            continue
        print(f"Get {code_type} Code Hamming Distance")
        n_bit = retrieval_codes[code_type].shape[-1]
        hamming_distances[code_type] = (hamming.pairwise(query_codes[code_type], retrieval_codes[code_type]) * n_bit).astype(np.int8) # n bit must be less than 126
    
    return hamming_distances

def calculate_cosine_distance(continuous_query_codes, continuous_retrieval_codes):
    print(f"Get Continuous Code Cosine Distance")
    distances = pairwise.cosine_distances(continuous_query_codes, continuous_retrieval_codes)
    return distances

def calculate_affinity(query_labels, retrieval_labels):
    affinities = {}
    affinities['integer'] = np.dot(query_labels, retrieval_labels.T).astype(np.int8) # n class must be less than 127
    affinities['binary'] = (affinities['integer'] > 0).astype(np.int8)
    return affinities



def evaluate_retrieval_model(query_codes, retrieval_codes, query_labels, retrieval_labels, affinity_info:dict=None, db_hash_tables:dict=None, mAP_topN=True, mAP_R=True, mAPt=False, mNDCG=False, RAMAP=False, mLGAP=False, seq_r=False, gc_content=False, max_seq_r=3, eval_max_radius=2):
    code_types = list(query_codes.keys())
    
    assert list(retrieval_codes.keys()) == code_types
    assert 'C' in code_types
    
    if seq_r or gc_content:
        quat_code_types = [code_type for code_type in code_types if 'Q' in code_type]
        assert len(quat_code_types) > 0, "There is no information about quaternary codes. Please enable the option '--convert_quat_code_with_constraints'"
    
    n_bit_dict = {}
    base_dict = {}
    
    for code_type in code_types:
        if 'B' in code_type:
            n_bit_dict[code_type] = query_codes['B'].shape[-1]
            base_dict[code_type] = 2
            
        elif 'Q' in code_type:
            n_bit_dict[code_type] = query_codes['B'].shape[-1] // 2
            base_dict[code_type] = 4
            assert int(n_bit_dict['B']/2) == n_bit_dict['B']/2
            
        elif 'C' in code_type:
            n_bit_dict[code_type] = query_codes['B'].shape[-1]
            base_dict[code_type] = None
        
        else:
            raise NotImplementedError(f"Unknown Type {code_type}")


    # 임시 조치: code {-1, +1}^{n_bit} -> code {+0, +1}^{n_bit}
    for code_type in code_types:
        query_code = query_codes[code_type]
        retrieval_code = retrieval_codes[code_type]
        
        if 'B' in code_type:
            query_codes[code_type] = ( query_code == +1).astype(np.uint8)
            retrieval_codes[code_type] = ( retrieval_code == +1).astype(np.uint8)
    

    
    if affinity_info == None:
        print("Get Affinities")
        affinities = calculate_affinity(query_labels, retrieval_labels)
    else:
        print("Load Affinities")
        affinities = affinity_info
        assert 'integer' in affinities.keys()
        assert 'binary' in affinities.keys()
        
    
    if db_hash_tables == None:  
        db_hash_tables = {}
        
    if RAMAP or mLGAP or mAP_R:       
        for code_type, codes in retrieval_codes.items():
            if code_type == "C":
                continue
            
            if code_type in db_hash_tables.keys():
                print(f"Load Hash Table: {code_type}")
            
            else:
                print(f"Get Hash Table: {code_type}")
                base = base_dict[code_type]
                db_hash_tables[code_type] = make_hash_table(codes, base)
        
    distances = calculate_hamming_distance(query_codes, retrieval_codes) 
    distances['C'] = calculate_cosine_distance(query_codes['C'], retrieval_codes['C'])
    mAP_topN_topk_tag_list = ['ALL', "5000", "1000", "10"]
    mAP_topN_topk_k_list = [-1, 5000, 1000, 10]
    eval_max_radius_list = [i+1 for i in range(eval_max_radius)]

    # Initialize Scores
    eval_score = {}
    for code_type in code_types:
        if mAP_topN:
            for tag in mAP_topN_topk_tag_list:
                eval_score[f'mAP_{code_type}@topN{tag}'] = 0

        if 'C' in code_type:
            continue
        
        if mAP_topN:
            for tag in mAP_topN_topk_tag_list:
                eval_score[f'mAP_{code_type}_best@topN{tag}'] = 0
                eval_score[f'mAP_{code_type}_worst@topN{tag}'] = 0
        
        if mAPt:
            eval_score[f'mAP_{code_type}_t'] = 0
            
        if mNDCG:
            eval_score[f'mNDCG_{code_type}'] = 0
        
        if mAP_R:
            for tag in eval_max_radius_list:
                eval_score[f'mAP_{code_type}@R<={tag}'] = 0
            
        if RAMAP:
            for tag in eval_max_radius_list:
                eval_score[f'RAMAP_{code_type}@<={tag}'] = 0

        
        if mLGAP:
            for tag in eval_max_radius_list:
                eval_score[f'mLGAP_{code_type}@<={tag}'] = 0
        

    for  i in tqdm(range(len(query_labels)), desc="Evaluation..", total=len(query_labels), leave=False):
        integer_affinity = affinities['integer'][i]
        binary_affinity = affinities['binary'][i]

        for code_type in code_types:
            distance = distances[code_type][i]
            query_code = query_codes[code_type][i]
            
            if ('C' in code_type) and (mAP_topN):
                for topk in mAP_topN_topk_k_list:
                    tag = "ALL" if topk == -1 else topk
                    eval_score[f'mAP_{code_type}@topN{tag}'] += calculate_ap(binary_affinity, distance, topk=topk)
            
            else:
                distance = distance.astype(np.int64)
                base = base_dict[code_type]
                n_bit = n_bit_dict[code_type]
                
                similarity = n_bit - distance
                
                if mAP_topN:
                    for topk in mAP_topN_topk_k_list:
                        tag = "ALL" if topk == -1 else topk

                        eval_score[f'mAP_{code_type}@topN{tag}'] += calculate_ap(binary_affinity, distance, topk=topk)
                        best_ap_topk, worst_ap_topk = calculate_bestworst_ap(binary_affinity, distance, topk=topk, n_bit=n_bit)
                        eval_score[f'mAP_{code_type}_best@topN{tag}'] += best_ap_topk
                        eval_score[f'mAP_{code_type}_worst@topN{tag}'] += worst_ap_topk

                    
                if mAPt:
                    eval_score[f'mAPt_{code_type}'] += compute_apt(binary_affinity, distance.astype("int64"), n_bit=n_bit)

                if mNDCG:
                    eval_score[f'mNDCG_{code_type}'] += compute_ndcg(np.expand_dims(integer_affinity, 0), np.expand_dims(similarity, 0))
                    
                if RAMAP or mLGAP or mAP_R:    
                    db_hash_table = db_hash_tables[code_type]
                
                    if mAP_R:
                        cosine_distance = distances['C'][i]
                        for max_radius in eval_max_radius_list:
                            eval_score[f'mAP_{code_type}@R<={max_radius}'] += calculate_ap_r(query_code, db_hash_table, binary_affinity, cosine_distance, max_radius, base)
                        
                    if RAMAP:
                        for max_radius in eval_max_radius_list:
                            eval_score[f'RAMAP_{code_type}@<={max_radius}'] += calculate_RAMAP(query_code, db_hash_table, binary_affinity, max_radius, n_bit, base)
                    
                    if mLGAP:
                        for max_radius in eval_max_radius_list:
                            eval_score[f'mLGAP_{code_type}@<={max_radius}'] += calculate_LGAP(query_code, db_hash_table, binary_affinity, max_radius, n_bit, base)
            
        
    for name in eval_score.keys(): # mean
        eval_score[name] /= len(query_labels)
    
    result= {}
    result['eval_score'] = eval_score
        
    if seq_r:
        seq_r = {}
        for code_type in quat_code_types:
            seq_r[f'query{code_type}'] =  compute_seq_r(query_codes[code_type])
            seq_r[f'retrieval{code_type}'] = compute_seq_r(retrieval_codes[code_type])

        seq_r_info = {}

        for type_name, dist_info in seq_r.items():
            for val_info, value in dist_info.items():
                new_name = f"seq_r-{type_name}-{val_info:02d}"
                seq_r_info[new_name] = value
        
        result['seq_r_info'] = seq_r_info
        
    
    if gc_content:
        gc_content = {}
        
        for code_type in quat_code_types:
            gc_content[f'query{code_type}'] =  compute_gc_content(query_codes[code_type])
            gc_content[f'retrieval{code_type}'] = compute_gc_content(retrieval_codes[code_type])
                

        gc_content_info = {}
        for type_name, dist_info in gc_content.items():
            for val_info, value in dist_info.items():
                new_name = f"gc_content-{type_name}-{val_info:.4f}"
                gc_content_info[new_name] = value
                
        result['gc_info'] = gc_content_info

    return result
