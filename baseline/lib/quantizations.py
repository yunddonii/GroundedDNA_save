import torch
import numpy as np
from itertools import zip_longest



def rank_and_assign(U, axis=0, n_zero_pad = 0):
    if type(U) == torch.Tensor:
        u_dtype = "tensor"
        device = U.device
    else:
        u_dtype = "numpy"
        U = torch.tensor(U)
        device = "cpu"

    if axis == 0:
        _, index = U.sort(0, descending=True)
        N, D = U.shape
        n_plus = N // 2  - n_zero_pad // 2
        n_zero = n_zero_pad
        n_minus = N - n_plus - n_zero
        
        assert (n_plus > 0) and (n_minus > 0) , f"n_zero_pad/2 must be smaller than batchsize/2: n_zero_pad: {n_zero_pad}, batchsize: {N}"

        B_creat = torch.cat((torch.ones(n_plus, D), torch.zeros( n_zero_pad , D), -torch.ones(n_minus, D)), axis=0).to(device)
        B = torch.zeros(U.shape).to(device).scatter_(0, index, B_creat)

    elif axis == 1:
        _, index = U.sort(1, descending=True)
        N, D = U.shape
        n_plus = D // 2  - n_zero_pad // 2
        n_zero = n_zero_pad
        n_minus = D - n_plus - n_zero

        B_creat = torch.cat((torch.ones(N, n_plus), torch.zeros( N , n_zero_pad), -torch.ones(N,n_minus)), dim=1).to(device)
        B = torch.zeros(N, D).to(device).scatter_(1, index, B_creat)

    else:
        raise NotImplementedError
    if u_dtype == "tensor":
        return B # {-1, +1}
    else:
        return B.numpy() # {-1, +1}^n_bit


def convert_b2q(binaries):
    if (binaries.min() == -1) and (binaries.max() == +1):
        binaries = (binaries + 1) / 2
    elif (binaries.min() == 0) and (binaries.max() == +1):
        pass
    else:
        raise ValueError("Binary Code must contain {-1,+1} or {0,1}")
    
    quaternaries =  binaries[:, ::2]*2 +  binaries[:, 1::2]
    return quaternaries # {0,1,2,3}^n_bit

def convert_u2b_with_gcconstrant(continuous_code, min_gc_content=0.25, max_gc_content=0.75):
    
    N, D = continuous_code.shape

    if D/2 != int(D/2):
        raise ValueError(f"Code dimension should be even number")

    if min_gc_content != (1-max_gc_content):
        raise NotImplementedError
    
    min_gc_num = D/2 * min_gc_content
    num_zeros = int(D/2 - min_gc_num*2)

    if min_gc_num != int(min_gc_num):
        raise ValueError(f"min_gc_content {min_gc_content} x  D/2 {D/2} is must be integer type")

    
    masked_binaries = rank_and_assign(continuous_code[:, 0::2], axis=1, n_zero_pad=num_zeros)
    B = np.zeros((N, D))
    B[:,0::2] = masked_binaries
    sign_position = B == 0
    B[sign_position] = np.sign(continuous_code)[sign_position]

    return B # {-1, +1}^n_bit


def compute_seq_r(quat_codes):
    N, code_len = quat_codes.shape
    max_seq_r = np.zeros(N, dtype=np.int64)
    current_seq_r = np.zeros(N, dtype=np.int64)
    current_bit = quat_codes[:, 0]
    for comparing_bit in quat_codes.transpose(1,0)[1:]:
        repeated_bit_idx = (current_bit == comparing_bit)
        current_seq_r[repeated_bit_idx] += 1
        current_seq_r[~repeated_bit_idx] = 0
        max_seq_r_idx = max_seq_r < current_seq_r
        max_seq_r[max_seq_r_idx] = current_seq_r[max_seq_r_idx]
        current_bit = comparing_bit

    max_seq_r += 1

    seq_r_dist = np.bincount(max_seq_r)[1:]
    seq_r_info = { i : count for i, count in zip_longest( range(1, code_len+1), seq_r_dist, fillvalue=0)}
   
    return seq_r_info

def compute_gc_content(quat_codes):
    __, code_len = quat_codes.shape
    gc_idx = ( quat_codes == 2 ) | ( quat_codes == 3 )
    gc_contents = gc_idx.sum(-1)
    gc_content_dist = np.bincount(gc_contents)
    gc_content_info = { i/code_len : count for i, count in zip_longest( range(code_len+1), gc_content_dist, fillvalue=0)}
    return gc_content_info


def get_max_seq_r_segment(quat_codes, max_seq_r=3):
    if type(quat_codes) == torch.Tensor:
        dtype="tensor"
        device=quat_codes.device
    elif type(quat_codes) == np.ndarray:
        dtype = "numpy"
        quat_codes = torch.tensor(quat_codes)
        device="cpu"
    else:
        raise ValueError("This function only supports np.ndarray or torch.Tensor")
    
    N, code_len = quat_codes.shape

    max_seq_r_in_code = torch.zeros(N, dtype=torch.int64, device=device)
    current_seq_r = torch.zeros(N, dtype=torch.int64, device=device)
    current_bit = quat_codes[:, 0]

    cur_seq_r_bit_start_idx = torch.zeros(N, dtype=torch.int64, device=device)
    max_seq_r_bit_start_idx = torch.zeros(N, dtype=torch.int64,device=device)

    for i, comparing_bit in enumerate( quat_codes.transpose(1,0)[1:] ):
        repeated_bit_idx = (current_bit == comparing_bit)
        current_seq_r[repeated_bit_idx] += 1
        current_seq_r[~repeated_bit_idx] = 0

        cur_seq_r_bit_start_idx[~repeated_bit_idx] = i+1

        max_sample_idx = max_seq_r_in_code < current_seq_r
        max_seq_r_in_code[max_sample_idx] = current_seq_r[max_sample_idx]
        max_seq_r_bit_start_idx[max_sample_idx] = cur_seq_r_bit_start_idx[max_sample_idx]

        current_bit = comparing_bit

    max_seq_r_in_code += 1

    bigger_than_start = torch.arange(0, code_len,device=device) >= torch.unsqueeze(max_seq_r_bit_start_idx, dim=1)
    lower_than_end = torch.arange(0, code_len,device=device) < torch.unsqueeze(max_seq_r_bit_start_idx + max_seq_r_in_code, dim=1)
    is_exceed = torch.unsqueeze(max_seq_r_in_code > max_seq_r, dim= 1)

    mask = bigger_than_start & lower_than_end &is_exceed

    mask = mask.repeat_interleave(2, dim=1)
    seg_start_idx = max_seq_r_bit_start_idx*2
    seg_len = max_seq_r_in_code*2

    if dtype=="numpy":
        mask = np.array(mask)
        seg_start_idx = np.array(seg_start_idx)
        seg_len = np.array(seg_len)


    return seg_start_idx, seg_len, mask

def convert_b2bq_with_seq_r_constrants(bi_codes, max_seq_r=3):
    if type(bi_codes) == torch.Tensor:
        dtype="tensor"
    elif type(bi_codes) == np.ndarray:
        dtype = "numpy"
        bi_codes = torch.tensor(bi_codes)
    else:
        raise ValueError("This function only supports np.ndarray or torch.Tensor")
    
    N, D = bi_codes.shape
    device = bi_codes.device

    assert max_seq_r > 2

    while True:
        quat_codes = convert_b2q(bi_codes)
        # 가장 Redundancy가 큰 위치 찾기 -> max_seq_r보다 크다면 True, 아니면 False
        start_idx, seg_len, seg_mask = get_max_seq_r_segment(quat_codes, max_seq_r=max_seq_r)

        if seg_mask.sum() == 0:
            break

        # Segment의 가운데 인덱스를 선택
        middle_idx = start_idx + torch.div(seg_len,4, rounding_mode="trunc")*2 + 1 
        invert_mask = torch.zeros(N,D, device=device).bool()
        invert_mask[torch.arange(N, device=device), middle_idx] = True
        invert_mask &= seg_mask
        
        bi_codes[invert_mask] = -bi_codes[invert_mask]
    
    if dtype == "numpy":
        bi_codes, quat_codes = np.array(bi_codes), np.array(quat_codes)
    else:
        bi_codes, quat_codes = bi_codes.to(device), quat_codes.to(device)


    return bi_codes, quat_codes