import argparse
import torch
import time
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import warnings
warnings.filterwarnings("ignore")
# from flag_gems.utils import libentry
# from flag_gems.utils.libentry import libtuner
# import flag_gems
# flag_gems.enable()
DEVICE = torch.device("cuda")

# @triton.heuristics(
# values={
# 'BLOCK_SIZE_M': lambda args: triton.next_power_of_2(args['M']) // 4,
# 'BLOCK_SIZE_N': lambda args: triton.next_power_of_2(args['N']) // 4,
# 'BLOCK_SIZE_K': lambda args: 32,
# 'GROUP_SIZE_M': lambda args: 8 ,
# })

@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K:
        tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
    return None 


