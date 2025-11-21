"""
Matrix Multiplication
=====================
In this tutorial, matmul on CPU with different input layouts is tested.

This tutorial is optimized for AMX-enabled CPUs.

"""

# %%
# Kernels
# -------

import torch

import triton
import triton.language as tl
import os
import math

DTYPE = os.getenv("DTYPE", "float16")
in_dtype = getattr(torch, DTYPE)
out_dtype = getattr(torch, DTYPE)
# Choose block size depending on dtype. We have more register
# capacity for bfloat16/float16 compared to float32.
BLOCK_SIZE_M = 8
BLOCK_SIZE_N = 8
BLOCK_SIZE_K = 8 
# GROUP_SIZE_M = 8

# todos: a placeholder to be python binded to the kai rhs pack function in C++
def get_rhs_packed_size(BLOCK_SIZE_K, BLOCK_SIZE_N):
    # each block packs BLOCK_SIZE_K x BLOCK_SIZE_N elements
    return BLOCK_SIZE_K * BLOCK_SIZE_N

@triton.jit
def pack_rhs_kernel_blockptr(rhs_ptr,      # input matrix B base pointer (K, N)
                             packed_ptr,   # output packed buffer base pointer
                             K, N,
                             rhs_stride,   # leading stride (elements per row)
                             bias_ptr,
                             BLOCK_SIZE_K: tl.constexpr,
                             BLOCK_SIZE_N: tl.constexpr,
                             PACKED_SIZE: tl.constexpr # Size of each packed block
                             ): 
    """
    Packs B (shape K x N) into a blocked buffer with tile sizes (BLOCK_SIZE_K, BLOCK_SIZE_N).
    Grid mapping: one program instance per K-tile (grid = (num_k_blocks,)); each instance loops over N-tiles.
    """
    pid = tl.program_id(0)
    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
    num_n_blocks = tl.cdiv(N, BLOCK_SIZE_N)

    k_block = pid % num_k_blocks
    k_base = k_block * BLOCK_SIZE_K
    n_base = 0

    packed_base = k_block * num_n_blocks * PACKED_SIZE;

    # build a block pointer for the source B: block shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
    rhs_block_ptr = tl.make_block_ptr(base=rhs_ptr,
                            shape=(K, N),
                            strides=(rhs_stride, 1),
                            offsets=(k_base, n_base),
                            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                            order=(0, 1))
    
    # packed block ptr points at  a flattened buffer of consists of mutple block of size PACKED_SIZE.
    packed_block_ptr = tl.make_block_ptr(base=packed_ptr,
                            shape=(PACKED_SIZE,),
                            strides=(1,),
                            offsets=(packed_base,),
                            block_shape=(PACKED_SIZE,),
                            order=(0,))

    if bias_ptr != None:
        bias_block_ptr = tl.make_block_ptr(base=bias_ptr,
                                shape=(N,),
                                strides=(1,),
                                offsets=(n_base,),
                                block_shape=(BLOCK_SIZE_N,),
                                order=(0,))
    else:
        bias_block_ptr = None

    for n_block in tl.range(0, num_n_blocks):
        n_base = n_block * BLOCK_SIZE_N

        # load the full block (masked automatically by block_ptr semantics)
        rhs_blk = tl.load(rhs_block_ptr, mask=None, other=0)

        # optional bias
        if bias_ptr != 0:
            n_offsets = n_base + tl.arange(0, BLOCK_SIZE_N)
            n_mask = n_offsets < N
            bias_block = tl.load(bias_ptr + n_offsets, mask=n_mask, other=0)
        else:
            bias_block = None
        
        # store into packed destination (same flattened layout as before)
        out_block_index = k_block * num_n_blocks + n_block
        out_block_base = out_block_index * PACKED_SIZE

        #todos: run_rhs_pack
        rhs_packed = run_rhs_pack(BLOCK_SIZE_K, BLOCK_SIZE_N, rhs_stride, rhs_blk, bias_block, rhs_stride, rhs_blk, bias_block)
        tl.store(packed_block_ptr, rhs_packed)

        rhs_block_ptr = tl.advance(rhs_block_ptr, 0, BLOCK_SIZE_N) 
        if bias_ptr != 0:
            bias_block_ptr = tl.advance(bias_block_ptr, BLOCK_SIZE_N)
        
        packed_block_ptr = tl.advance(packed_block_ptr, PACKED_SIZE)
        

def pack_rhs(rhs, rhs_stride, bias, K, N, BLOCK_SIZE_K, BLOCK_SIZE_N):
    # clearer names for counts of blocks/tiles
    num_k_blocks = (K + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    num_n_blocks = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    rhs_packed_size = get_rhs_packed_size(BLOCK_SIZE_K, BLOCK_SIZE_N)

    # Output buffer flattened
    out = torch.empty(num_k_blocks * num_n_blocks * rhs_packed_size, dtype=rhs.dtype, device=rhs.device)

    # pointers for Triton: use tensors directly (Triton will take buffer ptrs)
    rhs_ptr = rhs
    pack_ptr = out
    bias_ptr = bias if bias is not None else 0

    # launch one program per K_tile; each program loops N_tiles
    grid = (num_k_blocks,)
    pack_rhs_kernel[grid](rhs_ptr, pack_ptr, K, N, rhs_stride, bias_ptr, BLOCK_SIZE_K, BLOCK_SIZE_N, rhs_packed_size)

    return out

# %%
# Unit Test
# ---------
#
# We can test our custom matrix multiplication operation against a native torch implementation.
torch.manual_seed(0)

triton.runtime.driver.set_active_to_cpu()

M,  N, K = 32, 32, 32

if in_dtype.is_floating_point:
    a = torch.randn((M, K), device='cpu', dtype=in_dtype)
    b = torch.randn((K, N), device='cpu', dtype=in_dtype)
else:
    a = torch.randint(0, 5, (M, K), device='cpu', dtype=in_dtype)
    b = torch.randint(0, 5, (K, N), device='cpu', dtype=in_dtype)
c = torch.empty((M, N), device='cpu', dtype=out_dtype)
# torch_output = torch.matmul(a.to(out_dtype), b.to(out_dtype))
rtol = 0
a_tmp = torch.zeros((M * K), device='cpu', dtype=in_dtype)
b_tmp = torch.zeros((K * N), device='cpu', dtype=in_dtype)


# triton_output = matmul(a, b, c, a_tmp, b_tmp, M, N, K, True, False, False, False, False, False)
# if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
#     print("✅ TritonCPU and TorchCPU match")
# else:
#     print("❌ TritonCPU and TorchCPU differ, the maximum difference is "
#           f'{torch.max(torch.abs(triton_output - torch_output))}')
#     assert False
# triton_output = matmul(a, b, c, a_tmp, b_tmp, 512, 512, 512, False, True, True, True, True, DTYPE != "float32")
# if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
#     print("✅ TritonCPU pre-packed and TorchCPU match")
# else:
#     print("❌ TritonCPU pre-packed and TorchCPU differ, the maximum difference is "
#           f'{torch.max(torch.abs(triton_output - torch_output))}')
#     assert False

