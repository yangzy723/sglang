import torch
from typing import Optional

from sglang.manager.kernel_manager import Kernel
from sgl_kernel import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul
from sgl_kernel import fused_add_rmsnorm, rmsnorm
from sgl_kernel import FusedSetKVBufferArg, apply_rope_with_cos_sin_cache_inplace

from .my_triton import compute_position_kernel, write_req_to_token_pool_triton


# Rebuilt-PyTorch/sglang-v0.5.4/python/sglang/srt/layers/activation.py
class GeluAndMulKernel(Kernel):
    def __init__(self, in_tensor: torch.Tensor, out_tensor: torch.Tensor):
        """
        初始化内核，保存对输入和输出张量的引用。
        
        参数:
            in_tensor (torch.Tensor): (x) 形状为 (..., D)
            out_tensor (torch.Tensor): (out) 预先分配的形状为 (..., D/2) 的张量
        """
        super().__init__()        
        self.in_tensor = in_tensor
        self.out_tensor = out_tensor
    
    def execute(self):
        """
        当调度器批准时，实际执行 silu_and_mul 操作。
        该操作会就地填充 self.out_tensor。
        """
        gelu_and_mul(self.in_tensor, self.out_tensor)


# Rebuilt-PyTorch/sglang-v0.5.4/python/sglang/srt/layers/activation.py
class GeluTanhAndMulKernel(Kernel):
    def __init__(self, in_tensor: torch.Tensor, out_tensor: torch.Tensor):
        super().__init__()        
        self.in_tensor = in_tensor
        self.out_tensor = out_tensor
    
    def execute(self):
        gelu_tanh_and_mul(self.in_tensor, self.out_tensor)


# Rebuilt-PyTorch/sglang-v0.5.4/python/sglang/srt/layers/activation.py
class SiluAndMulKernel(Kernel):
    def __init__(self, in_tensor: torch.Tensor, out_tensor: torch.Tensor):
        super().__init__()        
        self.in_tensor = in_tensor
        self.out_tensor = out_tensor

    def execute(self):
        silu_and_mul(self.in_tensor, self.out_tensor)


class FusedAddRMSNormKernel(Kernel):
    def __init__(self, x: torch.Tensor, residual: Optional[torch.Tensor], weight: torch.Tensor, variance_epsilon: float):
        super().__init__()        
        self.x = x
        self.residual = residual
        self.weight = weight
        self.variance_epsilon = variance_epsilon
    
    def execute(self):
        fused_add_rmsnorm(self.x, self.residual, self.weight, self.variance_epsilon)


class RMSNormKernel(Kernel):
    def __init__(self, x: torch.Tensor, weight: torch.Tensor, variance_epsilon: float, out_tensor: torch.Tensor):
        super().__init__()
        self.x = x
        self.weight = weight
        self.variance_epsilon = variance_epsilon
        self.out_tensor = out_tensor
    
    def execute(self):
        out = rmsnorm(self.x, self.weight, self.variance_epsilon)
        self.out_tensor.copy_(out)
    

class BatchQKApplyRotaryPosIdsCosSinCacheKernel(Kernel):
    def __init__(
        self, 
        positions: torch.Tensor, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        head_size: int, 
        cos_sin_cache: torch.Tensor, 
        is_neox: bool, 
        fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None
    ):
        super().__init__()
        self.positions = positions
        self.Q = Q
        self.K = K
        self.head_size = head_size
        self.cos_sin_cache = cos_sin_cache
        self.is_neox = is_neox

        # 可选的 fused kv cache 参数
        self.fused_set_kv_buffer_arg = fused_set_kv_buffer_arg
    
    def execute(self):
        apply_rope_with_cos_sin_cache_inplace(
            positions=self.positions,
            query=self.Q,
            key=self.K,
            head_size=self.head_size,
            cos_sin_cache=self.cos_sin_cache,
            is_neox=self.is_neox,
            # Compatible with old sgl-kernel
            **(
                dict(fused_set_kv_buffer_arg=self.fused_set_kv_buffer_arg)
                if self.fused_set_kv_buffer_arg is not None
                else {}
            ),
        )

# Rebuilt-PyTorch/sglang-v0.5.4/python/sglang/srt/model_executor/forward_batch_info.py
class ComputePositionKernel(Kernel):
    def __init__(
        self,
        positions: torch.Tensor,
        extend_start_loc: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        has_prefix: bool,
        batch_size: int
    ):
        super().__init__()
        self.positions = positions
        self.extend_start_loc = extend_start_loc
        self.extend_prefix_lens = extend_prefix_lens
        self.extend_seq_lens = extend_seq_lens
        self.has_prefix = has_prefix
        self.batch_size = batch_size
    
    def execute(self):
        compute_position_kernel[(self.batch_size,)](
        self.positions,
        self.extend_start_loc,
        self.extend_prefix_lens,
        self.extend_seq_lens,
        self.has_prefix,
    )

# Rebuilt-PyTorch/sglang-v0.5.4/python/sglang/srt/mem_cache/common.py
class WriteReqToTokenPoolTriton(Kernel):
    def __init__(
        self,
        req_to_token: torch.Tensor,
        req_pool_indices_tensor: torch.Tensor,
        prefix_pointers: torch.Tensor,
        prefix_lens_tensor: torch.Tensor,
        seq_lens_tensor: torch.Tensor,
        extend_lens_tensor: torch.Tensor,
        out_cache_loc: torch.Tensor,
        shape1: int,
        shape0: int
    ):
        super().__init__()
        self.req_to_token = req_to_token
        self.req_pool_indices_tensor = req_pool_indices_tensor
        self.prefix_pointers = prefix_pointers
        self.prefix_lens_tensor = prefix_lens_tensor
        self.seq_lens_tensor = seq_lens_tensor
        self.extend_lens_tensor = extend_lens_tensor
        self.out_cache_loc = out_cache_loc
        self.shape1 = shape1
        self.shape0 = shape0
    
    def execute(self):
        write_req_to_token_pool_triton[(self.shape0,)](
            self.req_to_token,
            self.req_pool_indices_tensor,
            self.prefix_pointers,
            self.prefix_lens_tensor,
            self.seq_lens_tensor,
            self.extend_lens_tensor,
            self.out_cache_loc,
            self.shape1
        )
