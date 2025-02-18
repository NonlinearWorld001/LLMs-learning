import math
import struct
import inspect
import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from typing import Any, Optional, Tuple
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from LMConfig import LMConfig

# 归一化层(类)
class RMSNorm(nn.Module):
    # 定义输入：1.张量特征维度； 2.防止除0的常数
    def __init__(self, dim: int, eps=1e-6):
        super().__init__()
        self.eps = eps #防止除以0的极小数
        self.weight = nn.Parameter(torch.Tensor(dim)) #将可学习的权重矩阵参数化标记（在pytorch中只有被nn.Parameter标记才能学习）

    # 均方根归一化
    def normalize(self, x):
        result = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return result

    # 前向传播函数
    def forward(self, x):
        output = self.normalize(x.float()).type_as(x)
        return output * self.weight

# 用于预计算旋转位置编码的复数形式(pos_cis), dim：嵌入空间的维度, end：序列最大长度， theta：缩放因子,用于生成一个与输入序列长度和模型维度相对应的复数位置编码。
# 可以理解为用两个维度的特征通过分别作为复数的实部和虚部来合成一个更复杂的特征，然后进行编码，既可以利用旋转信息，又可以节省计算资源？ 分为一组的特征实际上是正交的？
def precompute_pos_cis(dim: int, end:int, theta: float=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) #频率，或者说单位旋转角度
    # 为什么隔2取数？ 每个复数可以对应两个维度的旋转。
    # 截取前 dim//2 个元素？ 确保频率向量的长度是 dim//2，因为每个复数可以对应两个维度的旋转。
    # 为什么除以dim？ 归一化操作，保证不同维度下的频率一致性。
    t = torch.arange(end, device=freqs.device) # 生成序列（的数轴位置）
    freqs = torch.outer(t, freqs).float()
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)
    return pos_cis

# 将预计算好的复数形式的位置编码（pos_cis）应用到查询（xq）和键（xk）中，使模型在计算注意力时能感知词语的位置关系。
def apply_rotary_embedding(xq, xk, pos_cis):
    def unite_shape(pos_cis, x):
        # x：输入张量（xq 或 xk），形状为 (batch_size, seq_len, num_heads, head_dim)
        ndim = x.ndim #获取x的维度
        assert ndim > 1 #保证维度大于1
        assert pos_cis.shape == (x.shape[1], x.shape[-1]) # 保证pos_cis的大小和输入的 xq 或 xk 的 seq_len大小一致，以及RoPE位置编码和复数化的头维度匹配
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)] # shape = [1, seq_len, 1, head_dim//2]，为什么是head_dim//2 <-- 最后一维会在应用这个函数前拆分成[head_dim//2，2]
        return pos_cis.view(*shape) #把pos_cis扩展成形状为 shape 的张量，seq_len维度和head_dim维度已经对齐，直接存入，另外的增加两个一维向量方便广播机制
    x_q = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) #将最后一维拆分成2维度，ex：[2,3,4,8]-->[2,3,4,[4,2]]，然后将新的最后2维匹配成复数
    x_k = torch.view_as_complex(xk.float().reshape(*xq.shape[:-1], -1, 2)) #同上
    pos_cis = unite_shape(pos_cis, x_q) # 调整 pos_cis 的形状和 x_q兼容
    xq_out = torch.view_as_real(x_q * pos_cis).flatten(3) # x_q * RoPE：将旋转位置编码应用到查询向量上。然后转换回实数形式，然后从索引为3的维度展开为1维
    xk_out = torch.view_as_real(x_k * pos_cis).flatten(3) # 对 x_k 矩阵操作同上
    return xq_out.type_as(xq), xk_out.type_as(xk) # 按照原有的数据类型输出，保证计算的一致性

# kv头复制器：查询头数和键值头数会存在不一样的情况
def repeat_kv(x: torch.Tensor, n_repeat: int) -> torch.Tensor:
    # 获取输入张量的形状，其中：
    # batch_size: 批量大小
    # seq_len: 序列长度
    # n_kv_heads: 注意力头的数量
    # head_dim: 每个注意力头的维度
    batch_size, seq_len, n_kv_heads, head_dim = x.shape 

    if n_repeat == 1:
        return x 
    return x[:, :, :, None, :].expand(batch_size, seq_len, n_kv_heads, n_repeat, head_dim).reshape(batch_size, seq_len, n_kv_heads * n_repeat, head_dim)

class Attention(nn.Module): 
    def __init__(self, args: LMConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is not None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads