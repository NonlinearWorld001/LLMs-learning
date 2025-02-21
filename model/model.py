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
    t = torch.arange(end, device=freqs.device) # 生成序列（的数轴位置）向量
    freqs = torch.outer(t, freqs).float() # 生成频率矩阵，每个元素表示对应位置的频率
    pos_cis = torch.polar(torch.ones_like(freqs), freqs) # 生成复数形式的旋转位置编码，每个元素表示对应位置的复数形式
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

# kv头复制器：查询头数和键值头数会存在不一样的情况（当键值头数量少于查询头时）， 该函数将键值头的特征复制n_repeat次，使得键值头的数量和查询头的数量一致
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
    # x[:, :, :, None, :]：在每个头维度上增加一个维度，扩展为[batch_size, seq_len, n_kv_heads, 1, head_dim]
    # expand(...):广播机制扩展维度，重复引用head_dim维度，扩展为[batch_size, seq_len, n_kv_heads, n_repeat, head_dim]
    # reshape(...):将扩展后的张量展平为[batch_size, seq_len, n_kv_heads * n_repeat, head_dim]

# 注意力层(类)， 这里使用的是分组注意力机制，和LLAMA的注意力机制类似
class Attention(nn.Module): 
    def __init__(self, args: LMConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is not None else args.n_kv_heads # 如果键值头数量为None，则使用查询头数量
        assert args.n_heads % self.n_kv_heads == 0 # 确保查询头数量是键值头数量的整数倍，这是为了矩阵乘法时，键值头数量和查询头数量一致
        self.n_local_heads = args.n_heads # 查询头数量，默认16
        self.n_local_kv_heads = self.n_kv_heads # 键值头数量，默认8
        self.n_rep = self.n_local_heads // self.n_local_kv_heads # 每个键值头对应的查询头数量，即用于输入repeat_kv的n_repeat参数
        self.head_dim = args.dim // args.n_heads # 每个头的维度，注意力模块的总维度由所有注意力头平分， 默认512/16=32

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False) # 查询权重矩阵，输入维度为dim，输出维度为n_heads * head_dim，是n_heads个head_dim的并行计算，无偏置的线性变换层
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False) # 键权重矩阵，输入维度为dim，输出维度为n_kv_heads * head_dim，是n_kv_heads个head_dim的并行计算，无偏置的线性变换层
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False) # 值权重矩阵，输入维度为dim，输出维度为n_kv_heads * head_dim，是n_kv_heads个head_dim的并行计算，无偏置的线性变换层    
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False) # 输出权重矩阵，输入维度为n_heads * head_dim，输出维度为dim，是n_heads个head_dim的并行计算，无偏置的线性变换层
        self.k_cache, self.v_cache = None, None # 键值缓存，用于存储当前状态之前的时间步的键和值的缓存，加速推理过程

        self.attn_dropout = nn.Dropout(args.dropout) # 注意力概率矩阵的随机失活层，在注意力权重s矩阵oftmax输出之后，在训练时随机丢弃部分注意力连接，防止模型过度依赖局部模式
        self.resid_dropout = nn.Dropout(args.dropout) # 残差连接的随机失活层，在注意力输出与残差连接相加之后，在训练时随机丢弃部分残差连接，防止网络过度适应特定路径，增强模型鲁棒性
        self.dropout = args.dropout # 随机失活的正则化比例，本质是在防止过拟合
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and args.flash_attn 
        # hasattr函数，用于检查某个对象是否具有某个属性或方法
        # 检查当前环境是否支持 Flash Attention， 并根据配置决定是否启用 Flash Attention（Flash Attention 是一种高效的注意力机制实现，利用分块计算和内存优化，能够显著加速 Transformer 模型中的注意力计算）

        mask = torch.full((1, 1, args.max_seq_length, args.max_seq_length), float("-inf")) # 4维张量适配多头注意力机制，生成一个形状为(batch, head, seq, seq)的掩码，用于屏蔽未来信息
        # 在自注意力机制中，softmax函数用于计算注意力权重，它将输入值转换为概率分布。如果掩码位置的值是 -inf，softmax函数会将这些位置的输出概率推向0，从而有效地忽略这些位置。
        mask =torch.triu(mask, diagonal=1)  # 保留上三角部分（实现因果注意力），下三角为0，表示允许注意力，这保证了只能看到当前位置及之前的部分的token（未来的状态仅由当前的状态决定）
        self.register_buffer("mask", mask, persistent=False) # 将掩码注册为（不可学习的）模型参数，仅在推理时使用，避免在每次前向传播时重新计算，persistent=False则表明掩码不被存入模型的state_dict中，每次动态生成。

    '''
    * 网络结构示意图：
                   输入 → wq → Q       ──┐       
                   输入 → wk → K → repeat_kv → K' ──┤ scaled_dot_product_attention → 输出 → wo → 最终输出
                   输入 → wv → V → repeat_kv → V' ──┘
    '''
    
    # Attention中的前向传播函数（方式）  
    def forward(self, x:torch.Tensor, pos_cis:torch.Tensor, kv_cache=False):
        bch_size, seqlen, _ = x.shape # 获取输入张量的batch_size和seq_len, x 的形状为[batch_size, seq_len, dim]
        '''举个例子 x.shape = [2,2,512] 
                                x = [[[1,2,3,4, ...],
                                      [5,6,7,8, ...]],  <--序列1
                                     [[9,10,11,12, ...],
                                      [13,14,15,16, ...]] <--序列2 ]
        '''

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x) # 将输入张量 x 分别通过 wq、wk 和 wv 线性变换层，得到查询向量Q、键向量K、值向量V
        xq = xq.view(bch_size, seqlen, self.n_local_heads, self.head_dim) # 将查询向量Q、键向量K、值向量V展平为(batch_size, seq_len, n_local_heads, head_dim)
        xk = xk.view(bch_size, seqlen, self.n_local_kv_heads, self.head_dim) # 将键向量K、值向量V展平为(batch_size, seq_len, n_local_kv_heads, head_dim)
        xv = xv.view(bch_size, seqlen, self.n_local_kv_heads, self.head_dim) # 将值向量V展平为(batch_size, seq_len, n_local_kv_heads, head_dim)

        xq, xk = apply_rotary_embedding(xq, xk, pos_cis) # 将查询向量Q、键向量K应用旋转位置编码

        # 一种更加高效的推理方式，在推理时，使用键值缓存来加速推理过程
        # 避免重复计算：历史token的键值不再重新计算，而是直接使用缓存中的值
        # 内存效率：只需存储O(n)的键值缓存，而非O(n²)的注意力矩阵
        if kv_cache and self.eval(): # 如果kv_cache为True(启用键值缓存)，并且模型处于推理模式
            if seqlen == 1 and all(cache is not None for cache in (self.k_cache, self.v_cache)): # 如果序列长度为1(表示在自回归阶段，正在生成单个新token)，并且k_cache和v_cache都存在
                xk = torch.cat((self.k_cache, xk), dim=1) # 把k_cache和xk沿着第一维拼接起来,将当前计算的键与历史缓存拼接，保持完整的上下文记忆
                xv = torch.cat((self.v_cache, xv), dim=1) # 把v_cache和xv沿着第一维拼接起来,将当前计算的值与历史缓存拼接，保持完整的上下文记忆
            self.k_cache, self.v_cache = xk, xv # 更新k_cache和v_cache,提供一个新的当前状态

        xk = repeat_kv(xk, self.n_rep) # 将键向量K复制n_rep次，使得键值头的数量和查询头的数量一致

        xq = xq.transpose(1, 2) 
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        # 将xq、xk、xv的维度从[batch_size, seq_len, n_local_heads, head_dim]转换为[batch_size, n_local_heads, seq_len, head_dim]
        
        if self.flash and seqlen != 1: # 在flash可用且序列长度不为1时，使用flash注意力机制加速推理
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, 
                                                                      attn_mask=None, 
                                                                      dropout_p=self.dropout if self.training else 0.0, 
                                                                      is_causal=True)
        else:# 在flash不可用或序列长度为1时，使用标准实现
            attn_weights = torch.matmul(xq, xk.transpose(2, 3))/ math.sqrt(self.head_dim) # 计算注意力权重矩阵
            attn_weights = attn_weights + self.mask[:, :, :seqlen, :seqlen] # 添加掩码
            attn_weights = F.softmax(attn_weights, dim=-1).type_as(xq) # 对注意力权重进行归一化处理
            attn_weights = self.attn_dropout(attn_weights) # 应用注意力概率矩阵的随机失活层
            output = torch.matmul(attn_weights, xv) # 计算加权值向量

        '''
        |        场景        |      训练模式     |      推理模式     |
        |--------------------|------------------|------------------|
        | 长序列(seqlen>1)   |使用Flash Attention|使用Flash Attention|
        | 短序列(seqlen=1)   |     不可能出现    |    使用标准实现    |
        | 不支持Flash的环境   |    使用标准实现   |    使用标准实现    |
        '''

        output = output.transpose(1, 2).contiguous().view(bch_size, seqlen, -1  ) # 将输出展平为[batch_size, seq_len, dim]
        output = self.wo(output) # 将输出通过输出权重矩阵层
        output = self.resid_dropout(output) # 应用残差连接的随机失活层
        return output
            

        

                
