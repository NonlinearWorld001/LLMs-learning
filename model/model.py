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
        self.head_dim = args.dim // args.n_heads # 每个头的维度，注意力模块的总维度由所有查询注意力头平分，默认512/16=32

        # 权重矩阵的本质作用是通过线性变换，将输入的特征维度映射到输出特征维度，从而实现特征的升维或降维
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
    * 神经网络结构示意：
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

        # 按注意力头拆分特征
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
            attn_weights = torch.matmul(xq, xk.transpose(2, 3))/ math.sqrt(self.head_dim) # 计算注意力权重矩阵，[batch_size, n_local_heads, seq_len, head_dim] * [batch_size, n_local_heads, head_dim, seq_len] 
            # 此时attn_weights的形状为[batch_size, n_local_heads, seq_len, seq_len]
            attn_weights = attn_weights + self.mask[:, :, :seqlen, :seqlen] # 添加掩码
            # 此时attn_weights的形状为[batch_size, n_local_heads, seq_len, seq_len]
            attn_weights = F.softmax(attn_weights, dim=-1).type_as(xq) # 对注意力权重进行归一化处理
            # 此时attn_weights的形状为[batch_size, n_local_heads, seq_len, seq_len]
            attn_weights = self.attn_dropout(attn_weights) # 应用注意力概率矩阵的随机失活层
            # 此时attn_weights的形状为[batch_size, n_local_heads, seq_len, seq_len]
            output = torch.matmul(attn_weights, xv) # 计算加权值向量
            # xv的形状为[batch_size, n_local_kv_heads, seq_len, head_dim]
            # output的形状为[batch_size, n_local_heads, seq_len, head_dim]

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
            
# 前馈神经网络层(类)：SwiGLU（Swish-Gated Linear Unit）前馈网络结构，实现门控机制
class FeedForward(nn.Module):
    def __init__(self, dim:int, hidden_dim:int=None, multiple_of:int=128, drop_out:float=0.0):
        '''
        dim: 输入特征的维度
        hidden_dim: 隐藏层的维度,通常是输入特征维度的4倍
        multiple_of: 隐藏层维度的倍数,通常是128
        drop_out: 随机失活层的概率
        '''
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim # 由 transformer论文：隐藏层维度通常是输入特征维度的4倍，方可以提供足够的表达能力
            hidden_dim = int(2 * hidden_dim / 3) # 2/3缩放：在使用SwiGLU等更高效的激活函数时，适当缩小隐藏层维度（约2.66x）依旧可以保持模型的性能，在参数上则可以减少约33%的参数计算量
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of) # 向上取整，并且是multiple_of(一般是128)的倍数，目的是实现GPU内存访问对齐和计算效率优化
                                                                                       # ⌈a / b⌉ = ⌊( a + b - 1 ) / b⌋  <--- 向上取整的公式

        self.w1 = nn.Linear(dim, hidden_dim, bias=False) # 升维层，处理输入，把输入特征升维到hidden_dim，提供给隐藏层处理，负责提取基础特征
        self.w2 = nn.Linear(hidden_dim, dim, bias=False) # 降维层，处理隐藏层的输出，把隐藏层的输出降维到dim，提供给残差连接
        self.w3 = nn.Linear(dim, hidden_dim, bias=False) # 门控层，与w1形成并行结构，负责控制信息流动，生成动态调节系数，提供非线性变换
                                                         # 门控信号根据输入 x 实时生成，使模型能自适应增强重要特征、抑制噪声（如某些语义或空间位置的特征），类似于注意力机制，可以对输入特征进行动态加权

        self.dropout = nn.Dropout(drop_out) # 随机失活层，在输出层之前应用，防止过拟合
    '''
        FeedForward 网络结构示意：
                      输入
                        ├─ w1 → Silu激活 ─┐
                        │                │—逐元素相乘 → w2 → Dropout → 输出
                        └─ w3 → 门控系数 ─┘
    '''
    def forward(self, x:torch.Tensor):
        return self.dropout( # 4. 随机失活层
            self.w2( # 3. 降维到原始维度
                F.silu(self.w1(x)) # 1. 升维+SwiGLU激活
                 * self.w3(x) # 2. 并行门控机制
            ) 
        ) 


# 混合专家门控机制层(类)
class MOEGate(nn.Module):
    def __init__(self, config:LMConfig):
        super().__init__()
        self.config = config # 配置参数
        self.top_k = config.num_experts_per_token # 每个token被路由的专家数量(即处理每个token需要的专家数量)
        self.n_routed_experts = config.n_routed_experts # 模型中参与路由的专家总数量，决定了有多少个专家会被用于处理数据(参与路由过程，参与路由专家越多，模型越复杂)

        self.scoring_func = config.scoring_func # 评分函数，默认为'softmax'
        self.alpha = config.aux_loss_alpha # 辅助损失的alpha参数，用于控制辅助损失的权重，辅助损失的权重越大，模型越倾向于学习专家的分布，从而提高模型的泛化能力
        self.seq_aux = config.seq_aux # 是否在序列级别上计算辅助损失，bool类型，默认为False

        self.norm_topk_prob = config.norm_topk_prob # 是否标准化top-k概率,即是否对每个token的专家选择概率进行归一化
        self.gate_dim = config.dim # 门控层的维度，决定了门控层的输入维度==token特征维度
        self.weight = nn.Parameter(torch.empty(self.n_routed_experts, self.gate_dim)) # 权重矩阵，用于存储每个专家的权重
        self.reset_parameters() # 重置参数

    def reset_parameters(self):
        import torch.nn.init as init # 仅在本函数中使用该包，避免污染全局命名空间，同时优化内存
        init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # 使用Kaiming初始化方法，均匀分布初始化权重矩阵

    def forward(self, hidden_states:torch.Tensor):
        # hidden_states: 输入到门控机制的特征表示，形状为[batch_size, seq_len, hidden_dim==token's feature_dim]
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous() # 确保输入张量是连续的，避免在计算过程中出现非连续内存访问

        batch_size, seq_len, hidden_dim = hidden_states.shape # 获取输入张量的batch_size和seq_len、hidden_dim
        
        hidden_states = hidden_states.view(-1, hidden_dim) # shape: [batch_size, seq_len, hidden_dim] --> [batch_size * seq_len, hidden_dim]
        logits = F.linear(hidden_states, self.weight) # 对hidden_states以self.weight为权重矩阵进行线性变换，计算每个专家的得分
        # logits的形状为[batch_size * seq_len, n_routed_experts]

        if self.scoring_func == 'softmax':
            scores = F.softmax(logits, dim=-1)
            # scores的形状为[batch_size * seq_len, n_routed_experts]，但是最后一维的值和为1，对于每一个专家的权重(score)已经被归一化/激活,转为概率分布
        else:
            raise NotImplementedError(f"scoring_func {self.scoring_func} not implemented in MOEGate")# only support softmax now
        
        # torch.topk函数：用于返回张量中最大或最小的k个元素，并返回它们的值和索引
        # 输入格式： torch.topk(input:输入的张量, k:返回的元素数量, dim:沿着哪个维度进行操作, larges:默认为True,表示返回最大值，若为False,则返回最小值, sorted:是否排序, out:可选参数，用于存储输出结果)
        topk_scores, topk_index = torch.topk(scores, k=self.top_k, dim=-1, sorted=True) # 返回每个token所应该使用的概率最高的k=self.top_k个专家的权重和索引
        # topk_scores的形状为[batch_size * seq_len, top_k], 数据类型为torch.float32
        # topk_index的形状为[batch_size * seq_len, top_k], 数据类型为torch.int64
        
        if self.top_k > 1 and self.norm_topk_prob: #如果决定每个token的输出专家数量大于1，并且需要对每个token的专家选择概率进行归一化，这样做可以更加精确地计算专家负载，保证梯度传播的稳定性，同时也能避免因概率稀释导致的训练不稳定
            denominator = topk_scores.sum(dim=-1, keepdim=True) + 1e-20 # 计算归一化所用的分母，+1e-20是为了防止分母为0
            topk_scores = topk_scores / denominator # 归一化

        # 如果处于训练模式，并且alpha大于0，则计算辅助损失：最终目的是防止"专家僵死"问题，提升所有专家的利用率，增强模型容量
        if self.training and self.alpha > 0: 
            scores_aux = scores # 辅助损失的所有专家的得分，形状为[batch_size * seq_len, n_routed_experts]
            aux_topk = self.top_k # 辅助损失的专家数量
            topk_index_aux_loss = topk_index.view(batch_size, -1) # 把辅助损失的专家索引按照batch分好，[batch_size * seq_len, top_k] --> [batch_size, seq_len * top_k]
            if self.seq_aux: # 如果需要计算序列级别的辅助损失，考虑序列长度的影响，需要对序列归一化，更适合处理存在不同序列长度的NLP任务
                scores_seq_aux = scores_aux.view(batch_size, seq_len, -1) # 再把辅助损失的得分按照batch和seq_len分好，[batch_size * seq_len, n_routed_experts] --> [batch_size, seq_len, n_routed_experts]
                ce = torch.zeros(batch_size, self.n_routed_experts, device=hidden_states.device)# 创建一个形状为[batch_size, n_routed_experts]的计数张量ce，用于存储每个专家的计数
                
                # 使用散射加法统计每个专家被选中的次数：
                # 统计每个batch中每个专家被选中的相对频率，将实际选中次数与期望次数（假设均匀分布）的比例作为监督信号，最终辅助损失会计算该比例与1的均方误差
                # 防止某些专家长期不被选择（"专家僵死"问题），避免热门专家被过度选择，提升所有专家的利用率，增强模型容量；通过超参数alpha控制负载均衡与任务损失的平衡
                ce.scatter_add_( #ce: 目标张量, [batch_size, n_routed_experts]
                    -1, # 操作维度，分散计算对应的索引则为ce[-1,topk_index_aux_loss的索引[1, x(x会遍历每一个索引)]对应的值]
                    topk_index_aux_loss,  # 专家索引 [batch_size, seq_len*top_k]
                    torch.ones(batch_size, seq_len*aux_topk, device=hidden_states.device) # 要累加的值（每个位置都是1）
                ).div_(seq_len*aux_topk/self.n_routed_experts) # 每个专家被选择的相对频率 = 除以[(序列长度*top_k)/ 专家总数] = [每个样本（batch）所有token的总选择次数]/专家综述 = 实际选择次数/期望选择次数 = 实际选择次数/(总选择机会/专家总数)
                # scatter_add_函数：在指定维度上，将源张量中指定索引位置的元素加到目标张量中，并返回目标张量
                # 输入格式(假设目标张量为aim): aim.scatter_add_(dim:沿着哪个维度进行操作, index:指定索引位置, src:源张量)
                # 在现在的ce中，元素大于1则表示该专家被选中的次数大于期望次数，过载；反之则小于期望次数，欠载(工作不到位)

                # 改进型aux_loss算法放的位置，放入改进型算法后，应把289行代码注释掉

                # ce是每一个batch中每个专家的相对出勤频率
                aux_loss = (ce * scores_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
                # scores_seq_aux.mean(dim=1): dim=1的维度是seq_len，这个计算得到的是每个序列所使用的专家的平均得分(每个token的专家选择概率)，反映了每个专家的平均能力大小
                # (ce * scores_seq_aux.mean(dim=1)): 每个专家的相对出勤频率 * 每个专家的平均得分
                # .sum(dim=1).mean(): 沿着dim=1的维度进行求和，然后求平均，得到每个样本（batch）所有token的平均得分
                # 辅助损失 = 每个样本（batch）所有token的平均得分 * 每个token的平均得分 * 超参数alpha
                '''
                以上辅助损失可能会出现所有专家都被冷落的情况, 即所有专家的相对出勤频率都为0, 此时辅助损失为0, 模型无法学习到专家的分布, 从而导致模型性能下降
                因此，可改进为：
                1.交叉熵计算损失(不一定好):
                               target_freq = torch.ones(n_routed_experts) / n_routed_experts  # 目标频率：均匀分布
                               aux_loss = F.cross_entropy(ce, target_freq) * alpha
                2.改进型aux_loss算法:
                               eps = 1e-8  # 避免数值不稳定
                               aux_loss = -torch.log(ce + eps).mean() * alpha
                '''

            else:# 如果不需要计算序列级别的辅助损失，则直接计算token级别的辅助损失
                mask_ce = F.one_hot(topk_index_aux_loss.view(-1), num_classes=self.n_routed_experts)
                # 将topk_index_aux_loss展平为1D张量（形状：[batch_size * seq_len * top_k]）
                # 生成one-hot编码矩阵（形状：[总token数*top_k, 专家数]），标记每个token选择的专家
                ce = mask_ce.float().mean(dim=0) # 计算每个专家的相对出勤频率
                pi = scores_aux.mean(dim=0) # 计算每个专家的平均得分
                fi = ce * self.n_routed_experts # 实际比例因子=每个专家的相对出勤频率 * 专家总数，表示实际选择次数与期望次数的比值（期望次数 = 总选择次数 / 专家总数） 
                aux_loss = (pi * fi).sum() * self.alpha 
                # 辅助损失 = sum(每个专家的平均得分 * 实际比例因子) * 超参数alpha
                # 鼓励得分高的专家（pi大）保持合理的使用频率（fi接近1）；惩罚某些专家长期被冷落或过度使用的现象
        else:
            aux_loss = 0

        return topk_index, topk_scores, aux_loss


# 混合专家前馈神经网络层(类)：1.创建n_routed_experts个专家；2.根据每个token的专家选择概率，选择对应的专家进行处理；3.如果n_shared_experts不为None，则创建一个共享专家
class MOEFeedForward(nn.Module):
    def __init__(self, config:LMConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(
                dim = config.dim, # 输入特征的维度
                hidden_dim = config.hidden_dim, # 隐藏层维度
                multiple_of = config.multiple_of, # 隐藏层维度的倍数
                drop_out = config.dropout, # 随机失活层的概率
            )
            for _ in range(config.n_routed_experts) # 生成n_routed_experts个(独立)专家
        ])

        self.gate = MOEGate(config) # 混合专家门控机制
        if config.n_shared_experts is not None:
            self.shared_experts = FeedForward(
                dim = config.dim, # 输入特征的维度
                hidden_dim = config.hidden_dim, # 隐藏层维度
                multiple_of = config.multiple_of, # 隐藏层维度的倍数
                drop_out=config.dropout, # 随机失活层的概率
            ) # 创建一个共享专家

    def forward(self, x:torch.Tensor):
        identity = x                       # 输入
        origin_shape = x.shape             # 输入的形状, [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, _ = x.shape   # 获取输入的batch_size和seq_len

        # 门控机制
        topk_index, topk_scores, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1]) # 展平, [batch_size, seq_len, hidden_dim] --> [batch_size * seq_len, hidden_dim]
        flat_topk_index = topk_index.view(-1) # 合并所有的batch和seq_len,[batch_size * seq_len, top_k] --> [batch_size * seq_len * top_k]

        if self.training:
            # 训练模式
            x = x.repeat_interleave(self.config.num_experts_per_token, dim=0)
            # 输入样本都被重复预设好的处理每个token需要的专家数量， 【batch_size * seq_len, hidden_dim] --> [batch_size * seq_len * top_k, hidden_dim]
            y = torch.empty_like(x, dtype=torch.float16) # 用于储存专家处理后的结果，所以和x的形状相同
            for i, expert in enumerate(self.experts):
                y[flat_topk_index == i] = expert(x[flat_topk_index == i])
                y = (y.view(*topk_scores.shape, -1) * topk_scores.unsqueeze(-1)).sum(dim=1)
                # topk_scores.shape = [batch_size * seq_len, top_k]
                # y.view(*topk_scores.shape, -1).shape = [batch_size * seq_len, top_k, hidden_dim]
                # topk_scores.unsqueeze(-1).shape = [batch_size * seq_len, top_k, 1]
                # 将每个token路由到的指定个数的专家输出，按其对应的门控权重进行加权融合，最终形成每个token的最终特征表示
        else:
            # 推理模式, 只选择最优专家
            y = self.moe_infer(x, flat_topk_index, topk_scores.view(-1, 1)).view(*origin_shape) 
            # 在推理模式下，调用 self.moe_infer 函数

            if self.config.n_shared_experts is not None: # 检查是否定义了共享专家
                y = y + self.shared_experts(identity) # 如果有共享专家，将共享专家处理原始输入 identity 的结果加到 y 上

            return y

        @torch.no_grad() # 装饰器，用于告诉PyTorch，这个函数在推理过程中不需要计算梯度 <-- 仅在推理过程中使用
        def moe_infer(self, x, flat_expert_indices, flat_expert_scores):
            '''
            x: 输入张量，形状为 [batch_size * seq_len, hidden_dim], 通常batch_size在这里为1
            flat_expert_indices: 展平的专家索引，形状为 [batch_size * seq_len * top_k]
            flat_expert_scores: 展平的专家得分，形状为 [batch_size * seq_len * top_k]
            flat_expert_scores 和 flat_expert_indices 一一对应
            '''
            expert_cache = torch.zeros_like(x) # 初始化专家缓存：创建一个与x形状相同的全0张量，用于存储专家处理后的结果
            index = flat_expert_indices.argsort() # 对flat_expert_indices排序，获取专家索引的排序索引，这样可以将分配给同一个专家的 token 聚集在一起：
                                                  #每num_experts_per_tok个元素表示第len(index)//num_experts_per_tok个专家处理的num_experts_per_tok个token在flat_expert_indices中的位置
            tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0) # 统计每个专家被分配的token数量，并且分别计算累计和
            # .bincount(): 统计每个专家被分配的token数量
            # .cpu().numpy(): 将结果转换为numpy数组 <-- numpy只能处理cpu上的数据
            # .cumsum(0): 计算每个专家被分配的token数量的累积和 ## 累积和可以用于快速定位每个专家处理的token范围，比如说在分布式计算中分配任务
            # 例如：tokens_per_expert = [2, 4, 6, 8, 10]，表示第0个专家处理0~2个token，第1个专家处理2~6个token，第2个专家处理6~12个token，第3个专家处理12~20个token，第4个专家处理20~30个token
            token_index = index // self.config.num_experts_per_token #将排序后的专家索引位置映射回原始token位置
            '''
            flat_expert_indices = [3,1, 0,2, 2,3, 1,0, 3,2],batch_size=1,seq_len=5,top_k=2
            假设num_experts_per_token=2,排序后的专家索引为index=[2,7, 1,6, 3,4,9, 0,5,8]
            则token_index = index // 2 → [1,3, 0,3, 1,2,4, 0,2,4]
            表示：
            - 0号专家处理token1和token3
            - 1号专家处理token0和token3
            - 2号专家处理token1、token2和token4
            - 3号专家处理token0、token2和token4
            '''
            for i, end_index in enumerate(tokens_per_expert):
                # i: 当前专家的索引 ; end_index: 当前专家处理的token的尾索引
                start_index = 0 if i ==0 else tokens_per_expert[i-1] # 确定当前专家处理的token(索引)范围，tokens_per_expert[i-1]：前一个专家处理的token的尾索引
                if start_index == end_index:
                    continue # 如果当前专家没有处理任何token，则跳过
                expert = self.experts[i] # 获取当前专家
                exp_token_index = token_index[start_index:end_index] # 获取当前专家处理的token索引，返回的是一个"列表"
                expert_tokens = x[exp_token_index] # 获取当前专家处理的tokens
                expert_out = expert(expert_tokens) # 安排专家来处理token, 获得每个专家处理后的生成结果
                expert_out.mul_(flat_expert_scores[index[start_index:end_index]]) # 将专家处理后的结果乘以对应的专家得分，即加权
                expert_cache.scatter_add_(0, exp_token_index.view(-1,1).repeat(1,x.shape[-1]),expert_out)
                # 将专家处理后的结果存储到专家缓存中
            return expert_cache

# 一个Transformer块(类)
class TransformerBlock(nn.Module):
    def __init__(self, layer_id:int, config:LMConfig):
        super().__init__()
        self.n_heads = config.n_heads # 注意力头数==查询头数
        self.dim = config.dim # 输入特征的维度
        self.head_dim = config.dim // config.n_heads # 每个注意力头的维度
        self.attention = Attention(config) # 创建一个注意力机制

        self.layer_ID =  layer_id # 当前层在transformer中的ID
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps) # 注意力归一化层，并设定稳定性参数
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps) # 前馈神经网络归一化层，并设定稳定性参数
        
        if config.use_moe:
            self.feed_forward = MOEFeedForward(config) # 若使用混合专家前馈神经网络层，则实例化之前定义的混合专家前馈神经网络层
        else:
            self.feed_forward = FeedForward(
                dim=config.dim,
                hidden_dim=config.hidden_dim,
                multiple_of=config.multiple_of,
                drop_out=config.dropout,
            ) # 若不使用混合专家前馈神经网络层，则实例化之前定义的FeedForward类

    # 前向传播函数
    def forward(self, x:torch.Tensor, pos_cis:torch.Tensor, kv_cache=False):
        '''
        x: 输入张量，形状为 [batch_size, seq_len, hidden_dim]
        pos_cis: 位置编码，形状为 [batch_size, seq_len, hidden_dim]
        kv_cache: 是否使用缓存
        '''     
        h = x + self.attention(self.attention_norm(x), pos_cis, kv_cache) # 注意力机制，嵌入位置编码进行注意力计算，并加上x进行残差连接以缓解梯度消失问题
        h = h + self.feed_forward(self.ffn_norm(h)) # 前馈神经网络，对h进行归一化后进行门控机制的实现，并同样加上h进行残差连接以缓解梯度消失问题
        return h
    

# 完整的Transformer模型(类)，把之前定义的所有类组合起来，是实现整个LLM的框架以及框架的填充
class Transformer(PreTrainedModel):
    config_class = LMConfig # 根据LMConfig类配置transformer类
    last_loss: Optional[torch.Tensor] # 储存最后一次损失，Optional[torch.Tensor]表示这个属性可能没有值（即为 None），例如在模型初始化时或尚未计算损失时

    def __init__(self, params:LMConfig | None = None):
        self.params = params or LMConfig() # 如果params为None，则使用LMConfig的默认参数
        super().__init__(self.params)
        self.vocab_size = self.params.vocab_size # 词汇表大小
        self.n_layers = self.params.n_layers # transformer的transformer block层(个)数
        
        self.token_embedding = nn.Embedding(self.vocab_size, self.params.dim) # 创建一个词汇表大小的嵌入层，用于将token转换为特征向量
        self.layers = nn.ModuleList() # 创建一个模块列表，用于存储所有TransformerBlock
        self.dropout = nn.Dropout(self.params.dropout) # 创建随机失活层，用于防止过拟合
        '''在self.layers中添加指定个数的TransformerBlock，并由pytorch自动管理，这就是整个基于transformer的LLM的网络结构'''
        for trans_block_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id=trans_block_id, config=self.params))
        self.norm = RMSNorm(self.params.dim, eps=self.params.norm_eps) # 创建一个归一化层，用于将特征向量归一化
        self.output = nn.Linear(self.params.dim, self.vocab_size, bias=False) # 创建一个线性层，用于将特征向量转换为词汇表大小的概率分布,输入特征维度=预定的token维度，输出特征维度=词汇表大小,输出形状为[batch_size, seq_len, vocab_size]
        self.token_embedding.weight = self.output.weight # 将token_embedding的权重与output的权重共享，在减少参数量的同时，保持token_embedding和output的语义空间一致
        pos_cis = precompute_pos_cis(self.params.dim // self.params.n_heads, self.params.max_seq_length) # 预计算位置编码
        self.register_buffer('pos_cis', pos_cis, persistent=False) # 将预计算的位置编码注册为模型的缓存(不可学习的参数)，缓存不会参与梯度更新

        self.apply(self._init_weights) # 使用_init_weights函数初始化模型参数; apply()函数会遍历模型的所有子模块，并调用_init_weights函数初始化每个子模块的参数
        '''(torch.nn.apply(fn)是深度优先搜索)'''
        for pn, p in self.named_parameters(): # 遍历模型中的所有参数, named_parameters()返回一个生成器，生成器中的每个元素是一个包含参数名称和参数值的元组
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'): # 如果参数名称以'w3.weight'或'wo.weight'结尾，w3是门控机制FFN(x)=(Swish(xW1)⊙xW2)W3中的W3，wo是输出层的权重矩阵
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.params.n_layers)) 
                # 使用正态分布初始化参数，均值为0，标准差为0.02除以sqrt(2 * n_layers), 通过缩小深层网络特定层权重的初始化标准差，平衡网络深度带来的梯度衰减/爆炸问题
                # 在训练深层Transformer类模型时，这种初始化策略能提升收敛速度和稳定性
                # 式子中的0.02和2都是经验值，也是需要根据实际情况进行调整的超参数
        self.loss = None # 初始化损失为None
        self.OUT = CausalLMOutputWithPast() # 初始化输出为CausalLMOutputWithPast
        '''
        CausalLMOutputWithPast: 一种因果语言模型输出，包含预测的下一个token概率分布和损失
        '''
        self.np_split_modules = [name for name, _ in self.named_modules()]
        # 获取所有模块的名称，并存储在self.np_split_modules列表中
        # named_modules()返回一个生成器，生成器中的每个元素是一个包含模块名称和模块本身的元组, 会定位到之前已经被注册为子模块的self.layers上，利用里面存储/标记的transformer block的名称生成可迭代对象
        # 例如，如果一个模块的名称是"module.layer.0.feed_forward.w3.weight"，那么它将被存储在self.np_split_modules列表中
        # 这个列表用于存储所有需要进行参数分割的模块名称
        
    #定义权重矩阵初始化函数
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) # 使用正态分布初始化权重矩阵module.weight，均值为0，标准差为0.02
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) # 如果模块有偏置，则将其初始化为0
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) # 和Linear层一样使用正态分布初始化权重矩阵module.weight，均值为0，标准差为0.02，但是词嵌入层不包括偏置，所以要分开初始化

    # 基于transformer的LLM的前向传播函数
    def forward(self, tokens:Optional[torch.Tensor] = None, targets:Optional[torch.Tensor] = None, kv_cache=False, **keyargs):
        '''
            tokens: 输入的token序列，形状为[batch_size, seq_len], 类型只能为torch.Tensor或者None
            targets: 目标token序列，形状为[batch_size, seq_len], 类型只能为torch.Tensor或者None
            kv_cache: 是否使用缓存
            **keyargs: 其他参数，应当是一个字典
        '''
        current_idx = 0 # 当前token的索引
        if 'input_ids' in keyargs:
            tokens = keyargs['input_ids'] # 允许用户通过input_ids来指定输入的token序列，这表明既允许传入tokens，也允许传入input_ids作为处理对象tokens
        if 'attention_mask' in keyargs:
            targets = keyargs['attention_mask'] # 允许用户通过attention_mask来指定哪些位置是填充的，允许指定其他来源的attention_mask
        if 'current_idx' in keyargs:
            current_idx = int(keyargs['current_idx']) #允许外部传入 current_idx 以控制处理位置：在生成任务中，逐步生成 token 时，通过 current_idx 指定当前生成的位置（避免重复处理已生成部分）
                                                                                        # 例如：第一次调用时 current_idx=0，生成一个 token 后，下一次调用 current_idx=1，依此类推
        
        _bsz, seqlen = tokens.shape # 获取输入的batch_size和seq_len

        h = self.token_embedding(tokens) # 将tokens转换为特征向量
        h = self.dropout(h) # 使用dropout层进行随机失活,实现正则化嵌入，防止过拟合低频词或特定样本，可以增加模型的鲁棒性和泛化能力，注意仅在训练阶段开启，推理时使用完整的词嵌入；在训练不稳定的时候可以调整失活率
        pos_cis = self.pos_cis[current_idx:current_idx+seqlen] # 获取从当前位置到当前位置+seq_len的位置编码（在473、474行预计算位置编码时，已经将位置编码缓存到self.pos_cis中，作为模型参数的一部分，可以随时被访问）
        for idx, layer in enumerate(self.layers):
            h = layer(h, pos_cis, kv_cache) # 将h和pos_cis传入transformer blocks中，在每一个transformer block中依次进行前向传播
        
        h = self.norm(h) # 对h进行归一化

        if targets is not None:
            logits = self.output(h) # 在targets不为None时（即存在掩码时），计算logits（一般是最后一层的原始输出，通常会被传递给激活函数计算概率分布，也会被传递给损失函数计算损失）
            self.last_loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1),
                                                               ignore_index=0, reduction='none') 
            # 计算交叉熵损失
            # logits.view(-1, logits.shape[-1])：将logits展平为[batch_size * seq_len, vocab_size]
            # targets.view(-1)：将targets展平为[batch_size * seq_len]
            # ignore_index=0表示忽略索引为0的token（通常是填充token：填充token是指在序列中添加的用于保持序列长度一致的token，通常用0表示），reduction='none'表示不进行平均，返回每个样本的损失
        else:
            logits = self.output(h[:, [-1], :]) # 在targets为None时（即不存在掩码时），计算logits, [:, [-1], :]表示取最后一个token的特征向量，仅用最后一个token的特征向量来计算logits（下一个token的预测）
            self.last_loss = None # 在targets为None时，不计算损失
        
        self.OUT.__setitem__('logits', logits) # 将logits存储到OUT中
        self.OUT.__setitem__('loss', self.last_loss) # 将loss存储到OUT中

        return self.OUT(logits=logits, loss=self.last_loss)
    
    @torch.inference_mode() # 装饰器，用于告诉PyTorch，这个函数在推理生成过程中不需要计算梯度
    def generate(self, idx, eos, max_new_tokens, temperature=0.7, top_k=8, stream=True, rp=1.1, kv_cache=True):
        '''
        idx: 初始输入的token序列，形状为[batch_size=1, seq_len]
        eos: 结束token的索引
        max_new_tokens: 最大生成token数
        temperature: 温度参数，用于控制生成结果的多样性
        top_k: 用于控制生成结果的多样性
        stream: 是否流式生成
        rp: 重复惩罚因子
        kv_cache: 是否使用缓存
        '''
        index = idx.shape[1] # 初始化位置索引，即当前输入的token序列的长度
        init_inference = True # 初始化推理标志,用于判断是否是第一次推理
        while idx.shape[1] < max_new_tokens-1: # 如果当前输入的token序列的长度小于最大生成token数，则继续生成
            if init_inference or not kv_cache:
                inference_res, init_inference = self(idx, kv_cache=kv_cache), False 
                # 如果当前是第一次推理或者禁用缓存的时候，则进行推理时处理整个序列
            else:
                inference_res = self(idx[:, -1:], kv_cache=kv_cache, current_idx=idx.shape[1]-1)
                # 后续推理or启用缓存的时候，则进行推理时处理最后一个token
            # inference_res: 推理结果，包含logits和loss; logits: 形状为[batch_size, vocab_size]
            logits = inference_res.logits[:, -1, :] 
            # 获取最后一个token的logits, 当前形状为[batch_size, vocab_size]，第二维的每个元素是没有经过softmax函数计算得到的原始输出

            for token in set(idx.tolist()[0]):# 在推理时只会输入一个序列，等同于只有一个batch，所以只用idx.tolist()[0]对第一个bacth进行处理即可
                logits[:, token] /= rp 
                # 应用重复惩罚因子，用于惩罚重复的token，避免模型生成重复的token
                # 原理是：如果一个token在之前的序列中出现过，那么在用于生成当前token的logits中，该token的logits值除以一个大于1的数，
                # 这样整个logits在后续利用softmax计算概率分布时，该token的概率会降低，同时提高其他token的概率，从而避免模型生成重复的token
            
            if temperature == 0.0:
                _, idx_next = torch.topk(logits, top_k=1, dim=-1) # 如果温度为0，则直接使用topk函数获取概率最大的token，即不考虑输出的多样性
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)), sorted=True) # v: 形状为[batch_size, k=min(top_k, logits.size(-1))]
                    logits[logits < v[:,[-1]]] = -float('inf') 
                    # v[:,[-1]]：形状为[batch_size, 1]，表示v的最后一个元素, 也是v中第看大（最小的）的元素；logits < v[:,[-1]]返回的是bool张量
                    # logits[logits < v[:,[-1]]]：logits中所有小于v中第k大元素的元素都被设置为-inf，即将logits < v[:,[-1]]中为True的元素设置为-inf

                probs = F.softmax(logits, dim=-1) # 使用softmax函数计算概率分布
                id_next = torch.multinomial(probs, num_samples=1, generator=None) # 使用多重采样函数获取下一个token

            if idx_next == eos:
                break # 如果下一个token是结束token，则停止生成

            idx = torch.cat((idx, id_next), dim=-1) # 将当前token和下一个token拼接起来

            if stream:
                yield idx[:, index:] # 如果stream为True，则返回从初始位置到当前的所有新的token
        
        if not stream:
            yield idx[:, index:] 

    @torch.inference_mode() # 装饰器，用于告诉PyTorch，这个函数在推理生成过程中不需要计算梯度
    def eval_func(self, idx):
        '''
        进行单步预测，并且返回原始的logits，用于模型评估/计算其他评估指标
        '''
        idx_cond = idx if idx.shape[1] <= self.params.max_seq_length else idx[:, -self.params.max_seq_length:]
        # 如果idx的长度大于max_seq_length，则只取最后max_seq_length个token, 否则直接返回idx
        inference_res = self(idx_cond)
        logits = inference_res.logits[:, -1, :]
        return logits



