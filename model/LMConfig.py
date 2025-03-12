from transformers import PretrainedConfig
from typing import List

class LMConfig(PretrainedConfig):
    model_type = "miniDeepSeek"

    def __init__(
            self,  
            dim: int=512,                  # 门控层的维度，决定了门控层的输入维度==token特征维度
            n_layers: int=8,               # transformer的层数--transformer由若干个transformer block组成，这个参数决定了我们的transformer有多少个transformer block
            n_heads: int=16,               # 总注意力头数（查询头数量），决定query向量的并行计算能力
            n_kv_heads: int=8,             # 键/值头的数量，决定key/value向量的复用程度
            # 每个key/value头会被多个query头共享（本例中16/8=2个query头共享1个key/value头，这个比例和标准transformer相比降低了50%参数量）
            vocab_size: int=6400,          # 词汇表大小，会影响token序列长度
            hidden_dim: int=None,          # 隐藏层维度，决定了模型的表达能力
            multiple_of: int=64,           # 隐藏层维度的倍数，决定了模型的计算效率
            norm_eps: float=1e-5,          # 归一化层的稳定性参数(防止除以零的极小数)
            max_seq_length: int=512,       # 最大序列长度，决定了模型可以处理的最大token数量
            dropout: float=0.0,            # 随机失活层的概率，防止过拟合
            flash_attention: bool=True,    # 是否使用flash attention，提高计算效率
            use_moe: bool=True,            # 是否使用混合专家机制，提高模型表达能力
            num_experts_per_token=3,       # 每个token选择的专家数量
            n_routed_experts=8,            # 总的专家数量
            n_shared_experts: bool = True, # 是否共享专家
            scoring_func='softmax',        # 评分函数，默认为'softmax'
            aux_loss_alpha=0.01,           # 辅助损失的alpha参数
            seq_aux=True,                  # 是否在序列级别上计算辅助损失
            norm_topk_prob=True,           # 是否标准化num_experts_per_token概率
            **kwargs,                      
    ):
        self.dim = dim
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_token = num_experts_per_token  
        self.n_routed_experts = n_routed_experts  
        self.n_shared_experts = n_shared_experts  
        self.scoring_func = scoring_func  
        self.aux_loss_alpha = aux_loss_alpha  
        self.seq_aux = seq_aux  
        self.norm_topk_prob = norm_topk_prob  
        super().__init__(**kwargs)