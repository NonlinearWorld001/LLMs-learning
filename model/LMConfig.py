from transformers import PretrainedConfig
from typing import List

class LMConfig(PretrainedConfig):
    model_type = "miniDeepSeek"

    def __init__(
            self,
            dim: int=512,
            n_layers: int=8,
            n_heads: int=16,   # 总注意力头数（查询头数量），决定query向量的并行计算能力
            n_kv_heads: int=8, # 键/值头的数量，决定key/value向量的复用程度
            # 每个key/value头会被多个query头共享（本例中16/8=2个query头共享1个key/value头，这个比例和标准transformer相比降低了50%参数量）
            vocab_size: int=6400,
            hidden_dim: int=None,
            multiple_of: int=64,
            norm_eps: float=1e-5,
            max_seq_length: int=512,
            dropout: float=0.0,
            flash_attention: bool=True,
            use_moe: bool=True,
            num_experts_per_token=3,
            n_routed_experts=8,
            n_shared_experts: bool = True,
            scoring_func='softmax',
            aux_loss_alpha=0.01,
            seq_aux=True,
            norm_topk_prob=True,
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
        self.num_experts_per_token = num_experts_per_token  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率
        super().__init__(**kwargs)