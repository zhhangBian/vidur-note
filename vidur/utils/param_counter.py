from math import ceil

from vidur.config import ReplicaConfig


# 对于replica的参数计算
# 依靠replica的config进行计算
class ParamCounter:
    # 依托Replia进行初始化，与每个replia紧密相关，计算每个replia的相关参数
    def __init__(self, replica_config: ReplicaConfig) -> None:
        self._replica_config = replica_config
        self._model_config = self._replica_config.model_config

        # 进行一系列的断言检查
        # 确保模型的配置（如头数、层数、嵌入维度）能够被并行配置整除，这是为了确保模型可以均匀地分布在多个设备上
        assert (
            self._model_config.num_q_heads % self._replica_config.tensor_parallel_size
            == 0
        )
        assert (
            self._model_config.num_layers % self._replica_config.num_pipeline_stages
            == 0
        )
        assert (
            self._model_config.embedding_dim % self._replica_config.tensor_parallel_size
            == 0
        )
        assert self._model_config.embedding_dim % self._model_config.num_q_heads == 0

        # 计算每层的参数数量
        # 每个流水线分配多少层
        self._num_layers_per_pipeline_stage = (
            self._model_config.num_layers // self._replica_config.num_pipeline_stages
        )
        # 每个注意力头的维度
        self._attention_head_dim = (
            self._model_config.embedding_dim // self._model_config.num_q_heads
        )
        # 每个张量并行工作单元（tensor parallel worker）应该有多少查询头
        self._q_heads_per_tensor_parallel_worker = (
            self._model_config.num_q_heads // self._replica_config.tensor_parallel_size
        )
        # 每个张量并行工作单元应该有多少键（key）和值（value）头
        # ceil为向上取整
        self._kv_heads_per_tensor_parallel_worker = ceil(
            self._model_config.num_kv_heads / self._replica_config.tensor_parallel_size
        )

    def get_num_parameters_per_layer(self) -> int:
        num_parameters = 0
        # weights for attention metrics Wq, Wk, Wv
        num_parameters += (
            # 对于每个头，Wq、Wk 和 Wv 的参数数量是相同的
            self._model_config.embedding_dim
            * self._attention_head_dim
            # 有 q_heads_per_tensor_parallel_worker 个查询头和两倍的键值头
            # 因为每个键值头被计算两次，一次为键，一次为值
            * (
                self._q_heads_per_tensor_parallel_worker
                + 2 * self._kv_heads_per_tensor_parallel_worker
            )
        )

        # weights for attention metrics Wo
        # 自注意力机制的输出权重
        num_parameters += (
            self._model_config.embedding_dim
            * self._attention_head_dim
            * self._q_heads_per_tensor_parallel_worker
        )
        
        
        # fc layer weights
        # MLP层通常包含一个或两个全连接（FC）层，它们的参数数量取决于输入和输出的维度
        # 门控MLP：门控机制引入了额外的参数
        if self._model_config.use_gated_mlp:
            num_parameters += (
                3
                * self._model_config.embedding_dim
                * self._model_config.mlp_hidden_dim
                # 参数需要在多个设备之间分配
                // self._replica_config.tensor_parallel_size
            )
        # 非门控MLP
        else:
            num_parameters += (
                2
                * self._model_config.embedding_dim
                * self._model_config.mlp_hidden_dim
                # 参数需要在多个设备之间分配
                // self._replica_config.tensor_parallel_size
            )

        return num_parameters

    # 得到每个设备的参数总量：层数*每层的平均参数量
    def get_num_parameters_per_device(self) -> int:
        num_parameters_per_layer = self.get_num_parameters_per_layer()
        return num_parameters_per_layer * self._num_layers_per_pipeline_stage
