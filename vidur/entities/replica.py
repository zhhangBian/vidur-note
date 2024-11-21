from math import ceil

from vidur.config import BaseRequestGeneratorConfig, ReplicaConfig
from vidur.entities.base_entity import BaseEntity
from vidur.logger import init_logger

logger = init_logger(__name__)


# 对分布式训练实体的抽象
class Replica(BaseEntity):
    def __init__(
        self,
        replica_config: ReplicaConfig,
        generator_config: BaseRequestGeneratorConfig,
    ) -> None:
        self._id = Replica.generate_id()

        self._replica_config = replica_config
        # 根据模型名得到相应model的参数
        # model都遵循了基本的模式，只在少数参数上有区别
        self._model_config = replica_config.model_config
        # 训练设备的参数
        self._device_config = replica_config.device_config
        # request生成器的参数
        self._generator_config = generator_config

        assert (
            self._model_config.num_layers % self._replica_config.num_pipeline_stages
            == 0
        )
        assert (
            self._model_config.embedding_dim % self._replica_config.tensor_parallel_size
            == 0
        )

    @property
    def id(self) -> int:
        return self._id

    @property
    def num_layers(self) -> int:
        return self._model_config.num_layers

    @property
    def num_q_heads(self) -> int:
        return self._model_config.num_q_heads

    @property
    def num_kv_heads(self) -> int:
        return self._model_config.num_kv_heads

    @property
    def embedding_dim(self) -> int:
        return self._model_config.embedding_dim

    @property
    def mlp_hidden_dim(self) -> int:
        return self._model_config.mlp_hidden_dim

    @property
    def use_gated_mlp(self) -> int:
        return self._model_config.use_gated_mlp

    @property
    def vocab_size(self) -> int:
        return self._model_config.vocab_size

    # 在pipeline中，模型被拆分为多少个阶段运行
    @property
    def num_pipeline_stages(self) -> int:
        return self._replica_config.num_pipeline_stages

    @property
    def num_layers_per_pipeline_stage(self) -> int:
        return self._model_config.num_layers // self._replica_config.num_pipeline_stages

    # 返回每个注意力头的维度
    @property
    def attention_head_dim(self) -> int:
        return self._model_config.embedding_dim // self._model_config.num_q_heads

    # 返回每个张量并行工作进程的查询头数量
    @property
    def q_heads_per_tensor_parallel_worker(self) -> int:
        return (
            self._model_config.num_q_heads // self._replica_config.tensor_parallel_size
        )

    # 返回每个张量并行工作进程的键值头数量
    @property
    def kv_heads_per_tensor_parallel_worker(self) -> int:
        return ceil(
            self._model_config.num_kv_heads / self._replica_config.tensor_parallel_size
        )

    # 返回张量并行工作进程的数量
    @property
    def num_tensor_parallel_workers(self) -> int:
        return self._replica_config.tensor_parallel_size

    # 返回设备的总内存（以 GB 为单位）
    @property
    def total_memory_gb(self) -> int:
        return self._device_config.total_memory_gb

    # 返回内存裕度比例
    @property
    def memory_margin_fraction(self) -> float:
        return self._replica_config.memory_margin_fraction

    # 返回request的最大token数
    @property
    def max_request_tokens(self) -> int:
        return self._generator_config.max_tokens

    # 返回每台设备的浮点运算能力（以 FLOPS 为单位）
    @property
    def per_device_flops(self) -> float:
        return self._device_config.fp16_tflops * 2**40

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "num_layers": self.num_layers,
            "num_q_heads": self.num_q_heads,
            "num_kv_heads": self.num_kv_heads,
            "embedding_dim": self.embedding_dim,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "use_gated_mlp": self.use_gated_mlp,
            "vocab_size": self.vocab_size,
            "num_pipeline_stages": self.num_pipeline_stages,
            "num_tensor_parallel_workers": self.num_tensor_parallel_workers,
        }
