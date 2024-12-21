from vidur.config import ReplicaConfig
from vidur.entities import BatchStage
from vidur.utils.param_counter import ParamCounter


# 计算在特定批处理阶段（BatchStage）下的模型浮点运算（FLOPs）利用率MFU
# MFU：Model Floating-point Utilization
class MFUCalculator:
    def __init__(self, replica_config: ReplicaConfig):
        param_counter = ParamCounter(replica_config)
        # 得到每个设备上的参数数量
        self._num_params_per_device = param_counter.get_num_parameters_per_device()
        # 利用传入的config得到每个训练实例的配置信息
        model_config = replica_config.model_config

        # 计算得到每个设备的层数
        self._num_layers_per_device = (
            model_config.num_layers // replica_config.num_pipeline_stages
        )
        # 每个设备的头数
        self._num_heads_per_device = (
            model_config.num_q_heads // replica_config.tensor_parallel_size
        )
        # 每个头的维度数
        self._head_dimension = model_config.embedding_dim // model_config.num_q_heads

        # 设备的计算能力
        self._device_flops = replica_config.device_config.fp16_tflops * 2**40

    # 计算多层感知机（MLP）部分的所需要的浮点运算量
    def _get_mlp_flops(self, batch_stage: BatchStage) -> float:
        # 根据批处理阶段中的token数量来估算所需的浮点运算
        num_tokens = sum(batch_stage.num_tokens)
        return 2 * num_tokens * self._num_params_per_device

    # 计算注意力机制部分的浮点运算量
    # 遍历批处理阶段中的每个请求，计算每个请求的注意力操作所需的浮点运算
    def _get_attention_flops(self, batch_stage: BatchStage) -> float:
        total_flops = 0
        # 遍历所有request进行计算
        for request, num_tokens in zip(batch_stage.requests, batch_stage.num_tokens):
            total_flops += (
                4  # for number of ops in attention
                * self._num_layers_per_device
                * self._num_heads_per_device
                * self._head_dimension
                * num_tokens  # q length
                * (num_tokens + request.num_processed_tokens)  # kv length
            )

        return total_flops

    # 计算推理过程中的性能整体利用率
    def get_mfu(self, batch_stage: BatchStage) -> float:
        # 估算房前
        mlp_flops = self._get_mlp_flops(batch_stage)
        attention_flops = self._get_attention_flops(batch_stage)
        
        total_flops = mlp_flops + attention_flops
        total_flops_per_second = total_flops / batch_stage.execution_time

        return total_flops_per_second * 100 / self._device_flops
