from vidur.config import ReplicaConfig
from vidur.entities.replica import Replica
from vidur.utils.param_counter import ParamCounter


# 对replica的内存进行计算
class MemoryPlanner:
    def __init__(self, replica_config: ReplicaConfig, replica: Replica) -> None:
        self._param_counter = ParamCounter(replica_config)
        self._replica = replica

    # 计算每个request所需要的 每个layer 所需的 KV-Cache 的内存大小
    def _get_kv_cache_memory_per_layer_per_request(self) -> int:
        return (
            2  # 2 bytes per float
            * 2  # one for key, one for value
            * self._replica.attention_head_dim
            * self._replica.kv_heads_per_tensor_parallel_worker
            * self._replica.max_request_tokens
        )

    # 每个设备上模型参数所需的内存大小
    def _get_parameter_memory_per_device(self) -> int:
        # 2 bytes per float
        return 2 * self._param_counter.get_num_parameters_per_device()

    # 每个设备上每个请求所需的KV-Cache内存大小
    # 基于每层的KV-Cache内存需求和模型的层数
    def _get_kv_cache_memory_per_device_per_request(self) -> int:
        return (
            self._get_kv_cache_memory_per_layer_per_request() * self._replica.num_layers
        )

    # 计算在可用内存限制下，可以处理的最大批次大小
    # 是在不考虑流水线下的request数目
    def get_max_batch_size(self) -> int:
        # 计算可用的内存
        available_memory = (
            self._replica.total_memory_gb
            * 1024**3
            # 内存的裕度比例
            * (1 - self._replica.memory_margin_fraction)
        )
        # 参数所需的内存
        parameter_memory_per_device = self._get_parameter_memory_per_device()
        # 每个request用于KV-Cache的平均内存
        kv_cache_memory_per_device_per_request = (
            self._get_kv_cache_memory_per_device_per_request()
        )

        # 可用内存大小
        memory_for_kv_cache = available_memory - parameter_memory_per_device
        # 得到可响应的request数
        number_of_requests = (
            memory_for_kv_cache // kv_cache_memory_per_device_per_request
        )

        assert (
            number_of_requests > 0
        ), "Not enough memory to store even a single request"

        return number_of_requests

    # 可响应的request数目：考虑流水线的并行性
    # 是最大批次大小和 流水线的阶段数
    def get_max_request_slots(self) -> int:
        return self.get_max_batch_size() * self._replica.num_pipeline_stages
