from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from vidur.config import SimulationConfig
from vidur.entities import Replica, Request
from vidur.execution_time_predictor import ExecutionTimePredictorRegistry
from vidur.scheduler.replica_scheduler.replica_scheduler_registry import (
    ReplicaSchedulerRegistry,
)


# 全局调度的基类，由不同的全局调度算法进行实现
# 由lor, random, round_robin三种调度算法
# 全局调度的含义：外部给一个request，由哪一个replica去响应
class BaseGlobalScheduler(ABC):
    def __init__(self, config: SimulationConfig, replicas: Dict[int, Replica]):
        self._config = config
        # 在全局中总共有多少个replica
        self._replicas = replicas
        self._num_replicas = len(self._replicas)

        # 执行时间预测
        execution_time_predictor = ExecutionTimePredictorRegistry.get(
            config.execution_time_predictor_config.get_type(),
            predictor_config=config.execution_time_predictor_config,
            replica_config=config.cluster_config.replica_config,
            replica_scheduler_config=config.cluster_config.replica_scheduler_config,
            metrics_config=config.metrics_config,
        )
        
        # 对每个replica给予一个调度器，遵循着相同的参数
        # 更低一层的调度器：replica层级上的内存和批调度策略
        self._replica_schedulers = {
            replica_id: ReplicaSchedulerRegistry.get(
                config.cluster_config.replica_scheduler_config.get_type(),
                replica_config=config.cluster_config.replica_config,
                replica_scheduler_config=config.cluster_config.replica_scheduler_config,
                request_generator_config=config.request_generator_config,
                replica=replica,
                num_stages=replica.num_pipeline_stages,
                execution_time_predictor=execution_time_predictor,
            )
            for replica_id, replica in replicas.items()
        }
        # 当前待响应的request
        self._request_queue = []

    # 按照到达时间对request进行排序
    def sort_requests(self) -> None:
        self._request_queue.sort(key=lambda request: request._arrived_at)

    # 添加request
    def add_request(self, request: Request) -> None:
        self._request_queue.append(request)

    def get_replica_scheduler(self, replica_id: int):
        return self._replica_schedulers[replica_id]

    # 返回replica的再低一层的stage-scheduler
    def get_replica_stage_scheduler(self, replica_id: int, stage_id: int):
        return self._replica_schedulers[replica_id].get_replica_stage_scheduler(
            stage_id
        )

    def is_empty(self) -> bool:
        return len(self._request_queue) == 0 and all(
            replica_scheduler.is_empty()
            for replica_scheduler in self._replica_schedulers.values()
        )

    # 进行调度，抽象方法，由子类实现
    @abstractmethod
    def schedule(self) -> List[Tuple[int, Request]]:
        pass
