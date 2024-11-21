from abc import ABC, abstractmethod
from typing import List

from vidur.config import (
    BaseReplicaSchedulerConfig,
    BaseRequestGeneratorConfig,
    ReplicaConfig,
)
from vidur.entities import Batch, Replica, Request
from vidur.execution_time_predictor import BaseExecutionTimePredictor
from vidur.logger import init_logger
from vidur.scheduler.replica_stage_scheduler import ReplicaStageScheduler
from vidur.scheduler.utils.memory_planner import MemoryPlanner

logger = init_logger(__name__)


# 在replica层级进行调度
# 按照论文：
# 1. 计算得到可用于KV-cache的内存
# 2. 在计算得到内存信息后，通过batching策略的API接口进行调度，如vllm
class BaseReplicaScheduler(ABC):
    def __init__(
        self,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        request_generator_config: BaseRequestGeneratorConfig,
        replica: Replica,
        num_stages: int,
        execution_time_predictor: BaseExecutionTimePredictor,
    ) -> None:
        self._config = replica_scheduler_config
        self._replica_config = replica_config
        self._request_generator_config = request_generator_config
        self._replica_id = replica.id
        # pipeline总共有多少个stage
        self._num_stages = num_stages

        # 每个序列可分的block数
        # sequence指在给定的内存和计算资源限制下，一个请求（sequence）可以被分割成的最大 block 数量
        self._max_blocks_per_sequence = (
            self._request_generator_config.max_tokens // self._config.block_size
        )

        # 是replica的内存计算抽象，计算可用于KV-Cache的内存数
        memory_planner = MemoryPlanner(self._replica_config, replica)

        # 每个sequence可视作一个request
        # 若没有定义block数，则反向推导得到当前的block数
        if not self._config.num_blocks:
            self._config.num_blocks = (
                self._max_blocks_per_sequence * memory_planner.get_max_request_slots()
            )
        # 当前可处理的批大小
        self._max_batch_size = min(
            memory_planner.get_max_batch_size(),
            self._config.batch_size_cap,
        )

        logger.debug(
            f"Obtained max batch size of {self._max_batch_size} for replica {self._replica_id}"
        )

        self._request_queue = []
        self._num_allocated_blocks = 0
        self._allocation_map = {}

        # 每个stage分配一个stage调度器
        self._replica_stage_schedulers = {
            stage_id: ReplicaStageScheduler(
                replica.id,
                stage_id,
                stage_id == num_stages - 1,
                execution_time_predictor,
            )
            for stage_id in range(num_stages)
        }

    # 当前未响应的request
    @property
    def num_pending_requests(self) -> int:
        return len(self._request_queue)

    @property
    def replica_id(self) -> int:
        return self._replica_id

    @property
    def num_allocated_blocks(self) -> int:
        return self._num_allocated_blocks

    @property
    def memory_usage_percent(self) -> int:
        return (self._num_allocated_blocks * 100) / self._config.num_blocks

    def is_empty(self) -> bool:
        return (
            self.num_pending_requests == 0
            and len(self._allocation_map) == 0
            and all(
                stage_scheduler.is_empty()
                for stage_scheduler in self._replica_stage_schedulers.values()
            )
        )

    def _get_request_next_num_tokens(self, request: Request) -> int:
        assert not request.completed

        if request.is_prefill_complete:
            return 1

        return request.num_prefill_tokens

    def add_request(self, request: Request) -> None:
        self._request_queue.append(request)

    def get_replica_stage_scheduler(self, stage_id: int):
        return self._replica_stage_schedulers[stage_id]

    # 可以进一步获得内存
    def can_allocate(self, num_blocks: int) -> bool:
        return self._config.num_blocks - self._num_allocated_blocks >= num_blocks

    # 获得一块内存以响应request
    def allocate(self, request_id: int, num_blocks: int) -> None:
        self._num_allocated_blocks += num_blocks
        if request_id not in self._allocation_map:
            self._allocation_map[request_id] = num_blocks
        else:
            self._allocation_map[request_id] += num_blocks

        assert self._num_allocated_blocks <= self._config.num_blocks

    # 释放一系列request的内存占用
    def free(self, *request_ids: List[int]) -> None:
        for request_id in request_ids:
            num_blocks = self._allocation_map.pop(request_id)
            self._num_allocated_blocks -= num_blocks

        assert self._num_allocated_blocks >= 0

    # 释放一个batch的内存占用
    def free_batch(self, batch: Batch) -> None:
        self.free(*batch.request_ids)

    @abstractmethod
    def on_batch_end(self, batch: Batch) -> None:
        pass

    @abstractmethod
    def _get_next_batch(self) -> Batch:
        pass

    # 对一些列batch进行调度
    def on_schedule(self) -> List[Batch]:
        scheduled_batches = []
        while self._num_running_batches < self._num_stages:
            batch = self._get_next_batch()
            if not batch:
                break
            scheduled_batches.append(batch)
            self._num_running_batches += 1
        return scheduled_batches
