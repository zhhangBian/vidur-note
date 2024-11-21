from typing import Tuple

from vidur.entities import Batch, BatchStage, ExecutionTime
from vidur.execution_time_predictor import BaseExecutionTimePredictor


# 在pipeline层级进行调度
# 目前仅支持一种调度算法：先来先服务
class ReplicaStageScheduler:
    def __init__(
        self,
        replica_id: int,
        stage_id: int,
        is_last_stage: bool,
        execution_time_predictor: BaseExecutionTimePredictor,
    ) -> None:
        self._replica_id = replica_id
        self._stage_id = stage_id
        self._is_last_stage = is_last_stage
        self._execution_time_predictor = execution_time_predictor

        self._batch_queue = []
        self._is_busy = False

    @property
    def is_last_stage(self) -> bool:
        return self._is_last_stage

    def is_empty(self) -> bool:
        return len(self._batch_queue) == 0

    # 添加一批request
    def add_batch(self, batch: Batch) -> None:
        self._batch_queue.append(batch)

    # 是否正处于忙碌状态
    def on_stage_end(self) -> None:
        self._is_busy = False

    # 进行调度
    # 默认调度算法为取出最先达到的batch
    def on_schedule(self) -> Tuple[Batch, BatchStage, ExecutionTime]:
        if self._is_busy or not self._batch_queue:
            return None, None, None

        self._is_busy = True
        # 取出一个batch
        batch = self._batch_queue.pop(0)
        # 得到完成batch的时间预测
        execution_time = self._execution_time_predictor.get_execution_time(
            batch,
            self._stage_id,
        )
        total_execution_time = execution_time.total_time
        model_execution_time = execution_time.model_time
        # 将结果进行封装
        batch_stage = BatchStage(
            batch.id,
            self._replica_id,
            self._stage_id,
            total_execution_time,
            model_execution_time,
            batch.requests,
            batch.num_tokens,
        )

        return batch, batch_stage, execution_time
