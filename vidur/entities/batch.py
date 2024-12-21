from typing import List

from vidur.entities.base_entity import BaseEntity
from vidur.entities.request import Request
from vidur.logger import init_logger

logger = init_logger(__name__)


# a decorator which checks if the request has been scheduled
def check_scheduled(func):
    def wrapper(self, *args, **kwargs):
        if not self._scheduled:
            raise ValueError("Batch has not been scheduled yet")
        return func(self, *args, **kwargs)

    return wrapper


def check_completed(func):
    def wrapper(self, *args, **kwargs):
        if not self._completed:
            raise ValueError("Batch has not been scheduled yet")
        return func(self, *args, **kwargs)

    return wrapper


# 用于封装一组请求的集合，请求被分配给特定的replica执行
# 封装了与一批请求相关的数据和行为，包括请求的调度、执行和完成状态
class Batch(BaseEntity):
    def __init__(
        self,
        replica_id: int,
        # 封装了当下待处理的请求集合
        requests: List[Request],
        num_tokens: List[int],
    ) -> None:
        self._id = Batch.generate_id()
        self._replica_id = replica_id

        self._requests = requests
        # 每个请request中包含的token数量
        self._num_tokens = num_tokens
        self._total_num_tokens = sum(num_tokens)
        # 预填充token列表：已经进行过预处理的token
        self._num_prefill_tokens = sum(
            [
                (t if not r.is_prefill_complete else 0)
                for r, t in zip(self.requests, self._num_tokens)
            ]
        )

        self._total_num_tokens_rounded = (self._total_num_tokens + 7) // 8 * 8

        self._scheduled_at = None
        self._completed_at = None
        self._scheduled = False
        self._completed = False

    @property
    def replica_id(self) -> int:
        return self._replica_id

    @property
    def creation_time(self) -> float:
        return self._creation_time

    @property
    def num_tokens(self) -> List[int]:
        return self._num_tokens

    @property
    def total_num_tokens(self) -> int:
        return self._total_num_tokens

    @property
    def num_prefill_tokens(self) -> int:
        return self._num_prefill_tokens

    @property
    def num_decode_tokens(self) -> int:
        return self.total_num_tokens - self.num_prefill_tokens

    @property
    @check_scheduled
    def scheduled_at(self) -> float:
        return self._scheduled_at

    @property
    @check_completed
    def completed_at(self) -> float:
        return self._completed_at

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def scheduled(self) -> bool:
        return self._scheduled

    @property
    def size(self) -> int:
        return len(self._requests)

    @property
    def requests(self) -> List[Request]:
        return self._requests

    @property
    def request_ids(self) -> List[int]:
        return [request.id for request in self._requests]

    @property
    def all_requests_completed(self) -> bool:
        return all([request.completed for request in self._requests])

    # 对request进行调度，更新调度时间并通知每个请求
    def on_schedule(
        self,
        time: float,
    ) -> None:
        self._scheduled_at = time
        self._scheduled = True

        for request in self._requests:
            request.on_batch_schedule(time)

    # 在这批请求处理结束时被调用，更新完成时间和处理每个请求的结束逻辑。
    def on_batch_end(self, time: float):
        self._completed = True
        self._completed_at = time

        for request, num_tokens in zip(self._requests, self._num_tokens):
            request.on_batch_end(time, num_tokens)

    # 被抢占的request集合
    @property
    def preempted_requests(self) -> List[Request]:
        return [request for request in self._requests if request.preempted]
    
    # 完成的request集合
    @property
    def completed_requests(self) -> List[Request]:
        return [request for request in self._requests if request.completed]

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "size": self.size,
            "replica_id": self._replica_id,
            "scheduled_at": self._scheduled_at,
            "completed_at": self._completed_at,
            "scheduled": self._scheduled,
            "request_ids": self.request_ids,
            "num_tokens": self._num_tokens,
            "num_prefill_tokens": self.num_prefill_tokens,
            "num_decode_tokens": self.num_decode_tokens,
        }
