from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


# 将request分配 给 当前未完成请求数量最少的replica 执行
class LORGlobalScheduler(BaseGlobalScheduler):
    """
    Least outstanding requests (LOR) global scheduler.
    """

    # 实现调度算法
    def schedule(self) -> List[Tuple[int, Request]]:
        # 按照到达时间对request进行排序
        self.sort_requests()

        request_mapping = []
        # keep a map of replica_id -> replica_scheduler
        # this is used to find the replica with the least outstanding requests
        # 建立replica-对应replica未完成request数量 的 字典
        pending_requests_map = {
            replica_scheduler.replica_id: replica_scheduler.num_pending_requests
            for replica_scheduler in self._replica_schedulers.values()
        }

        # using a very simple implementation here, to keep wiring simple
        # 对request-queue中的所有request进行响应
        while self._request_queue:
            request = self._request_queue.pop(0)
            replica_id = min(pending_requests_map.items(), key=lambda x: x[1])[0]
            pending_requests_map[replica_id] += 1
            request_mapping.append((replica_id, request))

        return request_mapping
