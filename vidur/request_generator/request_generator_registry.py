from vidur.request_generator.synthetic_request_generator import (
    SyntheticRequestGenerator,
)
from vidur.request_generator.trace_replay_request_generator import (
    TraceReplayRequestGenerator,
)
from vidur.types import RequestGeneratorType
from vidur.utils.base_registry import BaseRegistry


class RequestGeneratorRegistry(BaseRegistry):
    pass


# 模拟生成人工请求
RequestGeneratorRegistry.register(
    RequestGeneratorType.SYNTHETIC, SyntheticRequestGenerator
)
RequestGeneratorRegistry.register(
    RequestGeneratorType.TRACE_REPLAY, TraceReplayRequestGenerator
)
