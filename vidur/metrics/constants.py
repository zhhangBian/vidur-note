""" File to store names for different metrics captured """

import enum


class OperationMetrics(enum.Enum):
    MLP_UP_PROJ = "mlp_up_proj"
    MLP_ACTIVATION = "mlp_activation"
    MLP_DOWN_PROJ = "mlp_down_proj"
    MLP_DOWN_PROJ_ALL_REDUCE = "mlp_down_proj_all_reduce"
    ATTN_PRE_PROJ = "attn_pre_proj"
    ATTN_POST_PROJ = "attn_post_proj"
    ATTN_POST_PROJ_ALL_REDUCE = "attn_post_proj_all_reduce"
    ATTN_PREFILL = "attn_prefill"
    ATTN_KV_CACHE_SAVE = "attn_kv_cache_save"
    ATTN_DECODE = "attn_decode"
    ATTN_ROPE = "attn_rope"
    PIPELINE_SEND_RECV = "pipeline_send_recv"
    ADD = "add"
    INPUT_LAYERNORM = "input_layernorm"
    POST_ATTENTION_LAYERNORM = "post_attention_layernorm"


class CpuOperationMetrics(enum.Enum):
    SCHEDULE = "schedule"
    SAMPLER_E2E = "sample_e2e"
    PREPARE_INPUTS_E2E = "prepare_inputs_e2e"
    MODEL_EXECUTION_E2E = "model_execution_e2e"
    PROCESS_MODEL_OUTPUTS = "process_model_outputs"
    RAY_COMM_TIME = "ray_comm_time"


# 各种请求的时间枚举
class RequestMetricsTimeDistributions(enum.Enum):
    # 请求的端到端（End-to-End）时间，即从请求开始到结束的总时间
    REQUEST_E2E_TIME = "request_e2e_time"
    # 标准化的请求端到端时间，可能用于消除不同请求间的差异，以便进行比较
    REQUEST_E2E_TIME_NORMALIZED = "request_e2e_time_normalized"
    # 请求的执行时间，即实际处理请求所花费的时间
    REQUEST_EXECUTION_TIME = "request_execution_time"
    # 标准化的请求执行时间，用于比较不同请求的执行效率
    REQUEST_EXECUTION_TIME_NORMALIZED = "request_execution_time_normalized"
    # 模型执行时间，如果请求涉及机器学习模型，这可能指的是模型推理所花费的时间
    REQUEST_MODEL_EXECUTION_TIME = "request_model_execution_time"
    # 标准化的模型执行时间，用于比较不同模型或请求的推理效率
    REQUEST_MODEL_EXECUTION_TIME_NORMALIZED = "request_model_execution_time_normalized"
    # 请求的抢占时间，可能指的是高优先级任务抢占当前任务所导致的时间损失
    REQUEST_PREEMPTION_TIME = "request_preemption_time"
    # 请求的调度延迟，即请求提交到开始执行之间的等待时间
    REQUEST_SCHEDULING_DELAY = "request_scheduling_delay"
    # 标准化的请求执行时间加上抢占时间
    REQUEST_EXECUTION_PLUS_PREEMPTION_TIME = "request_execution_plus_preemption_time"
    # 预填充的端到端时间，可能指的是在请求处理前，为了优化性能而进行的预处理步骤所花费的时间
    REQUEST_EXECUTION_PLUS_PREEMPTION_TIME_NORMALIZED = (
        "request_execution_plus_preemption_time_normalized"
    )
    PREFILL_TIME_E2E = "prefill_e2e_time"
    # 预填充的执行时间加上抢占时间
    PREFILL_TIME_EXECUTION_PLUS_PREEMPTION = "prefill_time_execution_plus_preemption"
    # 标准化的预填充执行时间加上抢占时间
    PREFILL_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED = (
        "prefill_time_execution_plus_preemption_normalized"
    )
    # 解码时间加上抢占时间的标准化值，可能用于衡量解码请求（如视频流解码）的性能
    DECODE_TIME_EXECUTION_PLUS_PREEMPTION_NORMALIZED = (
        "decode_time_execution_plus_preemption_normalized"
    )


class TokenMetricsTimeDistribution(enum.Enum):
    DECODE_TOKEN_EXECUTION_PLUS_PREMPTION_TIME = (
        "decode_token_execution_plus_preemption_time"
    )


class RequestMetricsHistogram(enum.Enum):
    REQUEST_INTER_ARRIVAL_DELAY = "request_inter_arrival_delay"
    REQUEST_NUM_TOKENS = "request_num_tokens"
    REQUEST_PREFILL_TOKENS = "request_num_prefill_tokens"
    REQUEST_DECODE_TOKENS = "request_num_decode_tokens"
    REQUEST_PD_RATIO = "request_pd_ratio"
    REQUEST_NUM_RESTARTS = "request_num_restarts"


class BatchMetricsCountDistribution(enum.Enum):
    BATCH_NUM_TOKENS = "batch_num_tokens"
    BATCH_NUM_PREFILL_TOKENS = "batch_num_prefill_tokens"
    BATCH_NUM_DECODE_TOKENS = "batch_num_decode_tokens"
    BATCH_SIZE = "batch_size"


class BatchMetricsTimeDistribution(enum.Enum):
    BATCH_EXECUTION_TIME = "batch_execution_time"


class RequestCompletionMetricsTimeSeries(enum.Enum):
    REQUEST_ARRIVAL = "request_arrival"
    REQUEST_COMPLETION = "request_completion"


class TokenCompletionMetricsTimeSeries(enum.Enum):
    PREFILL_COMPLETIONS = "prefill_completion"
    DECODE_COMPLETIONS = "decode_completion"
