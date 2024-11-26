# 注意力层的输入配置
class AttentionInput:
    def __init__(
        self,
        prefill_chunk_size: int,
        kv_cache_size: int,
        batch_size: int,
        is_prefill: bool,
    ):
        # 预填充块的大小
        self.prefill_chunk_size = prefill_chunk_size
        # KV-cache大小
        self.kv_cache_size = kv_cache_size
        # 批次大小
        self.batch_size = batch_size
        # 是否为预填充请求
        self.is_prefill = is_prefill

    # 检查当前的 AttentionInput 对象是否有效
    def is_valid(self, max_seq_len: int):
        # 如果是预填充请求
        if self.is_prefill:
            if self.batch_size != 1:
                return False
            elif self.prefill_chunk_size == 0:
                return False
            elif self.prefill_chunk_size + self.kv_cache_size > max_seq_len:
                return False
        else:
            if self.prefill_chunk_size > 0:
                return False
            elif self.kv_cache_size == 0:
                return False
            elif self.kv_cache_size > max_seq_len:
                return False
        return True

    # 检查当前的 AttentionInput 对象是否在内存限制范围内
    def is_under_memory_limit(self, max_num_tokens: int):
        return (
            self.batch_size * (self.kv_cache_size + self.prefill_chunk_size)
            <= max_num_tokens
        )

    def __str__(self):
        return f"prefill_chunk_size: {self.prefill_chunk_size}, kv_cache_size: {self.kv_cache_size}, batch_size: {self.batch_size}, is_prefill: {self.is_prefill}"
