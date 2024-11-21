### 1116

1. 使用小ML模型对运行时间做估计，但是具体的训练不是很明白，构建了Transformer结构，但是其中的训练过程不是很清晰
2. 对config_search的部分没有看的很明白，论文重提到了qps是单调的，但是实际是为什么不是很清楚，之前不太了解相关的指标。需要查看论文来了解吗
3. 文章使用一个长度为$\sqrt{\sum_{i=1}^p p_i^2}$的prefill当做一个batch（与上述一个batch中有P个prefill等价），来计算时间，这个的具体原理想的不是很明白

## 1120

### 1

在对Profiling Communication Operators的模拟参数中，只对`all_reduce, send_recv`两类算子提供了模拟参数（data/compute），没有对all_gather阶段进行模拟。

这是不是因为all_gather阶段的张量并行操作默认执行了，不同策略之间的差异较小，就没有对参数进行引入

### 2

`sklearn_execution_time_predictor`中有一些时间是直接使用参数量进行直接计算的，依据是什么，想要学习的话可以看哪些论文呀

### 3

在`MemoryPlanner`中，有两个函数`get_max_batch_size`和`get_max_request_slots`，区别是是否乘以流水线的stage数目，这两个的区别是什么呢。

我的理解是batch_size即一批的request，是顺序的、依次响应的request数目，slots进一步考虑了流水线并行的效果。但是为什么效果是max_request_slots是前一个乘以流水线阶段数，这样子的流水线并行难道不会占用超量的内存的吗。

因为我觉得在流水线的每个阶段依然需要占用不变的内存，如果只是简单的相乘，会导致内存占用超标

gpt给的解答是，感觉还是有些疑惑

> 在 Vidur 框架中，流水线并行（Pipeline Parallelism）是一种将模型的不同层分配到不同计算设备上的并行策略，这样可以提高模型推理的吞吐量。流水线并行的主要特点是将模型的层序列化，并在多个设备上并行处理，类似于硬件流水线。这种方式可以显著提高处理速度，尤其是在处理大量请求时，因为它允许不同层的同时处理。
>
> 然而，流水线并行可能会增加处理单个请求的延迟，因为每个请求必须通过整个流水线才能完成。这就是为什么 Vidur 框架中的 `MemoryPlanner` 类要计算最大批次大小（`get_max_batch_size`）和最大请求插槽数（`get_max_request_slots`）的原因。这两个函数帮助 Vidur 框架管理内存使用，确保在流水线的每个阶段都不会超过可用内存的限制。
>
> `get_max_batch_size` 函数计算在不超过内存限制的情况下，单个批次中可以处理的最大请求数量。这个值是基于每个请求所需的KV-Cache内存和模型参数内存的需求计算得出的。
>
> `get_max_request_slots` 函数则进一步考虑了流水线的阶段数，计算在整个流水线中可以同时处理的最大请求数量。这个值是 `get_max_batch_size` 的结果乘以流水线的阶段数（`num_pipeline_stages`），它反映了在整个流水线中可以并行处理的请求数量。
>
> 流水线并行不会占用超过可用内存的内存，因为 Vidur 框架通过 `MemoryPlanner` 类的计算确保了每个流水线阶段的内存需求都被合理管理。`MemoryPlanner` 根据模型规格和并行配置计算出可用于KV-Cache的内存数，然后 Vidur 使用这些信息来提供高级管理API，这些API被用于实现自定义的批处理策略。这样，Vidur 可以确保在流水线并行处理中，每个阶段的内存使用都不会超过副本的总内存容量，从而避免内存溢出的问题。

### 4

在运行过程中的报错是不是不影响的，寻找了报错栈，在`*def* plot_histogram(self, path: str, plot_name: str) -> None:`这个函数中，但似乎并没有影响结果的输出，有正常的画图

```(空)
INFO 11-20 14:50:55 simulator.py:60] Starting simulation with cluster: Cluster({'id': 0, 'num_replicas': 1}) and 128 requests
INFO 11-20 14:50:59 simulator.py:80] Simulation ended at: 240.4991063385136s
INFO 11-20 14:50:59 simulator.py:83] Writing output
Error importing optional module IPython.core.display
Traceback (most recent call last):
  File "/home/pigkiller/Mlsys/vidur/env/lib/python3.10/site-packages/_plotly_utils/optional_imports.py", line 28, in get_module
    return import_module(name)
  File "/home/pigkiller/Mlsys/vidur/env/lib/python3.10/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1050, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1027, in _find_and_load
  File "<frozen importlib._bootstrap>", line 992, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "<frozen importlib._bootstrap>", line 1050, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1027, in _find_and_load
  File "<frozen importlib._bootstrap>", line 992, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "<frozen importlib._bootstrap>", line 1050, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1027, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1006, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 688, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 883, in exec_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
  File "/home/pigkiller/Mlsys/vidur/env/lib/python3.10/site-packages/IPython/__init__.py", line 55, in <module>
    from .terminal.embed import embed
  File "/home/pigkiller/Mlsys/vidur/env/lib/python3.10/site-packages/IPython/terminal/embed.py", line 16, in <module>
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
  File "/home/pigkiller/Mlsys/vidur/env/lib/python3.10/site-packages/IPython/terminal/interactiveshell.py", line 48, in <module>
    from .debugger import TerminalPdb, Pdb
  File "/home/pigkiller/Mlsys/vidur/env/lib/python3.10/site-packages/IPython/terminal/debugger.py", line 18, in <module>
    from concurrent.futures import ThreadPoolExecutor
  File "<frozen importlib._bootstrap>", line 1075, in _handle_fromlist
  File "/home/pigkiller/Mlsys/vidur/env/lib/python3.10/concurrent/futures/__init__.py", line 49, in __getattr__
    from .thread import ThreadPoolExecutor as te
  File "/home/pigkiller/Mlsys/vidur/env/lib/python3.10/concurrent/futures/thread.py", line 37, in <module>
    threading._register_atexit(_python_exit)
  File "/home/pigkiller/Mlsys/vidur/env/lib/python3.10/threading.py", line 1504, in _register_atexit
    raise RuntimeError("can't register atexit after shutdown")
RuntimeError: can't register atexit after shutdown
INFO 11-20 14:51:03 simulator.py:87] Metrics written
INFO 11-20 14:51:04 simulator.py:95] Chrome event trace written
```

