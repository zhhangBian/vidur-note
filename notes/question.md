## 1116

1. 使用小ML模型对运行时间做估计，但是具体的训练不是很明白，构建了Transformer结构，但是其中的训练过程不是很清晰
2. 对config_search的部分没有看的很明白，论文重提到了qps是单调的，但是实际是为什么不是很清楚，之前不太了解相关的指标。需要查看论文来了解吗
3. 文章使用一个长度为$\sqrt{\sum_{i=1}^p p_i^2}$的prefill当做一个batch（与上述一个batch中有P个prefill等价），来计算时间，这个的具体原理想的不是很明白

## 1120

### 1

在对Profiling Communication Operators的模拟参数中，只对`all_reduce, send_recv`两类算子提供了模拟参数（data/compute），没有对all_gather阶段进行模拟

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

### 提issue

在论文中，提到了对于Profiling Communication Operators，Ther are three collective operations, all_reduce, all_gather, send-recv

但是在对Profiling Communication Operators的设备模拟数据`data/compute`中，只对`all_reduce, send_recv`两类算子提供了模拟参数，没有对all_gather阶段进行模拟

我阅读了`vidur/profiling`中的相关代码，发现似乎认为是对all_gather的计算是设备无关的，就没有对参数进行引入，但是难道不应该是设备相关的吗。

以及在`vidur/profiling/collectives/main.py`中的`--collective`参数处理只支持了`choices=["all_reduce", "send_recv"]`，这是为什么呢

以上是我在阅读代码中的困惑，麻烦你做出一定的解答，如果有认识不到位之处，也麻烦纠正



Hello Vidur,

Thank you for sharing your work. While reading the code and documentation, I encountered some questions related to the *Profiling Communication Operators* mentioned in the paper.

In the paper, it is noted that there are three collective operations: `all_reduce`, `all_gather`, and `send_recv`. However, in the simulated device data located at `data/compute`, it seems that simulation parameters are provided only for `all_reduce` and `send_recv`. There are no simulation parameters for the `all_gather` operation.

After reviewing the relevant code in `vidur/profiling`, it appears that `all_gather` is treated as device-independent, and thus its parameters are not explicitly introduced. However, isn’t `all_gather` typically device-dependent? If so, could you clarify why it is treated as device-independent in this case?

Additionally, in `vidur/profiling/collectives/main.py`, the `--collective` argument only supports `choices=["all_reduce", "send_recv"]`. Could you explain the rationale behind excluding `all_gather` as an option here?

The above are my points of confusion while going through the code. I would greatly appreciate it if you could provide clarification or corrections if I have misunderstood any part of your work.

Thank you in advance for your time and insights!



感谢你的回答！

在我的理解中，经历过一次row-partition 和 column-partition 就是使用 all-reduce 了，如果只是一次row-partition or 一次 column partition 就必须要all-gather 了

但是我依然好奇，你能不能further explain “only use all reduce and send/recv operations -- which are sufficient to represent tensor and pipeline parallelism”，解释一下其中的原理以及是如何实现的

麻烦你了，感谢



Thank you for your response!

From my understanding, performing both a row-partition and a column-partition would involve using all-reduce, while performing only a row-partition or only a column-partition would require an all-gather operation.

However, I’m still curious. Could you kindly elaborate further on what you mean by “only use all-reduce and send/recv operations -- which are sufficient to represent tensor and pipeline parallelism”? Could you also explain the underlying principles and how this is implemented?

Thank you so much for your help!

## 1126

### 1

Unified Algorithm for Schedulers中提到了VLLM的调度算法：

- 对$R_w$队列设置

### 2

论文中的一些符号问题：

你好。我阅读了你的论文The Effect of Scheduling and Preemption on the Efficiency of LLM Inference Serving。你做了精彩的工作，但是我对CSP中的部分Variables定义有问题。

在你的文章3.2节的Variables部分中，$c_{i, j}$是token数，但是在Constraints中的Memory Management部分又参与了$m_{i, j}$的计算。token数参与内存计算，这里我无法理解。请问你能否解释一下

十分感谢

## 1116

1. 使用小ML模型对运行时间做估计，但是具体的训练不是很明白，构建了Transformer结构，但是其中的训练过程不是很清晰
2. 对config_search的部分没有看的很明白，论文重提到了qps是单调的，但是实际是为什么不是很清楚，之前不太了解相关的指标。需要查看论文来了解吗
3. 文章使用一个长度为$\sqrt{\sum_{i=1}^p p_i^2}$的prefill当做一个batch（与上述一个batch中有P个prefill等价），来计算时间，这个的具体原理想的不是很明白

## 1120

### 1

在对Profiling Communication Operators的模拟参数中，只对`all_reduce, send_recv`两类算子提供了模拟参数（data/compute），没有对all_gather阶段进行模拟

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

### 提issue

在论文中，提到了对于Profiling Communication Operators，Ther are three collective operations, all_reduce, all_gather, send-recv

但是在对Profiling Communication Operators的设备模拟数据`data/compute`中，只对`all_reduce, send_recv`两类算子提供了模拟参数，没有对all_gather阶段进行模拟

我阅读了`vidur/profiling`中的相关代码，发现似乎认为是对all_gather的计算是设备无关的，就没有对参数进行引入，但是难道不应该是设备相关的吗。

以及在`vidur/profiling/collectives/main.py`中的`--collective`参数处理只支持了`choices=["all_reduce", "send_recv"]`，这是为什么呢

以上是我在阅读代码中的困惑，麻烦你做出一定的解答，如果有认识不到位之处，也麻烦纠正



Hello Vidur,

Thank you for sharing your work. While reading the code and documentation, I encountered some questions related to the *Profiling Communication Operators* mentioned in the paper.

In the paper, it is noted that there are three collective operations: `all_reduce`, `all_gather`, and `send_recv`. However, in the simulated device data located at `data/compute`, it seems that simulation parameters are provided only for `all_reduce` and `send_recv`. There are no simulation parameters for the `all_gather` operation.

After reviewing the relevant code in `vidur/profiling`, it appears that `all_gather` is treated as device-independent, and thus its parameters are not explicitly introduced. However, isn’t `all_gather` typically device-dependent? If so, could you clarify why it is treated as device-independent in this case?

Additionally, in `vidur/profiling/collectives/main.py`, the `--collective` argument only supports `choices=["all_reduce", "send_recv"]`. Could you explain the rationale behind excluding `all_gather` as an option here?

The above are my points of confusion while going through the code. I would greatly appreciate it if you could provide clarification or corrections if I have misunderstood any part of your work.

Thank you in advance for your time and insights!



感谢你的回答！

在我的理解中，经历过一次row-partition 和 column-partition 就是使用 all-reduce 了，如果只是一次row-partition or 一次 column partition 就必须要all-gather 了

但是我依然好奇，你能不能further explain “only use all reduce and send/recv operations -- which are sufficient to represent tensor and pipeline parallelism”，解释一下其中的原理以及是如何实现的

麻烦你了，感谢



Thank you for your response!

From my understanding, performing both a row-partition and a column-partition would involve using all-reduce, while performing only a row-partition or only a column-partition would require an all-gather operation.

However, I’m still curious. Could you kindly elaborate further on what you mean by “only use all-reduce and send/recv operations -- which are sufficient to represent tensor and pipeline parallelism”? Could you also explain the underlying principles and how this is implemented?

Thank you so much for your help!

## 1126

### 1

Unified Algorithm for Schedulers中提到了VLLM的调度算法：

- 对$R_w$队列设置

### 2

论文中的一些符号问题：

你好。我阅读了你的论文The Effect of Scheduling and Preemption on the Efficiency of LLM Inference Serving。你做了精彩的工作，但是我对CSP中的部分Variables定义有问题。

在你的文章3.2节的Variables部分中，$c_{i, j}$是token数，但是在Constraints中的Memory Management部分又参与了$m_{i, j}$的计算。token数参与内存计算，这里我无法理解。请问你能否解释一下

十分感谢



Dear Dr. Kyoungmin Kim, Dr. Kijae Hong, Dr. Caglar Gulcehre, and Dr. Anastasia Ailamaki,

I hope this message finds you well.

I am Bian Zhuohang, a student at Beihang University. I have had the pleasure of reading your esteemed paper titled "The Effect of Scheduling and Preemption on the Efficiency of LLM Inference Serving," and I must say, the work you have presented is truly impressive. Your contributions to the field of large language model (LLM) inference systems are both insightful and valuable.

However, I have encountered a point of confusion regarding the definitions of certain variables within the Constraint Satisfaction Problem (CSP) framework that you have outlined. Specifically, in Section 3.2, under the "Variables" subsection, \( c_{i, j} \) is defined as the number of tokens processed by request \( i \) in batch \( j \). Later, in the "Memory Management" constraints, \( c_{i, j} \) appears to be involved in the calculation of \( m_{i, j} \), which I find somewhat perplexing.

Could you kindly elaborate on the role of \( c_{i, j} \) in the context of memory management? Understanding how the token count is integrated into the memory usage calculations would greatly assist me in my own studies and application of the concepts presented in your paper.

I would be extremely grateful for any clarification you could provide.

Thank you very much for your time and consideration. I am looking forward to your response and wish you continued success in your research endeavors.

Warm regards,
