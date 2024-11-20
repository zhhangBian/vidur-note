## 问题询问

### 1116

1. 使用小ML模型对运行时间做估计，但是具体的训练不是很明白，构建了Transformer结构，但是其中的训练过程不是很清晰
2. 对config_search的部分没有看的很明白，论文重提到了qps是单调的，但是实际是为什么不是很清楚，之前不太了解相关的指标。需要查看论文来了解吗
3. 文章使用一个长度为$\sqrt{\sum_{i=1}^p p_i^2}$的prefill当做一个batch（与上述一个batch中有P个prefill等价），来计算时间，这个的具体原理想的不是很明白

### 1120

#### 1

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

#### 2

对于Replica这个层次的抽象还存在着一些疑惑，目前的认识是：

- 在LLM的训练模拟中，广泛采用了分布式训练，会有多个训练实体，都遵循着相同的一套数据
- 





