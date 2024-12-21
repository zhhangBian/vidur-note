记录了simulate过程中的相关参数

文件结构为`{DEVICE}/{MODEL}`

## profilling

profilling相关过程的参数，包括MLP的参数和Attention机制的参数


## computing

推理过程中的通信过程相关参数

分为了两大类通信类型，实际上是根据通信算子的类型进行分类的
（可以见论文）：

- all_reduce
- send_recv

没有all-gather算子，用于张量并行部分的相关参数
