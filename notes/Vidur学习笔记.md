# 论文阅读

## 论文简介

Vidur是一个大型语言模型部署模拟框架，通过实验数据和预测模型相结合，准确模拟不同配置下的模型性能，显著降低成本和提高效率，将原本需要高昂成本的实验过程缩短至数小时，并可将潜在成本降低至实际成本的零头。

Vidur的作用是找到最有效的**部署配置**，而无需进行实际的物理实验：实际的物理实验消耗极大的GPU时间，在时间和资金上都花费极大。

Vidur 结合了实验数据和预测建模来模拟 LLMs 在不同配置下的性能。这种仿真允许评估关键性能指标，如延迟和吞吐量，而无需进行昂贵且耗时的物理试验。

Vidur 的一个关键组件是其配置搜索工具 Vidur-Search，它可以自动探索部署配置。该工具可以有效地确定满足预定义性能标准的最具成本效益的设置。

## Abstract

部署实验的主要耗时在于：需要探索由并行化策略、批处理技术和调度策略等系统旋钮形成的大型配置空间

Vidur——一个大规模，高保真度的，易于扩展的**LLM推理性能仿真框架**。 

Vidur**使用实验性profiling和预测建模结合来模拟LLM算子的性能**，并通过估计延迟和吞吐量等几种感兴趣的指标来评估不同工作负载的端到端推理性能。

模拟误差率仅在9%，但节省了大量的资源和时间。

## 简介

进行推理的成本很高，需要进行部署优化：

- 确定模型并行策略
- 确定模型的调度算法
- 一系列的配置参数，如batch size等，以满足所需的吞吐量和延迟限制
- 生成一些具有代表性的工作负载以进行测试，**这部分的成本很高**

系统地优化具有数百个配置选项的数十个模型的部署是昂贵且不切实际的。

Vidur是模拟器，基于模拟可以使用Vidur-Search来探索最佳配置，使得不用进行实机实验，可以降低时间和成本。

### 模拟的难点

模拟主要针对推理阶段进行模拟，模拟的难点有：

1. 需要在更细的时间进度上保持准确：相较于训练推理所花的时间更少
2. 相较于训练的输入大小几乎确定，推理阶段的输入有较大改变，以及调度策略对预填充和解码阶段的影响，导致迭代延迟有显著的变化
   1. 不可能模拟所有的输入情况，模拟需要分析预测策略
3. 由于推理工作负载的动态和有状态特性，预测中的小错误会导致级联效应

### Vidur的工作

Vidur为了解决上述问题，将LLM进行了抽象，为：绝大多数LLM共享类似的架构，可以拆分为token层次、序列层次和通信运算符

1. 首先接受模型规范，确定需要分析的各种运算符和一组最小的输入大小
2. 构建时间预测模型
3. 预测各类指标，如首次令牌时间（TTFT）、令牌间隔时间（TBT）、延迟、吞吐量，以及集群级指标，例如模型浮点利用率（MFU）和内存利用率。

开发了Vidur-Bench来评估推理心理。跟踪推理中的负载，是一个Benchmark，包括各种工作负载模式、调度器和服务框架，以及流行硬件的分析信息

Vidur-Search通过使用Vidur来模拟各种参数下的效果，确定给定模型、工作负载对的最高吞吐量/成本配置

## 背景

### LLM

LLM基于Transformer的self-attention机制构建

自我注意机制帮助语言模型学习输入序列中不同元素之间的关系，并随后产生输出序列

一个LLM包含两个重要模块：自注意力和多层感知机

### LLM推理效率

LLM推理请求处理由两个不同的阶段组成——预填充和解码

预填充阶段处理整个用户输入提示并生成第一个token，之后，由自回归机制每次生成一个token，直至终结token的出现

解码过程需要访问先前处理的token的key和value激活，以执行注意力操作。为了避免重复计算，现代LLM推理系统将它们存储在KV Cache中。

#### 张量并行TP

张量并行：TP通过将模型权重和KV缓存平均分配给GPU工作线程，在参与的GPU上对每一层进行分片

1. 通过更大的批处理大小提高推理吞吐量
2. 通过在多个GPU上拆分每个运算符来降低推理的延迟

但是张量并行需要频繁的块通信

#### 流程并行PP-Pipeline Parallelism

将不同的流程运行在不同GPU上，每个GPU负责一个运行的stage，输出激活通过send/recv操作跨GPU边界传输

#### Tradeoff

Tradeoff的两端是预填充和解码

- 预填充优先级调度通过生成具有更大批量的调度来实现更高的吞吐量，但会带来更高的延迟成本
- 解码优先级调度器可以实现低延迟，但以较低的吞吐量为代价

#### 模拟的配置空间

LLM推理的最优配置是模型m和trace t的函数，因此配置空间的复杂度是$O(|M|×|T|)$

## 模拟LLM的难点

原先的SOTA-LLM模拟器聚焦训练过程，很少有聚集推理过程的

- 模拟的时间尺度：LLM推理的时间粒度更细，相较于传统的DNN模拟需要更精细的模拟粒度
  - LLM推理是一项对延迟更敏感的任务，迭代时间可以更短
- LLM的每次模拟需求可能变化较大
  - 涉及到多个阶段：prefill和decode
  - 需求的序列长度可能变化很大
  - 在线推理过程中的批处理大小会根据系统负载和工作负载特性而变化
- 在训练任务中，每个iteration都是独立的，不会相互影响。而在推理阶段，request动态到达，如果哪一个batch的模拟误差太大，导致batch模式出现改变，会引起后续的级联错误。

## Vidur

Vidur在副本和集群级别模拟推理栈所有层的行为，包括模型执行和请求调度的各个层

### 关键观点

- 大多数LLM的架构相似
  - 只有激活函数、归一化层、残差连接等地方有不同的选择。
  - 这样可以用一个通用的模型规范来表示多数模型。除此之外，Vidur只需要建模一小部分算子
- LLM的算子可以进行不同分类
  - 一些算子的运行时间取决于这个batch中所有request的上下文长度之和，有的算子的运行时间只取决于当前Iteration的token数量。这样可以对不同类别的算子采取不同方式进行profile
  - 因为在decode阶段attention是memory-bound的计算，因此只使用一个batch的request的KV-Cache总量，就可以精确地建模kernel运行时间
- 并行策略的自动profiling
  - Vidur结合了LLM并行策略的领域知识
  - Vidur结合了关于LLM并行策略的领域知识，使其能够识别在每个设备上执行的计算子集。在profile阶段，我们从model specification中自动识别每个算子的张量切分配置

### 系统概述

Vidur主要有两个阶段。

第一个是**模型装载阶段**：使用model spec生成需要profile的算子集合。

- Vidur profiler收集算子的运行时间特征，并送入runtime estimator。
- 为了减少加入新模型的开销，在profiling过程中收集数据，然后训练一个ml模型用于预测使用的不同参数的算子运行时间。
- 在分析阶段收集最少的数据，然后训练小型机器学习模型，以生成在模拟过程中可能触发这些操作的大范围参数的预测
- 此阶段通过Vidur runtime estimator处理，生成operation-wise的runtime lookup table。

在模型装载完成后，用户可以使用不同的调度策略、并行策略和不同的workload来进行模拟。

- 在Vidur这个event-driven simulator中是一个可插拔的Hierarchical Scheduler，支持多种batching策略和内存管理功能。
- 模拟器提供详细的指标，例如request级别（normalized latency, time-to-first-token, time-between-tokens）和cluster级别（FLOPs utilization, KV-Cache utilization）。

### Profiler：分析器

基于绝大部分LLM相似的特点：基于此特点减少分析的复杂性

#### 算子

所有的算子可以被分为三类：

- Token-level Operator：
  - 一些算子例如linear、avtivation的**操作数维度取决于模型架构**
  - 但它们的**运行时间**只取决于batch中**要处理的token总数**
- Sequence-level Operator：
  - attention不仅取决于当前batch中的token数量，还取决于每个request的上下文长度。
- Communication Operator：all-reduce、all-gather这样的**通信算子**运行时间取决于要传输的数据量，与模型架构无关。

#### Profiling Token-level Operator

token-level算子主要分为两类：矩阵乘法和point-wise（或reduction、nomalization、activation）。

根据模型的规格，生成所有的**张量并行切分配置**，并进行profile。

这样在一个GPU上做profile就可以获得不同并行配置的trace。

#### Profiling Sequence-level Operators

批处理序列级运算符（如attention内核）对批处理中请求的上下文长度很敏感，从而将输入的状态空间扩展到配置文件

将prefill和decode阶段分开，分别做attention的profile，因为不同阶段的计算特性不同

- 在处理prefill阶段的attention时，发现**attention的时间是长度的二次方**

  - 假设有一个batch里有P个序列需要prefill，每个长度为$p_i$，则整个batch prefill attention所用时间与$\sum_{i=1}^p p_i^2$成正比

  - 本文使用一个长度为$\sqrt{\sum_{i=1}^p p_i^2}$的prefill当做一个batch（与上述一个batch中有P个prefill等价），来计算时间

    > 为什么：
    >
    > attention计算时间和prefill sequence的长度l为$O(l^2)$的关系
    >
    > 如果把p1和 p2的两个request 合并成 1个request forward，在attention层面看来需要计算 ($p_1^2$+$p_2^2$)的计算量，就会等价于 $\sqrt{\sum_{i=1}^p p_i^2}$ 长度的prefill request

- decode阶段的attention是**memory-bound的算子**，受到内存中KV-cache的约束

  - 算子的运行时间主要取决于**从KV-Cache中取数据的总量**
  - 而不需要知道一个batch中每个request的上下文长度是多少

当批处理中**不同请求的上下文长度之间存在较大偏差**时，注意力内核可能无法有效地并行化KV缓存获取操作。

然而现有的序列并行attention kernel（PagedAttention v2、FlashDecoding）可以高效地处理这个情况。因此只使用KV-Cache read总量来建模decode是合理的。

#### Profiling Communication Operators

LLM推理中主要使用三种集合通信：

1. all-reduce
2. all-gather：用于张量并行
3. send-recv：用于流水线并行

这些算子的**执行时间是模型无关的**，因此针对不同拓扑在预先就做profile。

### 运行时预估

收集了简单的数据，使用小ML模型对运行时间做估计

在预测训练时间中，简单的多项式估计不能建模CUDA kernel复杂的tile和wave quantization的行为

使用了随机森林进行预估

### 分层调度器

使用了三层分层调度器

1. global：对request进行路由

   1. 用于标准负载平衡策略，如轮询和最小未完成请求

2. replica：对batch和memory进行管理

   1. 包含一个内存计划器，它使用模型规范和并行配置来**计算**KV Cache可用的内存。
   2. 计算得到可用内存后，使用高效内存管理API进行管理
   3. API的使用可以方便引入新的batch管理机制

   > replica：副本管理策略

3. replica stage：处理流水线阶段内的微批调度

## Vidur-Bench

在得到了预测器Vidur后，提出了一系列衡量推理系统性能的指标：

1. 工作负载模式
2. 调度、批处理和路由策略
3. 推理框架

### 数据集

LLM推理对工作负载有类型高度敏感，如在一个请求中的输入和输出token的数量

vLLM为KV-Cache增量分配物理内存，以在GPU上跑更大的batch size，这样当decode token多的时候跑得很好；然而当prompt length比output token数量更多时（如摘要任务），增量内存分配就不那么有用了。

这里的output token是可以进行预测的

### 性能指标

提供了一系列性能指标用于评估推理系统性能

- **Operator-level metrics**：
  - 包括operator的输入尺寸和执行时间
  - 可用于优化重型任务
- **Request-level metrics**
  - 包括scheduling delay, prefill完成时间, timeto-first-token (TTFT), time-between-tokens (TBT)
- **Replica-level metrics**
  - batch size, 每个Iteration处理的token数量, busy和idle时间, 每个replica的内存和计算利用率
- **Hardware-level metrics**
  - 集群的GPU FLOPs和内存利用率

## Vidur-Search

利用仿真结果帮助找到最优的配置

### 仿真约束

包括以下的几个方面：

- 输入：
  - LLM模型
  - 工作workload
  - 可用的GPU
- 约束：
  - SLO（例如TTFT、TBT、QPS）
- 搜索空间：
  - 并行策略（TP vs PP）、并行度、调度策略、调度器参数、batchsize、GPU型号选择
- 优化目标：
  - 最大化QPS per dollar
  - 系统的容量被定义为在**不增加排队延迟**的情况下每秒可以支持的最大查询数

核心是解决**约束下的最优化问题**

- 可能的QPS是无限的
- 对于给定的工作负载，任何系统配置都将具有最大的QPS容量，在该容量下，它可以在不累积请求队列的情况下处理输入请求
- 利用单调性来进行二分，来找到最大的QPS

最佳配置取决于输入工作负载，并且工作负载会随着时间的推移而变化；

## 验证

Evaluation主要回答两个问题：

- Vidur是否可以准确预测不同模型、并行策略、workload的端到端性能。
- Vidur是否可以在给定硬件配置下回答LLM部署的what-if问题

基于VLLM进行了验证

1. 工作负载的变化会极大地改变最佳配置
2. 由于架构细节的变化，即使是尺寸相似的模型也可能具有非常不同的性能特征

## 专有名词

### SKU

在讨论大模型推理加速时，提到的GPU SKU是指特定型号或配置的图形处理单元（Graphics Processing Unit）。SKU是库存保有单位（Stock Keeping Unit）的缩写，在商业中用来表示一个特定的产品变体。对于GPU来说，SKU可以指代不同的方面，比如不同的计算能力、内存大小、功耗、接口类型等。

### QPS

QPS，全称为Queries Per Second，中文译为“每秒查询率”

### Profilling

#### 宏观概念

profiling指的是对模型或用户行为进行分析和评估的过程，以**识别关键特征和性能指标**。具体来说，profiling在LLM中有几个相关的含义：

1. **性能分析工具**：在深度学习和模型优化中，profiling工具用于提供模型**训练或推理过程中**时间和资源消耗的见解，帮助**识别瓶颈和优化资源利用率**。

   1. PyTorch中的`torch.profiler`就是一个性能分析工具，它可以帮助开发者理解模型在执行过程中各个部分的时间和资源消耗情况。
   2. vLLM通过设置环境变量`VLLM_TORCH_PROFILER_DIR`来启用性能追踪。这个环境变量指定了保存追踪数据的目录

   **性能评估**：LLM-Profiler是一个工具，用于评估在线服务引擎的性能，包括速度和吞吐量，适配了多种常见的LLM推理框架。这个工具注重实际在线推理场景下的性能测试，考虑业务延迟要求和符合线上实际请求分布下的系统吞吐量。

2. **内存和计算优化**：在某些研究中，profiling用于**分析不同模块在LLM中的行为**，以便为它们的KV缓存提供不同的优化策略。例如，FastGen算法基于对不同模块行为的分析（即profiling），调整数据存储方式，以提高效率。

profiling在LLM中是一个多面性的概念，既包括技术性能分析，也包括用户行为分析和个性化服务的创建。

#### 在Vidur中的含义

在Vidur框架中，profiling指的是对LLM的**并行策略和操作符的自动性能分析过程**。这个过程包括以下几个关键方面：

1. **模型装载阶段**：在这个阶段，Vidur使用**模型规格**来生成**需要进行性能分析的计算操作符集合**。Vidur的分析器收集这些操作符的运行时间特征，并将其送入运行时估计器。

   **操作符分类**：所有操作符可以分为三类：Token-level Operator、Sequence-level Operator和Communication Operator。

   1. Token-level操作符的运行时间仅取决于批次中要处理的token总数；
   2. Sequence-level操作符的运行时间不仅取决于当前批次中的token数量，还取决于每个请求的上下文长度；
   3. Communication Operator如all-reduce、all-gather等通信操作符的运行时间取决于要传输的数据量，与模型架构无关。

2. **并行策略的自动分析**：不同的模型并行配置有**不同的内存、计算和网络通信特征**。

   1. Vidur结合了关于LLM并行策略的领域知识，使其能够**识别在每个设备上执行的计算子集**。
   2. 在profile阶段，Vidur**从model specification中自动识别每个算子的张量切分配置**。
   3. 因此，Vidur可以使用GPU做最少的profile来模拟各种并行化方案。

3. **运行时估计器**：

   1. 为了减少加入新模型的开销，Vidur在profiling过程中收集数据，然后训练一个机器学习模型用于预测使用的不同参数的算子运行时间。
   2. 这个通过Vidur运行时估计器处理，生成operation-wise的runtime lookup table。

4. **网络（集体操作）分析**：**网络分析不依赖于模型**，因此可以为所有模型使用相同的网络分析数据。

   1. 但是，需要确保网络分析数据适用于所使用的节点配置。

5. **CPU开销分析**：这包括实现开销，如调度时间、采样时间、去标记化等。为了更好的保真度，这些也应该被分析，但它们将模拟器与实现紧密绑定，例如`vLLM`。相关脚本可用，但尚未文档化。

综上所述，在Vidur中，"profiling"是一个涉及对LLM操作符运行时间特征的收集和分析的过程，目的是为了模拟和优化LLM的并行策略和性能

### Profelling

在LLM的推理过程中，prefilling是指在自回归生成之前，对输入提示（prompt）中的令牌（tokens）**计算键值（key-value，简称KV）缓存的过程**。这个过程的目的是为了在生成下一个令牌时避免重复计算，通过预先填充KV缓存来加速推理过程。

具体来说，prefilling阶段会处理整个用户输入的提示，并产生第一个输出令牌。在随后的解码（decode）阶段，会逐个生成输出令牌，直到生成一个特殊的序列结束令牌，此时请求处理完成。在解码过程中，需要访问之前处理过的令牌的键和值激活，以执行注意力操作。为了避免重复计算，现代LLM推理系统会将它们存储在KV-Cache中。

Prefilling的主要作用包括：

1. **减少重复计算**：通过预先计算并存储KV缓存，避免了在生成每个新令牌时重复计算的需要。
2. **提高推理效率**：Prefilling有助于提高LLM推理的速度和效率，尤其是在处理长输入提示时。
3. **优化资源分配**：在处理不同长度的输入提示时，Prefilling可以通过优化计算资源的分配来提高整体的推理性能。

在实际应用中，Prefilling阶段的计算效率较高，因为数据量较大，更容易遇到计算瓶颈。因此，针对Prefilling的优化方向主要是算子合并、简化等，以降低模型计算量。总的来说，Prefilling是LLM推理中一个关键的优化阶段，对于提高模型的响应速度和减少计算资源占用具有重要意义。

### replica

在大型语言模型（LLM）的训练中，“replica”通常指的是模型的一个副本或复制品。在分布式训练或模型部署时，可能会创建模型的多个副本（replicas），以便于并行处理或负载均衡。

#### 训练过程

训练过程中，每个replica都**有完整的参数，但是对不同的数据集**进行训练。

参数可以独立更新，或者在某些情况下，通过同步机制来保持一致性。这样做可以提高训练效率和模型服务的可用性。

确保各个副本（replicas）之间的一致性以及最终训练得到统一的模型，主要依赖于以下几个关键技术和策略：

1. **数据并行性**：在数据并行训练中，数据集被分割成多个分片，每个分片分配给不同的设备。每个设备保存模型副本的完整副本，并在分配的数据集分片上进行训练。反向传播后，模型的梯度通过All-Reduce操作全部减少，以保持不同设备上的模型参数同步。
2. **全分片数据并行（FSDP）技术**：FSDP技术在数据并行worker中统一分片模型参数并训练数据，其中每个微批数据的计算均针对每个GPU worker进行本地计算。FSDP通过操作重排序和参数预取最大限度地减少气泡，以通过计算积极地重叠通信。
3. **同步训练算法**：同步训练算法如All-Reduce是确保各计算节点间梯度同步的重要手段。所有节点计算出本地梯度后，通过高效的通信协议（例如Ring All-Reduce）汇总所有节点的梯度信息，然后统一更新模型参数。
4. **参数服务器架构**：Parameter Server作为中心化的存储和协调器，负责维护和更新模型参数。各个计算节点异步地从参数服务器读取参数，计算局部梯度，再将梯度发送回参数服务器进行更新。
5. **异步训练与优化策略**：异步训练允许不同节点根据自己的进度更新全局模型，但可能导致不稳定的收敛性和一致性问题。实践中，研究者尝试通过控制更新频率、优化通信策略以及使用延迟补偿等方法平衡效率与稳定性。
6. **超大规模LLM中的通信开销与梯度一致性解决方案**：为降低通信开销，可采取梯度压缩、稀疏通信、选择性通信等策略。同时，为了保证梯度一致性，还引入了诸如同步屏障、动态调整学习率等算法和技术。
7. **零冗余优化器（ZeRO）**：ZeRO通过在所有GPU上完全分片模型状态来优化内存冗余。在训练过程中，每个GPU进行独立的前向和后向传播来计算梯度，然后使用ReduceScatter操作在数据并行组内的所有GPU之间同步梯度。每个GPU负责更新特定部分的模型参数。随后，更新后的模型参数片段从其他GPU上收集，使用AllGather操作，确保所有GPU都有最新的模型参数。

通过上述技术和策略，分布式训练过程中的各个模型副本能够保持一致性，并最终训练得到一个统一的模型。

#### 推理过程

replica在推理过程中起到了提高效率、可用性、吞吐量和资源共享的作用。

1. **分布式推理**：在分布式计算环境中，"replica"指的是模型的不同副本
   1. 它们可以**分布在不同的计算节点上。这样可以并行处理多个推理请求**，提高模型的响应速度和吞吐量
   2. 每个副本可以独立地处理一部分任务，从而实现负载均衡和提高效率。
2. **模型可用性**：在推理过程中，"replica"可以增加模型的**可用性和容错能力**。如果一个副本出现问题或者需要维护，其他副本可以继续提供服务，确保不间断的推理能力。
3. **提高吞吐量**：在某些情况下，"replica"可以类似于数据库中的分区（partition），每个分区存储一部分数据。这样可以提高消息写入的吞吐量，并且可以提高消费者消费消息的并发量。在LLM推理中，这意味着可以同时处理更多的请求，提高整体的处理能力。
4. **资源共享**：有些项目，如Petals，通过分布式计算的方式运行大型语言模型，采用类似BitTorrent的方式，将模型的不同部分分布在多个用户的设备上，实现高效的推理和微调。这种方式依赖于社区用户共享GPU资源，用户可以贡献自己的GPU来增加计算能力，这里的"replica"就是指分布在不同设备上的模型副本。
5. **API接口和模型管理**：在API接口中，"replica"的数量可以作为参数之一，用于指定模型副本的数量。这有助于在不同的负载和需求下调整资源分配，优化性能。





# 代码阅读

## 开始

### 环境准备

使用conda/mamba配置本地的包环境，没有集中安装到conda的env_list中

```shell
mamba env create -p ./env -f ./environment.yml
# 选择本地的环境进行代码运行
conda activate ./env
```

### 运行

整个vidur被封装为了一个python包，以包为整体对象进行运行

```shell
# 不使用参数
python -m vidur.main

# 使用完整参数
python -m vidur.main  \
--replica_config_device a100 \
--replica_config_model_name meta-llama/Llama-2-7b-hf  \
--cluster_config_num_replicas 1 \
--replica_config_tensor_parallel_size 1 \
--replica_config_num_pipeline_stages 1 \
--request_generator_config_type synthetic \
--length_generator_config_type trace \
--interval_generator_config_type static \
--[trace|zipf|uniform|fixed]_request_length_generator_config_max_tokens 4096 \
--trace_request_length_generator_config_trace_file ./data/processed_traces/arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv \
--synthetic_request_generator_config_num_requests 128  \
--replica_scheduler_config_type vllm  \
--[vllm|lightllm|orca|faster_transformer|sarathi]_scheduler_config_batch_size_cap 256  \
--[vllm|lightllm]_scheduler_config_max_tokens_in_batch 4096
```

#### 参数含义

1. **Number of replicas**：这个参数指的是副本的数量，也就是你希望运行的服务实例的数量。在Vidur中，这可能涉及到模型或服务的并行部署，以提供高可用性和负载均衡。
2. **TTFT (Time-to-First-Token)**：这是从请求到达系统到生成第一个输出标记（token）的时间。它是衡量系统响应速度的一个重要指标。
3. **TBT (Time-Between-Tokens)**：这是用户观察到的标记之间的延迟，即系统生成连续输出标记之间的时间间隔。
4. **TP (Tensor Parallelism)**：指的是张量并行，是一种模型并行技术，用于在多个GPU上分布模型的不同部分以加速训练或推理。
5. **PP (Pipeline Parallelism)**：指的是流水线并行，是一种并行技术，用于将模型的不同层分配到不同的GPU上，以提高模型的吞吐量。
6. **Request E2E Time**：这是请求的端到端时间，即从请求到达系统到请求完全处理完成的时间。
7. **Batch Size**：这是每个批次处理的请求数量。较大的批次大小通常意味着更高的吞吐量。
8. **Prefill Tokens**：这是请求中用于填充的标记数量，是模型推理过程中的一部分。
9. **Decode Tokens**：这是请求中用于解码的标记数量，也是模型推理过程中的一部分。
10. **P:D Ratio (Prefill to Decode Ratio)**：这是填充标记与解码标记的比例，它可以帮助理解工作负载的特性。

## 层次结构

阅读代码可以得到如下的层次结构：

- Cluster：代表了对最顶层训练集群的抽象，包含了模型、设备、请求。是最顶层的抽象

  - replica：分布式推理过程会有多个推理实体，可以并行地响应request，提高响应速度和吞吐量

    - 每个replica中的模型都是相同的，共享一套参数

  - schedule：调度策略

    - global：对外部（也即顶层）的request进行调度。经典的调度算法与round-robin

    - replica：在每个replica的层次上进行调度，包括batching策略和内存管理策略

      - 通过模型的信息和并行策略计算得到可用于KV-cache的内存
      - 在计算得到内存信息后，通过batching策略的API接口进行调度，如vllm

      > 在需要高吞吐量和高内存效率的情况下，直接运行LLM可能会面临性能瓶颈
      >
      > 相应的batching策略通过**优化内存管理和计算效率**，使得LLM在这些场景下的性能得到显著提升

    - replica_stage：最底层的调度策略：在流水线层级的微批调度

### 策略类

策略类都继承了同一个类`BasePolyConfig`











## 训练数据

训练数据在`data/profiling`下，包含了两部分数据：

- compute的数据：运行时（推理）的相关数据
- network的数据：网络分析不依赖于模型，因此可以为所有模型使用相同的网络分析数据。

## 可视化结果

在simulate的过程中，支持两种trace：

- **event_trace**：通常用于记录特定事件的详细信息，包括事件的开始和结束时间、持续时间、线程ID等。
  - 这种追踪方式更关注于细粒度的事件记录，适合于分析系统内部的操作和性能瓶颈
  - 为json格式
- **chrome_trace**：是Vidur导出的模拟结果，采用Chrome的追踪格式。
  - 它可以通过`chrome://tracing/`或`edge://tracing/`进行可视化，主要用于展示模拟过程中的整体性能和事件流。
  - Chrome Trace提供了一个用户友好的界面，便于查看和分析事件的时间线和性能数据
  - 为Chrome Trace格式































