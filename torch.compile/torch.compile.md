# Ref
1. [论文](https://docs.pytorch.org/assets/pytorch2-2.pdf?utm_source=chatgpt.com)
2. [源码](https://github.com/pytorch/pytorch?utm_source=chatgpt.com)
3. [deepwiki](https://deepwiki.com/pytorch/pytorch)
4. [目前的torch.compile](https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/?utm_source=chatgpt.com)
5. [关于Lazy evaluation](https://zhuanlan.zhihu.com/p/53750888)
6. [用CUDAGraph加速Pytorch](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/?utm_source=chatgpt.com)
7. [关于vmap函数](https://zhuanlan.zhihu.com/p/567662038)
8. [关于dispatcher](https://azjf.github.io/pytorch_dispatcher.html)

# [学习代码](https://github.com/Chosen-David/LLM-Inference-Notes)
本部分的学习代码位于/LLM-Inference-Notes/torch.compile/tutorial/下，主要包括：
## Dispatcher_Tutorial：Pytorch中Dispatcner的教程

包括：
Dispatcher_Tutorial.ipynb：浅浅解析Dispatcher和用法
vmap_source_explained.ipynb：解读Pytorch中的vmap源码,并手写了一个mini-vmap



# 论文阅读

## Abstruct

本文介绍了两个对流行的 PyTorch 机器学习框架的扩展：TorchDynamo 和 TorchInductor，它们实现了在 PyTorch 2 中发布的 torch.compile 功能

其中，TorchDynamo 是一个 Python 层级的即时编译器（JIT），它在不牺牲 Python 灵活性的前提下，使得 PyTorch 程序能够进行图编译。更具体的，它在 CPython 中钩挂（hook） Python 的 frame evaluation API [9]，在 Python 字节码执行之前动态地修改这些字节码。

TorchInductor 是 TorchDynamo 的默认编译后端，它会将 PyTorch 程序转换为 OpenAI 的 Triton（用于 GPU）以及 C++（用于 CPU）。它引入了一种新的 define-by-run（按运行时定义）循环层级的中间表示（IR），以便于添加新的操作符（operator）降级／转换（lowerings）。

## Intro

现代的机器学习框架可以分为两类：eager 模式框架（如 PyTorch [32] 和 JAX [8]），以及 graph 模式框架（如 TensorFlow [1]、Caffe [25]、Theano [5] 和 CNTK [37]）。

Eager 模式框架采用的是命令式的 define-by-run（边定义边执行） 方法，即机器学习模型直接表现为代码，每次需要运行模型时，代码都会立即被执行。

Graph 模式框架则采用更声明式的 define-and-run（先定义后执行） 方法，用户需要先通过图构建 API 搭建一个计算图，然后再运行该图。

### 为什么是eager模式


这里你可能奇怪，torch.compile貌似也是先生成FX Graph这种图然后给后端进行CodeGen的，为什么说是eager模式的呢？

我认为是因为Graph capture 阶段是动态／运行时触发的，不是静态要求用户先写计算图。这个算是二者优点的一个结合。

在 eager 模式下，框架一次只能“看到”一个算子（operator），因此它不能自动执行那种跨算子边界的优化，比如算子融合（fusion）或算子调度（scheduling）。

### 关于lazy eval
简而言之就是初始化的部分等到真的要调用的时候再用。在这个论文里面是一种前人的解决方法，就是等算子真正调用的时候再被临时分配和调度。不过延迟管理本身有开销，并且会牺牲python即时执行的特性。

值得注意的是，PyTorch/XLA（TPU 等后端）：XLA tensors 依然是 lazy。也即这种方式仍然在pytorch中有应用的场景。

可能是由于XLA/TPU 是 Google／OpenXLA 的一个非常成熟的编译器基础设施。有能力把多个算子融合、重排、选择最优 kernel／layout，从硬件层面得到很高效的图级优化。要能这么做，就得先“看到图”——lazy 模型正好能聚集算子序列／控制流。因此比较适合。

### 关于record/replay 
在程序运行时，对某些操作做监视／捕捉。比如跟踪每一步算子调用、输入输出、控制流路径、参数等。可以把这段过程看做一个 trace 或图（graph）。可能是整个模型的一次前向加 backward，也可能是部分网络。使用已经记录下来的操作或图结构，再次执行这些操作。这次重放通常跳过记录开销（例如跳过 Python 层的大量调度／检查／解释等），直接以一种更高效的方式执行，例如一次性把 kernel 调用提交给 GPU、减少 CPU−GPU 同步、减少中间张量分配等。

> “Record/replay is unsound and can produce incorrect behavior

然而，我们的作者认为这样是不安全可靠的。它在静态或“近似静态”的情形下效果最好，不过python主要是动态场景。

不过值得注意的是，我发现pytorch中CUDA Graphs 功能似乎用到了这种record/replay的方法。先把一段 CUDA 工作**record/capture**成一张图，然后在后续迭代里**replay**整张图，以减少 Python/C++/驱动层的调度开销。PyTorch 提供了 torch.cuda.CUDAGraph、torch.cuda.graph(...) 上下文等 API 来做这一套捕获→重放的流程


关于这点，我认为是由于这个和CUDAGraph不是一个场景，而是一种“动态建图和优化”的场景。因此不适用于这种偏静态的方法

## Pytorch先前图捕获的工作

### torch.jit.trace
使用“例子输入（example inputs）+ 记录／重放（record/replay）”的方法来生成一个 TorchScript 图。在于PyTorch 的 dispatcher 层

# 源码阅读