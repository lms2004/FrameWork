# VLLM 学习路线图（完整版）

## 📌 阶段 1：核心架构基础
| 顺序 | 示例名称              | 学习目标                          | 关键源码文件                     |
|------|-----------------------|-----------------------------------|----------------------------------|
| 1    | `Basic`              | 掌握离线推理全流程                | `vllm/engine/llm_engine.py`      |
| 2    | `LLM Engine Example` | 深入理解引擎工作机制             | `vllm/engine/async_llm_engine.py`|
| 3    | `Batch LLM Inference`| 批处理调度机制                   | `vllm/core/scheduler.py`         |
| 4    | `Torchrun Example`   | 分布式环境初始化                 | `vllm/engine/arg_utils.py`       |
| 5    | `Data Parallel`      | ✨ 数据并行实现原理              | `vllm/executor/ray_gpu_executor.py` |

## ⚡ 阶段 2：性能优化技术
| 顺序 | 示例名称                      | 学习目标                          | 关键源码文件                     |
|------|-------------------------------|-----------------------------------|----------------------------------|
| 6    | `Prefix Caching`              | KV缓存重用机制                  | `vllm/worker/cache_engine.py`    |
| 7    | `Automatic Prefix Caching`    | 自动前缀检测实现                | `vllm/core/block_manager.py`     |
| 8    | `Eagle`                       | 推测解码加速原理                | `vllm/spec_decode/` 目录         |
| 9    | `MLPSpeculator`               | ✨ 草稿模型实现细节             | `vllm/spec_decode/mlp_speculator.py` |
| 10   | `Disaggregated Prefill`       | 预填充与解码分离策略            | `vllm/core/scheduler.py`         |
| 11   | `Disaggregated-Prefill-V1`    | ✨ 分离式预填充优化方案         | `vllm/core/disaggregated_scheduler.py` |

## 🧩 阶段 3：模型与硬件适配
| 顺序 | 示例名称                      | 学习目标                          | 关键源码文件                     |
|------|-------------------------------|-----------------------------------|----------------------------------|
| 12   | `LoRA With Quantization`      | 量化+LoRA联合应用               | `vllm/model_executor/layers/quantization/` |
| 13   | `Neuron INT8 Quantization`    | AWS Neuron量化适配              | `vllm/executor/neuron_executor.py` |
| 14   | `Neuron Speculation`          | ✨ Neuron推测执行优化           | `vllm/spec_decode/neuron_worker.py` |
| 15   | `TPU`                         | Google TPU适配                  | `vllm/executor/tpu_executor.py`  |
| 16   | `Profiling TPU`               | ✨ TPU性能分析工具              | `vllm/utils/tpu_profiler.py`     |

## 🌐 阶段 4：多模态与高级应用
| 顺序 | 示例名称                      | 学习目标                          | 关键源码文件                     |
|------|-------------------------------|-----------------------------------|----------------------------------|
| 17   | `Vision Language`             | 基础图文多模态处理              | `vllm/model_executor/models/llava.py` |
| 18   | `Vision Language Multi Image` | ✨ 多图输入处理                 | `vllm/multimodal/prompter.py`    |
| 19   | `Vision Language Embedding`   | ✨ 多模态嵌入提取               | `vllm/multimodal/embedding.py`   |
| 20   | `Audio Language`              | ✨ 音频语言多模态支持           | `vllm/multimodal/audio.py`       |
| 21   | `Encoder Decoder`             | ✨ 编码器-解码器架构支持        | `vllm/model_executor/input_metadata.py` |
| 22   | `Encoder Decoder Multimodal`  | ✨ 多模态编解码架构             | `vllm/multimodal/encoder_decoder.py` |
| 23   | `Chat With Tools`             | 函数调用实现机制                | `vllm/model_executor/guided_decoding.py` |
| 24   | `Structured Outputs`          | 结构化输出控制                  | `vllm/model_executor/guided_decoding.py` |
| 25   | `Prompt Embed Inference`      | ✨ 提示嵌入优化技术             | `vllm/prompt_adapter/` 目录      |
| 26   | `Embed Jina Embeddings V3`    | ✨ Jina嵌入集成                | `vllm/entrypoints/openai/jina.py` |
| 27   | `Embed Matryoshka Fy`         | ✨ 嵌套嵌入技术                | `vllm/model_executor/embeddings.py` |

## 🚀 阶段 5：生产部署与优化
| 顺序 | 示例名称                      | 学习目标                          | 关键源码文件                     |
|------|-------------------------------|-----------------------------------|----------------------------------|
| 28   | `Save Sharded State`          | 分布式模型保存策略               | `vllm/worker/worker.py`          |
| 29   | `Load Sharded State`          | 分布式模型加载实现               | `vllm/worker/worker.py`          |
| 30   | `Profiling`                   | 性能监控系统设计                 | `vllm/engine/metrics.py`         |
| 31   | `Simple Profiling`            | ✨ 轻量级性能分析工具           | `vllm/utils/simple_profiler.py`  |
| 32   | `Metrics`                     | 运行时指标收集与分析             | `vllm/engine/stats.py`           |
| 33   | `OpenAI Batch`                | ✨ OpenAI兼容批处理API          | `vllm/entrypoints/openai/api_server.py` |
| 34   | `Reproducibility`             | ✨ 可复现性保障机制             | `vllm/config.py` (随机种子控制)  |

## 🧠 高阶专项
| 顺序 | 示例名称                      | 学习目标                          | 关键源码文件                     |
|------|-------------------------------|-----------------------------------|----------------------------------|
| 35   | `Qwen 1M`                    | 百万token上下文管理              | `vllm/attention/backends/paged_attention.py` |
| 36   | `Qwen2 5 Omni`               | ✨ 通义多模态模型支持           | `vllm/model_executor/models/qwen.py` |
| 37   | `MultiLoRA Inference`        | 动态LoRA切换策略                 | `vllm/lora/worker_manager.py`    |
| 38   | `RLHF`                       | 奖励模型基础集成                 | `vllm/rlhf/` 目录               |
| 39   | `RLHF Colocate`              | ✨ RLHF协同部署优化             | `vllm/rlhf/colocate.py`          |
| 40   | `RLHF Utils`                 | ✨ RLHF工具函数集               | `vllm/rlhf/utils.py`             |
| 41   | `Prithvi Geospatial MAE`     | ✨ 地理空间多模态模型           | `vllm/multimodal/geospatial.py`  |
| 42   | `Neuron Eagle`               | ✨ Neuron+推测解码融合          | `vllm/spec_decode/neuron_eagle.py` |
| 43   | `Mistral-Small`              | ✨ Mistral模型优化实践          | `vllm/model_executor/models/mistral.py` |

> ✅ 已覆盖全部43个示例
> 💡 **学习建议**：重点关注三大核心机制的交互相：
> 1. **PagedAttention** (`vllm/core/block_manager.py`)
> 2. **Continuous Batching** (`vllm/core/scheduler.py`)
> 3. **Decoding Engine** (`vllm/engine/llm_engine.py`)