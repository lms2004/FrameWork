# SPDX-License-Identifier: Apache-2.0
"""
This example shows how to use Ray Data for data parallel batch inference.

Ray Data is a data processing framework that can handle large datasets
and integrates tightly with vLLM for data-parallel inference.

As of Ray 2.44, Ray Data has a native integration with
vLLM (under ray.data.llm).

Ray Data provides functionality for:
* Reading and writing to cloud storage (S3, GCS, etc.)
* Automatic sharding and load-balancing across a cluster
* Optimized configuration of vLLM using continuous batching
* Compatible with tensor/pipeline parallel inference as well.

Learn more about Ray Data's LLM integration:
https://docs.ray.io/en/latest/data/working-with-llms.html
"""

import ray
from packaging.version import Version
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig

assert Version(ray.__version__) >= Version("2.44.1"), (
    "Ray version must be at least 2.44.1"
)

# Uncomment to reduce clutter in stdout
# ray.init(log_to_driver=False)
# ray.data.DataContext.get_current().enable_progress_bars = False

""""
read_text
    -> 从文本文件(或文件列表）逐行读取数据，创建分布式数据集（Dataset）
    示例：
        # 读取远程文件
        ds = ray.data.read_text("s3://anonymous@ray-example-data/this.txt")
        # 读取多个本地文件
        ds = ray.data.read_text(["local:///path/to/file1", "local:///path/to/file2"])
    return Dataset
"""
# 优化后的读取方式（添加关键参数）
ds = ray.data.read_text(
    "local://data/api.txt",
    include_paths=True,          # 添加文件来源路径列
    drop_empty_lines=False,      # 保留空行（根据需求调整）
    encoding="utf-8",            # 显式指定编码（默认就是utf-8）
    ignore_missing_paths=True    # 避免路径不存在时报错
)

"""
vLLMEngineProcessorConfig
    配置 vLLM 引擎处理器, 用于在Ray Data中集成vLLM引擎进行大规模语言模型推理
    -> 支持：
        1. 分布式批处理推理
        2. 聊天模板应用
        3. 动态分词/反分词
        4. 多模态图像处理（需设置 has_image=True)
"""
config = vLLMEngineProcessorConfig(
    model_source="./models",    # 模型来源
    engine_kwargs={
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 4096,
        "max_model_len": 16384,
    },               # vllm 引擎核心参数
    concurrency=1,   # 数据并行的工作进程数
    batch_size=64,   # 发送到vLLM的批次大小
)

"""
build_llm_processor 
    用于构建 ​​分布式 LLM 推理处理器​​，将预处理、模型推理、后处理集成到 Ray Data 流水线中
    -> 支持：
        1. ​​分布式批处理​​：自动并行化大规模数据推理
        2. ​动态参数配置​​：支持逐行定制采样参数（如温度、最大 token 数）
        3. 灵活数据处理​​：通过预处理/后处理函数定制输入输出格式
        4. ​​无缝集成​​：与 Ray Data 的 map()、iter_torch_batches() 等方法协同工作
    
    return type(Processor)  
"""
vllm_processor = build_llm_processor(
    config,
    preprocess=lambda row: dict(
        messages=[
            {"role": "system", "content": "ray.data.read_text 怎么使用？有哪些参数？"},
            {"role": "user", "content": row["text"]},
        ],
        sampling_params=dict(
            temperature=0.3,
            max_tokens=250,
        ),
    ), # 预处理函数，接受单行数据 → 返回引擎所需格式

    postprocess=lambda row: dict(
        answer=row["generated_text"],
        **row,  # This will return all the original columns in the dataset.
    ), # 后处理函数，解析模型输出 → 返回最终结果
)

"""
Processor
-> 是Ray Data中处理LLM任务的核心组件，采用 ​预处理 → 多阶段处理 → 后处理​ ​的流水线架构。

def __call__(self, dataset: Dataset) -> Dataset:
    Execute the processor:
        preprocess -> stages -> postprocess.
        Note that the dataset won't be materialized during the execution.

        Args:
            dataset: The input dataset.

        Returns:
            The output dataset.
"""
ds = vllm_processor(ds)

# Peek first 10 results.
# NOTE: This is for local testing and debugging. For production use case,
# one should write full result out as shown below.
outputs = ds.take(limit=10)

for output in outputs:
    prompt = output["prompt"]
    generated_text = output["generated_text"]
    print(f"Prompt: {prompt!r}")
    print(f"Generated text: {generated_text!r}")

# Write inference output data out as Parquet files to S3.
# Multiple files would be written to the output destination,
# and each task would write one or more files separately.
#
# ds.write_parquet("s3://<your-output-bucket>")