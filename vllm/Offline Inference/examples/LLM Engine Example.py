# SPDX-License-Identifier: Apache-2.0
"""
This file demonstrates using the `LLMEngine`
for processing prompts with various sampling parameters.
"""

import argparse

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils import FlexibleArgumentParser


def create_test_prompts() -> list[tuple[str, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters."""
    return [
        (
            "A robot may not injure a human being",
            SamplingParams(temperature=0.0, logprobs=1, prompt_logprobs=1),
        ),
        (
            "To be or not to be,",
            SamplingParams(temperature=0.8, top_k=5, presence_penalty=0.2),
        ),
        (
            "What is the meaning of life?",
            SamplingParams(n=2, temperature=0.8, top_p=0.95, frequency_penalty=0.1),
        ),
    ]


def process_requests(engine: LLMEngine, test_prompts: list[tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0

    print("-" * 50)
    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1

        request_outputs: list[RequestOutput] = engine.step() # 每次生成一个token

        for request_output in request_outputs:
            # 当前请求完成，输出结果
            if request_output.finished:
                print(request_output)
                print("-" * 50)


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""

    # Namespace() -> 创建命名空间对象。
    #   1. 命名空间对象是一种轻量级的容器，用于存储变量和属性。
    #   2. 它类似于字典，但可以使用点操作符来访问和设置属性

    # 修改模型路径: facebook/opt-125m -> 本地 ./models
    args.model = "./models"  # 修改模型路径为本地 ./models

    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using the LLMEngine class directly"
    )

    # add_cli_args 创建新 ModelConfig 默认对象(修改不了) -> add_argument -> parse_args
    parser = EngineArgs().add_cli_args(parser) 

    return parser.parse_args() # 设置默认参数


def main(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine(args)
    test_prompts = create_test_prompts()
    process_requests(engine, test_prompts)


if __name__ == "__main__":
    args = parse_args()
    main(args)