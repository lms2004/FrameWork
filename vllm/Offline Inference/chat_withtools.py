# SPDX-License-Identifier: Apache-2.0

# ruff: noqa
import json
import random
import string

from vllm import LLM
from vllm.sampling_params import SamplingParams

# This script is an offline demo for function calling
#
# If you want to run a server/client setup, please follow this code:
#
# - Server:
#
# ```bash
# vllm serve mistralai/Mistral-7B-Instruct-v0.3 --tokenizer-mode mistral --load-format mistral --config-format mistral
# ```
#
# - Client:
#
# ```bash
# curl --location 'http://<your-node-url>:8000/v1/chat/completions' \
# --header 'Content-Type: application/json' \
# --header 'Authorization: Bearer token' \
# --data '{
#     "model": "mistralai/Mistral-7B-Instruct-v0.3"
#     "messages": [
#       {
#         "role": "user",
#         "content": [
#             {"type" : "text", "text": "Describe this image in detail please."},
#             {"type": "image_url", "image_url": {"url": "https://s3.amazonaws.com/cms.ipressroom.com/338/files/201808/5b894ee1a138352221103195_A680%7Ejogging-edit/A680%7Ejogging-edit_hero.jpg"}},
#             {"type" : "text", "text": "and this one as well. Answer in French."},
#             {"type": "image_url", "image_url": {"url": "https://www.wolframcloud.com/obj/resourcesystem/images/a0e/a0ee3983-46c6-4c92-b85d-059044639928/6af8cfb971db031b.png"}}
#         ]
#       }
#     ]
#   }'
# ```
#
# Usage:
#     python demo.py simple
#     python demo.py advanced

model_name = "/hy-tmp/FrameWork/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
# or switch to "mistralai/Mistral-Nemo-Instruct-2407"
# or "mistralai/Mistral-Large-Instruct-2407"
# or any other mistral model with function calling ability

sampling_params = SamplingParams(max_tokens=8192, temperature=0.0)

"""
LLM (use Interface) -> LLMEngine(developer Interface)
    一个用于​​文本生成​​的离线推理工具，集成了分词器、语言模型（支持分布式 GPU）和 KV 缓存管理
    Parms:
        1. model: The name or path of a HuggingFace Transformers model. (本地路径也行)
        2. tokenizer: The name or path of a HuggingFace Transformers tokenizer.
        3. tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
                if available, and "slow" will always use the slow tokenizer.
        4. max_seq_len_to_capture: Maximum sequence len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode. Additionally for encoder-decoder models, if the
            sequence length of the encoder input is larger than this, we fall
            back to the eager mode.
        5. tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        6  gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
            reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's
            throughput. However, if the value is too high, it may cause out-of-
            memory (OOM) errors. (需要微调 0.6 -> 0.7 跑通 DeepSeek-R1-0528-Qwen3-8B)
"""
llm = LLM(model=model_name, max_seq_len_to_capture=128,tensor_parallel_size=2, gpu_memory_utilization=0.8)


def generate_random_id(length=9):
    characters = string.ascii_letters + string.digits
    random_id = "".join(random.choice(characters) for _ in range(length))
    return random_id


# simulate an API that can be called
def get_current_weather(city: str, state: str, unit: "str"):
    return (
        f"The weather in {city}, {state} is 85 degrees {unit}. It is "
        "partly cloudly, with highs in the 90's."
    )


tool_functions = {"get_current_weather": get_current_weather}


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find the weather for, e.g. 'San Francisco'",
                    },
                    "state": {
                        "type": "string",
                        "description": "the two-letter abbreviation for the state that the city is"
                        " in, e.g. 'CA' which would mean 'California'",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit to fetch the temperature in",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "state", "unit"],
            },
        },
    }
]

messages = [
    {
        "role": "user",
        "content": "Can you tell me what the temperate will be in Dallas, in fahrenheit?",
    }
]

outputs = llm.chat(messages, sampling_params=sampling_params, tools=tools)
output = outputs[0].outputs[0].text.strip()

# append the assistant message
messages.append(
    {
        "role": "assistant",
        "content": output,
    }
)

"""
    实际测试 
        1. Ministral-3b-instruct 不满足 chat withtools 的要求，重复生成单词
        2. DeepSeek-R1-0528-Qwen3-8B ：虽然得到正确答案，但是不能按照要求输出 json 格式(没有 function call 能力)
"""
tool_calls = json.loads(output)
tool_answers = [
    tool_functions[call["name"]](**call["arguments"]) for call in tool_calls
]

# append the answer as a tool message and let the LLM give you an answer
messages.append(
    {
        "role": "tool",
        "content": "\n\n".join(tool_answers),
        "tool_call_id": generate_random_id(),
    }
)

outputs = llm.chat(messages, sampling_params, tools=tools)

print(outputs[0].outputs[0].text.strip())
# yields
#   'The weather in Dallas, TX is 85 degrees fahrenheit. '
#   'It is partly cloudly, with highs in the 90's.'