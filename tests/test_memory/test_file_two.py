import unittest
import weakref

import pytest
import torch
from parameterized import parameterized_class
from transformers import AutoModelForCausalLM

from llmcompressor.utils.metric_logging import get_GPU_usage_nv


@pytest.mark.integration
@parameterized_class(
    [
        {"model": "meta-llama/Llama-2-7b-hf"},
        {"model": "meta-llama/Llama-2-7b-hf"},
    ]
)
class TestSparsitiesGPU(unittest.TestCase):
    model = None

    def setUp(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model, device_map="cuda:0", torch_dtype=torch.bfloat16
        )
        weakref.finalize(self.model, lambda: print("finalized"))

    def test_memory(self):
        print(f"allocated: {torch.cuda.memory_allocated(0)}")
        print(f"reserved: {torch.cuda.memory_reserved(0)}")
        print(get_GPU_usage_nv())

    def tearDown(self):
        # torch.cuda.empty_cache()
        pass


# def test_one():
#     model = AutoModelForCausalLM.from_pretrained(
#         "meta-llama/Llama-2-7b-hf", device_map="cuda:0", torch_dtype=torch.bfloat16
#     )
#     weakref.finalize(model, lambda: print("finalized"))

#     print(f"allocated: {torch.cuda.memory_allocated(0)}")
#     print(f"reserved: {torch.cuda.memory_reserved(0)}")
#     print(get_GPU_usage_nv())

# def test_two():
#     model = AutoModelForCausalLM.from_pretrained(
#         "meta-llama/Llama-2-7b-hf", device_map="cuda:0", torch_dtype=torch.bfloat16
#     )
#     weakref.finalize(model, lambda: print("finalized"))

#     print(f"allocated: {torch.cuda.memory_allocated(0)}")
#     print(f"reserved: {torch.cuda.memory_reserved(0)}")
#     print(get_GPU_usage_nv())
