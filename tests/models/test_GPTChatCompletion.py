import os

import unittest

from unittest.mock import patch

from arabic_llm_benchmark import Benchmark
from arabic_llm_benchmark.models import GPTChatCompletionModel


class TestAssetsForGPTChatCompletionPrompts(unittest.TestCase):
    @classmethod
    @patch("os.environ")
    def setUpClass(cls, os_env_mock):
        # Handle environment variables required at runtime
        os_env_mock.__getitem__.side_effect = lambda x: "test_str"

        # Load the benchmark assets
        benchmark = Benchmark(benchmark_dir="assets")
        all_assets = benchmark.find_runs()

        # Filter out assets not using the GPT model
        cls.assets = [
            asset
            for asset in all_assets
            if asset["module"].config()["model"] in [GPTChatCompletionModel]
        ]

    @patch("os.environ")
    def test_gpt_prompts(self, os_env_mock):
        "Test if all assets using this model return data in an appropriate format for prompting"
        os_env_mock.__getitem__.side_effect = lambda x: "test_str"
        for asset in self.assets:
            with self.subTest(msg=asset["name"]):
                config = asset["module"].config()
                dataset = config["dataset"](**config["dataset_args"])
                data_sample = dataset.get_data_sample()
                prompt = asset["module"].prompt(data_sample["input"])

                self.assertIsInstance(prompt, list)

                for message in prompt:
                    self.assertIsInstance(message, dict)
                    self.assertIn("role", message)
                    self.assertIsInstance(message["role"], str)
                    self.assertIn("content", message)
                    self.assertIsInstance(message["content"], str)