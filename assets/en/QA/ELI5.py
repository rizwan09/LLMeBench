from llmebench.datasets import HuggingFaceDataset, DPRDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import QATask
import json 

def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-35-turbo (version 0301)",
        "description": "GPT35 model hosted on Azure, using the Completion API. API version '2023-03-15-preview'.",
        "scores": {"EM": "--"},
    }


def config():
    return {
        "dataset": DPRDataset,
        "dataset_args": {
            "huggingface_dataset_name": "eli5",
            "sub_split": "rc.wikipedia",
            "column_mapping": {
                "input": "question",
                "label": "answer",
                "multi_column":True,
            },
        },
        "task": QATask,
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
        "general_args": {"custom_test_split": "validation"},
    }

def prompt(data_row):
    return data_row

    
  
    

def post_process(response):
    if not response:
        return None
    raw_answer = json.loads(response["choices"][0]["message"]["content"])['answer']

    return raw_answer
