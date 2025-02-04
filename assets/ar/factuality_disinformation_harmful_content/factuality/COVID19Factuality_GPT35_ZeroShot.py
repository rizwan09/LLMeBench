from llmebench.datasets import COVID19FactualityDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import FactualityTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-35-turbo (version 0301)",
        "description": "GPT35 model hosted on Azure, using the Completion API. API version '2023-03-15-preview'.",
        "scores": {"Weighted-F1": "0.103"},
    }


def config():
    return {
        "dataset": COVID19FactualityDataset,
        "task": FactualityTask,
        "model": LegacyOpenAIModel,
        "model_args": {
            "class_labels": ["yes", "no"],
            "max_tries": 30,
        },
    }


def prompt(input_sample):
    prompt_string = (
        f"Detect the information in the sentence as correct or incorrect. Use label as yes or no.\n\n"
        f"text: {input_sample}\n"
        f"label: \n"
    )
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": prompt_string,
            }
        ],
    }


def post_process(response):
    label = response["choices"][0]["text"].lower().replace(".", "").lower()

    if label.startswith("I am unable to verify".lower()) or label.startswith(
        "I am unable to categorize".lower()
    ):
        label_fixed = None
    elif "incorrect" in label or "label: no" in label:
        label_fixed = "no"
    elif "correct" in label or "label: yes" in label:
        label_fixed = "yes"
    elif "no" == label or "yes" == label:
        label_fixed = label
    else:
        label_fixed = None

    return label_fixed
