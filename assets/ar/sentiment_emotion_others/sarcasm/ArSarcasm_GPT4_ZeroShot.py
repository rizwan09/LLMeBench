from llmebench.datasets import ArSarcasmDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import SarcasmTask


def config():
    return {
        "dataset": ArSarcasmDataset,
        "dataset_args": {},
        "task": SarcasmTask,
        "task_args": {},
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["TRUE", "FALSE"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "## INSTRUCTION\nYou are an expert in sarcasm detection.\n\n",
        },
        {
            "role": "user",
            "content": 'You are an AI assistant, an expert at detecting sarcasm in text. Say yes if the tweet is sarcastic and say no if the tweet is not sarcastic: "'
            + input_sample
            + '"',
        },
    ]


def post_process(response):
    content = response["choices"][0]["message"]["content"].lower()

    if (
        content.startswith("no")
        or "\nNo" in content
        or "tweet is not sarcastic" in content
        or "answer is no" in content
        or "would say no" in content
    ):
        return "FALSE"
    elif content == "yes" or content == "نعم":
        return "TRUE"
    else:
        return None
