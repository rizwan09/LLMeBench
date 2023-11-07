from llmebench.datasets import HuggingFaceDataset
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
        "dataset": HuggingFaceDataset,
        "dataset_args": {
            "huggingface_dataset_name": "hotpot_qa",
            "sub_split": "distractor",
            
            "column_mapping": {
                "input": "question",
                "label": "answer",
                "input_id": "idx",
            },
            "multi_column_data": True,
        },
        "task": QATask,
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
        "general_args": {"custom_test_split": "validation"},
    }


def prompt(data_row):

    id = data_row["id"]
    question = data_row["question"]
    answer = data_row["answer"]
    supporting_facts = data_row["supporting_facts"]
    contexts = data_row["context"]["sentences"]


    paragraphs = [''.join(docs) for docs in contexts]
    
    prompt_string = (
        f"Question:{question}\nContext:{paragraphs}"
        f"Output josn:\n\n"
    )

    system_string = (
        f"You are a question answering agent. Given a context and a question, your task is to answer the question based on the context." 
        f"Generate the answer in a json output format with 'answer' tag"
        f"Instead of a full sentence, your answer must be the shortest word or phrase or named enitity."
        f" Some example outputs are: yes; no; Ibn Sina; Doha, Qatar; 2,132 seats, Los Angeles, California etc.,.\n\n " 
    )
    return [
        {
            "role": "system",
            "content": system_string,
        },
        {"role": "user", "content": prompt_string},
    ]

    
  
    

def post_process(response):
    if not response:
        return None
    raw_answer = json.loads(response["choices"][0]["message"]["content"])['answer']

    return raw_answer
