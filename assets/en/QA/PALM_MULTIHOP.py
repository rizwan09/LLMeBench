from llmebench.datasets import HuggingFaceDataset
from llmebench.models import FastChatModel
from llmebench.tasks import QATask
import json
import datasets
import re
import string
import sys
from collections import Counter

import pprint
import google.generativeai as palm
palm.configure(api_key="Your key")

dataset_name = "hotpot_qa"

def normalize_answer( s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return str(text).lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score( prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score( prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def evaluate(true_labels, predicted_labels):
    f1, exact_match, total = 0, 0, 0
    for ground_truth, prediction in zip(true_labels, predicted_labels):
        total += 1
        if prediction is None:
            continue                
        if isinstance(ground_truth, str):
            ground_truths = [ground_truth]
        else:
            ground_truths = ground_truth
        print("index: ", total-1)
        print("ground truth: ", ground_truths)
        print("predictions: ", prediction)
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths
        )
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths
        )
    print("Total: ", total)
    print("exact_match: ", exact_match)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {"exact_match": exact_match, "f1": f1}


def get_model():
    models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
    return models[0].name


def cot_prompt_hotpotqa_huggingface_palm_works(data_row):
    
    question = data_row["question"]
    
    contexts = data_row["context"]["sentences"]
    paragraphs = [''.join(docs) for docs in contexts]
    
    prompt_string = (
        f"Question: {question}\nContext: {paragraphs}"
        f"Output josn:\n\n"
    )

    prompt = f"""
    You are a question answering agent. Given a context and a question, your task is to answer the question based on the context. 
    Generate the answer in a json output format with 'answer' tag and an 'step_by_step_reasoning' tag 
    
    {prompt_string} 
    

    """
    
    return prompt

def cot_prompt_hotpotqa_huggingface_palm(data_row):
    
    question = data_row["question"]
    
    contexts = data_row["context"]["sentences"]
    paragraphs = [''.join(docs) for docs in contexts]
    
    prompt_string = (
        f"Question: {question}\nContext: {paragraphs}"
        f"Output josn:\n\n"
    )

    prompt = f"""
    You are a question answering agent. Given a context and a question, your task is to answer the question based on the context. 
    Think step by step and generate the answer in a json output format with 'answer' tag and 'step_by_step_reasoning' tag 
    Instead of a full sentence, your answer must be the shortest word or phrase or named enitity. 
    Some example outputs 'answer' are: yes; no; Ibn Sina; Doha, Qatar; 2,132 seats, Los Angeles, California etc.,. Please make sure it's valid json. 
          

    {prompt_string} 
    

    """
    
    return prompt



def e2g_prompt_hotpotqa_huggingface_palm_works(data_row):
    
    question = data_row["question"]
    
    contexts = data_row["context"]["sentences"]
    paragraphs = [''.join(docs) for docs in contexts]
    
    prompt_string = (
        f"Question: {question}\nContext: {paragraphs}"
        f"Output josn:\n\n"
    )

    prompt = f"""
    You are a question answering agent. Given a context and a question, your task is to answer the question based on the context. 
    Generate the answer in a json output format with 'answer' tag and an 'evidence_reasoning' tag 
    
    {prompt_string} 

    -----------
    Important: In evidence_reasoning tag, all your reasoning path with explicit evidence from the context. Don't assume facts in your head.   


    """
    
    return prompt

def e2g_prompt_hotpotqa_huggingface_palm(data_row):
    
    question = data_row["question"]
    
    contexts = data_row["context"]["sentences"]
    paragraphs = [''.join(docs) for docs in contexts]
    
    prompt_string = (
        f"Question: {question}\nContext: {paragraphs}"
        f"Output josn:\n\n"
    )

    prompt = f""" 
    You are a question answering agent. Given a context and a question, your task is to answer the question based on the context. 
    Generate the answer in a json output format with 'answer' tag and an 'evidence_and_explanation' tag 
    Instead of a full sentence, your answer must be the shortest word or phrase or named enitity. 
    Some example outputs 'answer' are: yes; no; Ibn Sina; Doha, Qatar; 2,132 seats, Los Angeles, California etc.,. Please make sure it's valid json.
          
    
    {prompt_string} 


    """
    
    return prompt

def palm_api(prompt, method="cot"):
    failed = False
    ans= ""
    evidence = ""
    if method=="e2g":
        sep = "evidence_and_explanation"
        stop_sequences=['<answer>', '<evidence_and_explanation>']
    elif method == "cot":
        sep = "step_by_step_reasoning"
        stop_sequences=['<answer>', '<step_by_step_reasoning>']
    completion = palm.generate_text(model=get_model(), prompt=prompt, stop_sequences=stop_sequences,temperature=0, max_output_tokens=800)
    if not completion.result:
        return ans, evidence, completion, True
    try:
        ans =  json.loads(completion.result)['answer'].strip("[").strip("]").strip(",").strip("{").strip("}").strip()
        evidence = str(json.loads(completion.result)[sep]).strip("[").strip("]").strip(",").strip("{").strip("}").strip()
    except:
        try: 
            ans, evidence = completion.result.split(sep, maxsplit=1)
            evidence = evidence.replace(sep, "").replace("[", "").replace("]", "").replace("{", "").replace("}", "").strip().strip(",")
        except:
            try: 
                if "answer" in completion.result and sep in completion.result:
                    answer, evidence = completion.result.split(sep, maxsplit = 1)
                    ans = answer.split("answer")
                
                else:
                    try:
                        # ans =completion.result.replace("\n","").replace("{", "").split('"answer":')[1].split('"step_by_step_reasoning":')[0].replace('"',"").strip(",").strip()
                        ans, evidence = completion.result.split(sep, maxsplit=1)
                        evidence = evidence.replace(sep, "").replace("[", "").replace("]", "").replace("{", "").replace("}", "").strip().strip(",")          
                    except:          
                        failed = True
            except: 
                pass
    if sep in ans: 
        evidence += ans.split(sep, maxsplit=1)[1]
        ans = ans.split(sep, maxsplit=1)[0]
   
    ans = ans.replace(":", "").replace('"', "").replace("[", "").replace("]", "").replace("{", "").replace("}", "").strip().strip(",").strip()
    if "answer" in ans: ans = ans.split("answer", maxsplit=1)[1].strip().strip(":")
    print(f"returning {ans} {evidence}\n")
    return ans, evidence, completion, failed

def main_cot(palm_not_processed_index=None, limit=None):
    print("loading data")
    dataset = datasets.load_dataset(dataset_name, "distractor", split="validation", cache_dir="./data")
    print("data loading done")
    true_labels = []
    pred_labels = []
    total = 0
    failed = 0
    with open("./data/"+dataset_name+"_cot_palm.jsonl", "w") as wf:
        for sample in dataset:
            print("Index: ", total)
            print("Question: ", sample["question"])
            total+=1
            if palm_not_processed_index:
                if total not in palm_not_processed_index:
                    continue
            prompt = cot_prompt_hotpotqa_huggingface_palm(sample)
            
            ans, evidence, completion, fail  = palm_api(prompt, "cot")
            if fail: failed+=1
            true_labels.append(sample["answer"])
            pred_labels.append(ans)
            sample.update({"idx": str(total-1), "predicted_answer": ans, "step_by_step_reasoning": evidence})
            try: 
                wf.write(json.dumps(sample)+"\n")
                
            except:
                
                wf.write("\n")
            if total==limit: 
                break
        
        print(json.dumps(evaluate(true_labels, pred_labels)))
        print(f"Processed {total-failed} out of {total}")
        return pred_labels

def main_e2g(limit=None):
    print("loading data")
    dataset = datasets.load_dataset(dataset_name, "distractor", split="validation", cache_dir="./data")
    print("data loading done")
    true_labels = []
    pred_labels = []
    total = 0
    failed = 0
    palm_not_processed_index = []
        
    with open("./data/"+dataset_name+"_e2g_palm.jsonl", "w") as wf:
       
        for sample in (dataset):
            print("Index: ", total)
            print("Question: ", sample["question"])
            total+=1
            # if total>16: break
            # if total<15: continue
            prompt = e2g_prompt_hotpotqa_huggingface_palm(sample)
            ans, evidence, completion, fail  = palm_api(prompt, "e2g")
            ans =  ans.strip('"').strip(":").strip().strip(",").strip()
            evidence = evidence.strip('"').strip(":").strip().strip(",").strip()
            import pdb
            # pdb.set_trace()
            if not ans:
                prompt = cot_prompt_hotpotqa_huggingface_palm(sample)
                ans, evidence, completion, fail  = palm_api(prompt, "cot")
                if fail: failed+=1
            else:
                print("evidence: ", evidence)
                sample["context"]["sentences"] = [evidence] 
                prompt = e2g_prompt_hotpotqa_huggingface_palm(sample)
                ans1, evidence1, completion1, fail1  = palm_api(prompt, "e2g")
                
                if ans1:
                    ans = ans1
                    evidence = evidence1
            # pdb.set_trace()
            if not ans:
                prompt = cot_prompt_hotpotqa_huggingface_palm(sample)
                ans, evidence, completion, fail  = palm_api(prompt, "cot")
            ans = ans.strip('"').strip(":").strip().strip(",").strip()
            if not ans:
                palm_not_processed_index.append(total) 
            true_labels.append(sample["answer"])
            pred_labels.append(ans)
            try: 
                sample.update({"idx": total-1, "predicted_answer": ans, "evidence_and_explanation": evidence})
                wf.write(json.dumps(sample)+"\n")
            except:
                wf.write("\n")
            
            if limit and total==limit: 
                break

        print("Not processd: ", str(palm_not_processed_index))
        print(json.dumps(evaluate(true_labels, pred_labels)))
        print(f"Processed {total-failed} out of {total}")

        return true_labels, pred_labels, palm_not_processed_index
    
        
pred_labels = main_cot(limit=1453)
# true_labels, pred_labels, palm_not_processed_index= main_e2g()

# cot_preds_for_not_processed_examples = main_cot(palm_not_processed_index)
# for idx, pred in zip(palm_not_processed_index, cot_preds_for_not_processed_examples):
#     pred_labels[idx-1] = pred
    
# print(json.dumps(evaluate(true_labels, pred_labels)))

with open("./data/"+dataset_name+"_e2g_palm.jsonl", "r") as e2gf, open("./data/"+dataset_name+"_cot_palm.jsonl", "r") as cotf:
    e2g_preds = []
    cot_preds = []
    true_labels = []
    for i, (e2g_l, cot_l) in enumerate(zip(e2gf.readlines(), cotf.readlines())):
        if i>99: break
        e2g = json.loads(e2g_l)
        cot=  json.loads(cot_l)
        assert e2g["answer"] == cot["answer"]
        true_label = e2g["answer"]
        true_labels.append(true_label)
        e2g_preds.append(e2g["predicted_answer"])
        cot_preds.append(cot["predicted_answer"])
        sep_e2g = "evidence_and_explanation"
        sep_cot = "step_by_step_reasoning"
        if e2g["predicted_answer"]==true_label and cot["predicted_answer"]!=true_label:
            print(f"Idx {i}, ")
            print("Qustion: ", e2g["question"])
            print("True answer and e2g answer: ", e2g["predicted_answer"])
            print("True evidence: ",  e2g[sep_e2g])
            print("Incorrect cot answer: ", cot["predicted_answer"])
            print("Incorrect evidence: ",  cot[sep_cot])
            print("-"*10)

    print("cot results: ", json.dumps(evaluate(true_labels, cot_preds)))
    print("e2g results: ", json.dumps(evaluate(true_labels, e2g_preds)))
