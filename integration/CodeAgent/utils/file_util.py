

import gzip
import json
import os
from pathlib import Path
import pickle

def load_pkl_gz(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

def save_pkl_gz(data, path):
    with gzip.open(path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Data saved to {path}")

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f.readlines()]
    
def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Data saved to {file_path}")

def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Data saved to {file_path}")

def json_to_dict(input_file, key):
    data_dict = {}
    with open(input_file, 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            data_dict[data.get(key)] = data
    return data_dict

def get_result():
    data_list = load_json("")
    dmcoder_result = load_jsonl("")
    f1 = ""
    data1 = json_to_dict(f1, "namespace")
    dmcoder_passk = 0
    for d in dmcoder_result:
        if d["Result"] == "Pass":
            dmcoder_passk += 1
    pass_k = 0
    once_pass_k = 0
    error = 0
    for idx, task_info in enumerate(data_list):
        namespace = task_info.get("namespace", f"unknown_{idx}")
        fname = namespace.replace(".", "_")
        persist_file = ""
        persist = load_json(persist_file)
        if persist["test_history"][0]["passed"] or (len(persist["test_history"]) > 1 and persist["test_history"][1]["passed"] and persist["test_history"][0]["error"] == ""):
            once_pass_k+=1
        if persist["test_history"][-1]["passed"]:
            pass_k+=1
        if data1.get(namespace)["Result"] == "Pass" and not persist["test_history"][-1]["passed"]:
            error+=1
            # persist["test_history"]= {
            #     "retry": 0,
            #     "passed": True,
            #     "error": ""
            # },
            # persist["generated_history"]= {
            #     "retry": 0,
            #     "code": f'{task_info.get("input_code")}\n{data1.get(namespace)["completion"]}'
            # }
            # save_json(persist, "")
            
    print("DMcoderpass_1: " + str(round(dmcoder_passk/len(data_list)*100, 2)) + "%")
    print("pass_1: " + str(round(pass_k/len(data_list)*100, 2)) + "%")
    print("agentpass_1: " + str(round(once_pass_k/len(data_list)*100, 2)) + "%")
    print(pass_k)
    print(error)

# get_result()

def get_CoderEval_result():
    data_list = load_json("")
    dmcoder_result = load_json("")
    dmcoder_passk = 0
    pass_k = 0
    once_pass_k = 0
    for idx, task_info in enumerate(data_list):
        namespace = task_info.get("namespace", f"unknown_{idx}")
        fname = namespace.replace(".", "_").replace("\\\\", "\\")
        persist_file = ""
        persist = load_json(persist_file)
        if persist["test_history"][0]["passed"] or (len(persist["test_history"]) > 1 and persist["test_history"][1]["passed"] and persist["test_history"][0]["error"] == ""):
            once_pass_k+=1
        if persist["test_history"][-1]["passed"]:
            pass_k+=1
    # print("CoderEvalpass_1: " + str(round(dmcoder_passk/len(data_list)*100, 2)) + "%")
    print(pass_k)
    print("pass_1: " + str(round(pass_k/len(data_list)*100, 2)) + "%")
    print("agentpass_1: " + str(round(once_pass_k/len(data_list)*100, 2)) + "%")

# get_CoderEval_result()