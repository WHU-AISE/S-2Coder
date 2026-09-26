
import json
import gzip
import pickle
from tqdm import tqdm
# from config import *

DATASET_DIR = "/home/user/nsy/dataset/"
EXPERIMENT_DIR = "/home/user/lyl/Experiment/"

def get_func_dict(dataset, repo_name):
    dir_path = DATASET_DIR+"Functions/"
    func_dict_path = dir_path+dataset+"/"+repo_name+".pkl.gz"
    return load_pkl_gz(func_dict_path)

def load_pkl_gz(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

def get_dataset_all_info(dataset):
    dir_path = "/home/user/lyl/Dataset/dataset_all_info/"
    dataset_all_info_path = dir_path+dataset+"_dataset_all_infor.json"
    return load_json(dataset_all_info_path)

def get_sub_req_info(dataset, type=""):
    dir_path = EXPERIMENT_DIR+"divide/"
    sub_req_info_path = dir_path+dataset+f"/{type}requirement_all.jsonl"
    return load_jsons(sub_req_info_path)

def get_dataset_name(dataset, language=None):
    if dataset == "DevEval":
        return dataset
    elif dataset == "CoderEval":
        if language == "java":
            return "CoderEvalJava"
        elif language == "python":
            return "CoderEvalPython"
        else:
            raise ValueError(f"Unsupported language: {language} for dataset CodeEval")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

def get_top_similar(score_list, top_k):
    sorted_list = sorted(score_list, key=lambda x: x[1], reverse=True)
    return sorted_list[:top_k]

def save_retrieve_socre(dataset, scores, sub=""):
    dir_path = EXPERIMENT_DIR+f"retrieve_scores/{dataset}/"
    retrieve_score_path = dir_path+sub+"_req_retrieve_scores.json"
    save_json(scores, retrieve_score_path)

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_jsons(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f.readlines()]

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def save_jsons(data, file_path):
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

def get_useful_calls(no_repeat_calls, all_top_similar):
    useful_calls = []
    for sim in all_top_similar:
        node = sim
        if node in no_repeat_calls:
            useful_calls.append(node)
    return useful_calls


file = "/home/inspur/lyl/RepoScope/datasets/CoderEvalJava.json"
data_all = load_json("/home/inspur/lyl/RepoScope/datasets/CoderEvalJava_dataset_all_infor.json")

dataset = load_json(file).get("RECORDS")
for i in tqdm(range(230)):
    d = dataset[i]
    da = data_all[i]
    d["file_path"] = da.get("function_file_path").replace("\\", "/")


save_json(dataset, "/home/inspur/lyl/RepoScope/datasets/CoderEvalJava.json")

