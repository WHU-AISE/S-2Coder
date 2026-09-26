import json
import networkx as nx
import gzip
import pickle
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch

root_dir_path = "/home/user/nsy/dataset/All_experiment0804/"


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)  # Use ensure_ascii=False for non-ASCII characters

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')  # Write each item as a JSON line

def load_pkl_gz(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

def save_pkl_gz(data, path):
    with gzip.open(path, "wb") as f:
        pickle.dump(data, f)



def get_all_sub_files(directory_path):
    all_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    return all_files


def bool_is_function_node(node_name, repo_name):
    if repo_name == "skywalking" or repo_name == "Telethon":
        if len(node_name.split("::")) == 5:
            return True
    else:
        if len(node_name.split("::")) == 7:
            return True
    return False


def load_model(model_name):
    if model_name == "unixcoder":
        model_path = "/home/inspur/nsy/4-3090/models/unixcoder"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        return tokenizer, model, config
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    

def emb_text(model_name, tokenizer, model, config, text, device, max_length=512):
    if model_name == "unixcoder":
        tokens = tokenizer.tokenize(text)
        tokens = tokens[:max_length-4]
        tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + tokens + [tokenizer.sep_token] 
        tokens_ids = [tokenizer.convert_tokens_to_ids(tokens)]
        source_ids = torch.tensor(tokens_ids).to(device)

        mask = source_ids.ne(config.pad_token_id)
        token_embeddings = model(source_ids,attention_mask = mask.unsqueeze(1) * mask.unsqueeze(2))[0]
        sentence_embeddings = (token_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1)
        return sentence_embeddings[0].cpu().detach().numpy()  

    else:
        raise ValueError(f"Unsupported model: {model_name}")