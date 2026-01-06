from utils import *

def load_dataset(dataset_name, language):
    if dataset_name == "CoderEval":
        return load_CoderEval(language)
    elif dataset_name == "DevEval":
        return load_DevEval()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_CoderEval(language):
    dataset_path = f"./datasets/CoderEval{language}.json"
    dataset_json = load_json(dataset_path)["RECORDS"]
    dataset = []
    for data in dataset_json:
        repo = data["project"]
        function_name = data["name"]
        line = data["lineno"]
        file_path = "/".join(data["package"].split(".")) + "/" + data["file_name"]
        req = data["human_label"]
        dataset.append({
            "repo": repo,
            "function_name": function_name,
            "line": line,
            "file_path": file_path,
            "requirement": req
        })
    return dataset

def load_DevEval():
    dataset_path = "./datasets/DevEval.jsonl"
    dataset_json = load_jsonl(dataset_path)
    dataset = []
    for data in dataset_json:
        repo = data["project_path"].split('/')[-1]
        function_name = data["namespace"].split('.')[-1]
        line = data["signature_position"][0]
        file_path = "/".join(data["completion_path"].split("/")[1:])
        req = data["requirement"]["Functionality"] + "\n" + data["requirement"]["Arguments"]
        dataset.append({
            "repo": repo,
            "function_name": function_name,
            "line": line,
            "file_path": file_path,
            "requirement": req
        })
    return dataset




