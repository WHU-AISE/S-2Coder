from utils import load_model, emb_text, load_json, save_pkl_gz
import torch

def parse_dataset(dataset_name, model_name, tokenizer, model, config, device):
    dataset_all_info_path = "/home/inspur/nsy/4-3090/dataset_all_info/"+dataset_name+"_dataset_all_infor.json"
    dataset_all_infor = load_json(dataset_all_info_path)

    req_emb_list = []

    for i in range(len(dataset_all_infor)):
        item = dataset_all_infor[i]
        if "CoderEval" in dataset_name:
            req = item['human_requeriment']
        elif "DevEval" in dataset_name:
            req = item['all_requeriment']
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        req_emb = emb_text(model_name, tokenizer, model, config, req, device)
        req_emb_list.append({
            "id": i,
            "req": req,
            "req_emb": req_emb
            } 
        )
    return req_emb_list



if __name__ == "__main__":
    dataset_list = ["CoderEvalJava", "CoderEvalPython", "DevEval"]
    model_name = "unixcoder"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model, config = load_model(model_name)
    model.to(device)

    Req_emb_dict = {}

    for dataset_name in dataset_list:
        print(f"Processing dataset: {dataset_name}")
        req_emb_list = parse_dataset(dataset_name, model_name, tokenizer, model, config, device)
        Req_emb_dict[dataset_name] = req_emb_list
    
    save_path = "/home/inspur/nsy/4-3090/req_emb_unixcoder.pkl.gz"
    save_pkl_gz(Req_emb_dict, save_path)


"""
{
    "dataset_name": [
        {
            "id": int, 第几条
            "req": str, 需求描述
            "req_emb": List[float] unixcoder_emb
        },
        ...
    ]

}

"""