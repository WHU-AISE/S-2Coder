from utils import save_pkl_gz, load_json, get_all_sub_files, load_model, emb_text, bool_is_function_node
# Load model directly
from codeGRAG import build_csg_from_code
import torch
import os

def build_slide_window_for_file(file_path, file_code, window_size = 50, slice_step = 10):
    code_windows = []
    file_code = file_code.replace('\n\n', '\n')
    code_lines = file_code.splitlines()
    delta_size = window_size // 2

    for line_no in range(0, len(code_lines), slice_step): # line_no starts from 0
        start_line_no = max(0, line_no - delta_size)
        end_line_no = min(len(code_lines), line_no + window_size - delta_size)
        window_lines = [i for i in code_lines[start_line_no:end_line_no]]
        if not window_lines:  # all empty lines
            continue
        window_text = '\n'.join(window_lines)
        code_windows.append({'code': window_text,
                             'emb': [],
                             'metadata':{
                                "file_path": file_path,
                                "start_line_no": start_line_no,
                                "end_line_no": end_line_no,
                             }})
    return code_windows

def get_code_blocks(file_code):
    file_code = file_code.replace('\n\n', '\n')
    lines = file_code.splitlines()
    blocks = []
    current_block = []
    start_line = None

    for i, line in enumerate(lines):  # 行号从1开始
        if line.strip():  # 非空行
            if start_line is None:
                start_line = i  # 记录块的起始行号
            current_block.append(line)
        else:  # 空行
            if current_block:
                blocks.append({
                    "lines": current_block,
                    "start_line": start_line,
                    "end_line": i - 1
                })
                current_block = []
                start_line = None

    # 处理最后一个块
    if current_block:
        blocks.append({
            "lines": current_block,
            "start_line": start_line,
            "end_line": len(lines)
        })

    return blocks

def split_large_block(block, T = 15):
    blocks = []
    lines = block["lines"]
    start_line = block["start_line"]
    end_line = block["end_line"]

    for i in range(0, len(lines), T):
        sub_block_lines = lines[i:i+T]
        sub_block = {
            "lines": sub_block_lines,
            "start_line": start_line + i,
            "end_line": min(start_line + i + T - 1, end_line)
        }
        blocks.append(sub_block)
    return blocks

def build_split_aggregate_for_file(file_path, file_code, T = 15):
    code_blocks = []
    current_blocks = {}
    mini_blocks = get_code_blocks(file_code)
    for mini_block in mini_blocks:
        if len(mini_block["lines"]) > T:
            if current_blocks:
                code_blocks.append(current_blocks)
                current_blocks = []
            split_blocks = split_large_block(mini_block, T)
            code_blocks.extend(split_blocks)
        else:
            if not current_blocks:
                current_blocks = mini_block
            else:
                if len(current_blocks["lines"]) + len(mini_block["lines"]) <= T:
                    current_blocks["lines"].extend(mini_block["lines"])
                    current_blocks["end_line"] = mini_block["end_line"]
                else:
                    code_blocks.append(current_blocks)
                    current_blocks = mini_block

    if current_blocks:
        code_blocks.append(current_blocks)

    split_aggregate_blocks = []
    for block in code_blocks:
        block_text = '\n'.join(block["lines"])
        split_aggregate_blocks.append({
            'code': block_text,
            'emb': [],
            'metadata':{
                "file_path": file_path,
                "start_line_no": block["start_line"],
                "end_line_no": block["end_line"],
            }
        })
    return split_aggregate_blocks
    


def emb_slide_windows_for_a_file(model_name, tokenizer, model, config, device, file_path, file_code, window_size = 50, slice_step = 10):
    code_windows = build_slide_window_for_file(file_path, file_code, window_size, slice_step)
    embed_code_windows = []
    for i in range(len(code_windows)):
        code_window = code_windows[i]
        code_emb = emb_text(model_name, tokenizer, model, config, code_window['code'], device)
        embed_window = {
            'code': code_window['code'],
            'emb': code_emb,
            'metadata': code_window['metadata']
        }
        embed_code_windows.append(embed_window)
    
    return embed_code_windows

def emb_split_aggregate_for_a_file(model_name, tokenizer, model, config, device, file_path, file_code, T = 15):
    code_split_aggregate = build_split_aggregate_for_file(file_path, file_code, T)
    embed_code_split_aggregate = []
    for i in range(len(code_split_aggregate)):
        code_block = code_split_aggregate[i]
        code_emb = emb_text(model_name, tokenizer, model, config, code_block['code'], device)
        embed_block = {
            'code': code_block['code'],
            'emb': code_emb,
            'metadata': code_block['metadata']
        }
        embed_code_split_aggregate.append(embed_block)
    
    return embed_code_split_aggregate

def build_retrieve_corpus_for_a_repo(graph_code_path, model_name, tokenizer, model, config, device, type, window_size = 50, slice_step = 10, T = 15):
    if type == "slide_window":
        save_path = graph_code_path.replace("/graph_code/", "/slide_window/")
    elif type == "Split-Aggregate":
        save_path = graph_code_path.replace("/graph_code/", "/split_aggregate/")
    else:
        raise ValueError(f"Unsupported retrieve corpus type: {type}")
    save_path = save_path.replace("_all.json", ".pkl.gz")

    if os.path.exists(save_path):
        print(f"Retrieve corpus already exists at {save_path}, skipping...")
        return


    repo_graph_code = load_json(graph_code_path)

    repo_retrieve_corpus = {}

    for key in repo_graph_code.keys():
        if "::" in key:
            continue
        if type == "slide_window":
            embed_code_windows = emb_slide_windows_for_a_file(model_name, tokenizer, model, config, device, key, repo_graph_code[key]['code'], window_size, slice_step)
        elif type == "Split-Aggregate":
            embed_code_windows = emb_split_aggregate_for_a_file(model_name, tokenizer, model, config, device, key, repo_graph_code[key]['code'], T)
        else:
            raise ValueError(f"Unsupported retrieve corpus type: {type}")
        for i in range(len(embed_code_windows)):
            repo_retrieve_corpus[key+"::"+str(i)] = embed_code_windows[i]

    
    save_pkl_gz(repo_retrieve_corpus, save_path)


def build_CodeGRAG_corpus_for_a_repo(graph_code_path, model_name, tokenizer, model, config, device):
    if "CoderEvalPython" in graph_code_path or "DevEval" in graph_code_path:
        language = "python"
    elif "CoderEvalJava" in graph_code_path:
        language = "java"
    else:
        raise ValueError(f"Unsupported dataset language")
    save_path = graph_code_path.replace("/graph_code/", "/codeGRAG_index/")
    repo_name = graph_code_path.split("/")[-1].replace("_all.json", "")
    save_path = save_path.replace("_all.json", ".pkl.gz")

    if os.path.exists(save_path):
        print(f"Retrieve corpus already exists at {save_path}, skipping...")
        return
    
    repo_graph_code = load_json(graph_code_path)
    repo_retrieve_corpus = {}

    for key in repo_graph_code.keys():
        if bool_is_function_node(key, repo_name):
            code = repo_graph_code[key]["code"]
            csg = build_csg_from_code(code, language)
            if csg is None:
                embed = emb_text(model_name, tokenizer, model, config, code, device)
                repo_retrieve_corpus[key] = {
                    "code": code,
                    "csg": None,
                    "emb": embed
                }
            else:
                embed = emb_text(model_name, tokenizer, model, config, code+str(csg), device)
                repo_retrieve_corpus[key] = {
                    "code": code,
                    "csg": str(csg),
                    "emb": embed
                }
    
    save_pkl_gz(repo_retrieve_corpus, save_path)


def build_retrieve_corpus_for_all_repos():
    model_name = "unixcoder"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model, config = load_model(model_name)
    model.to(device)

    root_dir_path = "/home/inspur/nsy/4-3090/All_experiment0804/graph_code"
    all_graph_code_files = get_all_sub_files(root_dir_path)

    for graph_code_file in all_graph_code_files:
        if graph_code_file.endswith("_all.json"):
            print(f"Processing repository graph code file: {graph_code_file}")
            #build_retrieve_corpus_for_a_repo(graph_code_file, model_name, tokenizer, model, config, device, "slide_window")
            build_CodeGRAG_corpus_for_a_repo(graph_code_file, model_name, tokenizer, model, config, device)
    
if __name__ == "__main__":
    build_retrieve_corpus_for_all_repos()
    print("Finished building retrieve corpus for all repositories.")




    



