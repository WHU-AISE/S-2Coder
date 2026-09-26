import os
import json

from tqdm import tqdm
import javalang

def get_call_line(repo_root, out_file):
    results = {}

    for root, _, files in os.walk(repo_root):
        for f in files:
            if not f.endswith(".java"):
                continue
            file_path = os.path.join(root, f)
            with open(file_path, "r", encoding="utf-8") as src_file:
                src = src_file.read()
            try:
                tree = javalang.parse.parse(src)
            except Exception as e:
                print(f" {file_path}: {e}")
                continue

            # 
            for _, cls_node in tree.filter(javalang.tree.ClassDeclaration):
                class_name = cls_node.name
                for _, method_node in cls_node.filter(javalang.tree.MethodDeclaration):
                    method_name = method_node.name
                    caller_line = method_node.position.line if method_node.position else None
                    caller = class_name + "::" + method_name + "::" + str(caller_line)

                    for _, call_node in method_node.filter(javalang.tree.MethodInvocation):
                        key = file_path.split("/", 8)[8]  #  atmosphere 
                        if key not in results:
                            results[key] = {}

                        if caller not in results[key]:
                            results[key][caller] = []

                        results[key][caller].append({
                            "callee": call_node.member,
                            "line": call_node.position.line if call_node.position else None
                        })

    #  JSON
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"， {out_file}")


def main():
    repo_dir = "/home/inspur/lyl/RepoScope/repos/CoderEvalJava"
    out_dir = "/home/inspur/lyl/RepoScope/datasets/Call_Line"
    for repo_name in tqdm(os.listdir(repo_dir)):
        repo_root = os.path.join(repo_dir, repo_name)
        out_file = os.path.join(out_dir, repo_name+".json")
        get_call_line(repo_root, out_file)

main()