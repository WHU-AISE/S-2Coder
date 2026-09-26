"""
 Agent - 
"""
from collections import defaultdict
import math
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = ""
from pydantic import BaseModel, Field
import networkx as nx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Dependency(BaseModel):
    """"""
    node_id: str = Field(description="ID")
    callers: list = Field(description="")

class DependencyAnalyzer:
    """"""
    
    def __init__(self, graphs_root: str = '', dataset: str = 'DevEval'):
        """
        
        
        Args:
            **kwargs: （temperature, api_base ）
        """
        self.graphs_root = graphs_root
        self.dataset = dataset
        self.dependency_graph = None

    def load_graphml(self, repo_name: str):
        file_path = os.path.join(self.graphs_root, f"{repo_name}.graphml")
        return nx.read_graphml(file_path)

    def get_callers(self, target_node, function_file_path, function_name, line_num):
        """
         target_node （）
        """
        callers = []
        for u, v, attrs in self.dependency_graph.edges(data=True):
            if attrs.get("relationship") != "call" or v != target_node:
                continue
            parts = u.split("::")
            if parts[0] == function_file_path and parts[2] == function_name and int(parts[3]) == line_num:
                continue
            if self.dataset == "CoderEvalPython":
                if parts[0] == function_file_path and parts[2] == function_name:
                    continue
                if "passk_validte" in u or "\\\\test_" in u:
                    continue
            if "build\\lib" in u:
                continue
            if "site-packages" in u:
                continue
            callers.append(u)
        return callers

    def get_callees(self, target_node):
        """
        （）
        """
        callees = []
        for u, v, attrs in self.dependency_graph.edges(data=True):
            if attrs.get("relationship") != "call" or u != target_node:
                continue
            callees.append(v)
        return callees

    def get_function_calls(self, input_code, repo_name, repo_data, target_node, function_file_path, function_name, line_num):
        """
         + PPL：
        1.  target_node ；
        2. ；
        3.  PPL ；
        4. 
        """
        # ： target_node ，，
        self.dependency_graph = self.load_graphml(repo_name)
        callers = self.get_callers(target_node, function_file_path, function_name, line_num)
        # ： caller 
        caller_to_callees = {}
        for caller in callers:
            callees = self.get_callees(caller)
            caller_to_callees[caller] = tuple(sorted(set(callees)))
        # ：
        grouped_funcs = defaultdict(list)
        for caller, callees_key in caller_to_callees.items():
            grouped_funcs[callees_key].append(caller)
        grouped = list(grouped_funcs.values())
        # ： PPL 
        selected_funcs = []
        for group in grouped:
            if len(group) == 1:
                selected_funcs.append(repo_data.get(group[0], {}).get("code", ""))
                continue
            ranked = self.rank_docs_by_ppl(input_code, group, 1)
            selected_funcs.append(repo_data.get(ranked[0][1], {}).get("code", ""))

        functions = [f[1] for f in self.rank_docs_by_ppl(input_code, selected_funcs, 2)]
        return Dependency(node_id=target_node, callers=functions)

    def rank_docs_by_ppl(
        self,
        input_code: str,
        docs: list[str],
        top_k: int,
        model_name: str = "gpt2",
    ):
        """
         PPL (PPL )

        Args:
            input_code (str): 
            docs (list[str]): 
            top_k: 
            model_name (str): HuggingFace (: "gpt2")

        Returns:
             return_all=False: (best_doc, best_ppl)
             return_all=True:  [(ppl, doc), ...]（ PPL ）
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()

        scores = []
        query = (
            "You are writing the function below. Based on the following code snippets, which one best helps you complete it?\n\n"
            "Target code:\n" + input_code
        )
        for d in docs:
            text = f"Q: {query}\nContext: {d}\nA:"
            ppl = self.get_ppl(model, tokenizer, text)
            scores.append((ppl, d.replace("\n\n", "\n")))

        scores.sort(key=lambda x: x[0])
        return scores[:top_k]

    def get_ppl(self, model, tokenizer, text: str) -> float:
        """ Perplexity (PPL)"""
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        return math.exp(outputs.loss.item())