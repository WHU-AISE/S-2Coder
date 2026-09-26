"""
 Agent - 
（）
"""
import re
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field
import numpy as np
from pathlib import Path
from llm_client import get_embedding
from utils.file_util import load_jsonl
from itertools import chain


class CodeSnippet(BaseModel):
    """"""
    node_id: str = Field(description="ID")
    file_path: str = Field(description="")
    function_name: str = Field(description="/")
    code: str = Field(description="")
    similarity_score: float = Field(default=0.0, description="")


class CodeRetriever:
    """ - """
    
    def __init__(self, dataset: str = "DevEval"):
        self.dataset = dataset
        self.dataset_path = Path("") / dataset
        self.standard_types = {"str", "int", "float", "bool", "list", "dict", "tuple", "set", "None", "Optional", "String"}
    
    def retrieve(self, data: Dict, task_description: str, top_k: int = 1, repo_data: Optional[Dict] = None, seen_node_ids: set = None) -> List[CodeSnippet]:
        """
        
        
        Args:
            data: ，
            task_description: 
            top_k: 
            repo_data:  {node_id: {code, code_embedding, annotation_embedding}}
            
        Returns:
            
        """
        #  embedding， task_description 
        if task_description is not None:
            print(f"🔄  embedding...")
            try:
                embedding_list = get_embedding(task_description, model="text-embedding-3-small")
                requirement_embedding = np.array(embedding_list)
                print(f"✓ Embedding ，: {len(embedding_list)}")
            except Exception as e:
                print(f"❌  embedding : {e}")
                return []
        
        if requirement_embedding is not None and repo_data is not None:
            return self._retrieve_by_embedding(data, repo_data, requirement_embedding, top_k, seen_node_ids)
        
        # ，
        print(f"⚠️  ，")
        return []
    
    def _retrieve_by_embedding(self, data: Dict, repo_data: Dict, requirement_embedding: np.ndarray, top_k: int = 5, seen_node_ids: set = None) -> List[CodeSnippet]:
        """
        
        
        Args:
            repo_data:  {node_id: {code, code_embedding, annotation_embedding}}
            requirement_embedding:  embedding 
            top_k: 
            
        Returns:
            
        """
        try:
            #  repo_data
            if not repo_data:
                print(f"❌ ")
                return []
            
            # 
            similarities = []
            
            for node_id, node_info in repo_data.items():
                if node_id in seen_node_ids:
                    continue
                if node_id.split("::")[0] == data.get("function_file_path") and node_id.split("::")[2] == data.get("function_name") and int(node_id.split("::")[3]) == data.get("line_num"):
                    continue
                # CoderEvalPython  ground truth +（ line_num）
                if self.dataset == "CoderEvalPython":
                    node_parts = node_id.split("::")
                    if len(node_parts) >= 3 and node_parts[0] == data.get("function_file_path") and node_parts[2] == data.get("function_name"):
                        continue
                    if "passk_validte" in node_id or "\\test_" in node_id:
                        continue
                if "build\\lib" in node_id:
                    continue
                if "site-packages" in node_id:
                    continue
                #  annotation_embedding  code_embedding
                annotation_embedding = node_info.get('annotation_embedding')
                code_embedding = node_info.get('code_embedding')
                code = node_info.get('code', '')
                
                if annotation_embedding is None or code_embedding is None:
                    continue
                
                #  numpy 
                annotation_emb = np.array(annotation_embedding) if not isinstance(annotation_embedding, np.ndarray) else annotation_embedding
                code_emb = np.array(code_embedding) if not isinstance(code_embedding, np.ndarray) else code_embedding
                
                # 
                annotation_sim = self._cosine_similarity(requirement_embedding, annotation_emb)
                code_sim = self._cosine_similarity(requirement_embedding, code_emb)
                
                # （：50% annotation + 50% code）
                combined_sim = 0.5 * annotation_sim + 0.5 * code_sim
                
                similarities.append({
                    'node_id': node_id,
                    'similarity': combined_sim,
                    'node_info': node_info,
                    'annotation_sim': annotation_sim,
                    'code_sim': code_sim,
                    'code': code  
                })
            
            # 
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            print(f"✓ ， Top-{top_k} ")
            
            #  top_k 
            snippets = []
            for item in similarities[:top_k]:
                node_info = item['node_info']
                function_name=item['node_id'].split("::")[0].split("\\")[-1]
                snippet = CodeSnippet(
                    node_id=item['node_id'],  #  node_id 
                    function_name=function_name,  # 
                    file_path=item['node_id'].split(function_name)[0].split("::")[0],  # 
                    code=node_info.get('code', '').replace("\n\n", "\n"),
                    similarity_score=float(item['similarity'])
                )
                snippets.append(snippet.model_dump())
            return snippets
            
        except Exception as e:
            print(f"❌ : {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        
        
        Args:
            vec1: 1
            vec2: 2
            
        Returns:
             (0-1)
        """
        try:
            #  numpy 
            v1 = np.array(vec1) if not isinstance(vec1, np.ndarray) else vec1
            v2 = np.array(vec2) if not isinstance(vec2, np.ndarray) else vec2
            
            # 
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # 
            similarity = np.dot(v1, v2) / (norm1 * norm2)
            
            #  0-1 （）
            return max(0.0, min(1.0, float(similarity)))
        except Exception as e:
            print(f": {e}")
            return 0.0

    def extract_io_types(self, docstring: str) -> Tuple[List[Tuple[str, str]], Optional[str]]:
        """
         docstring 
        """
        inputs = []
        output = None
        
        param_pattern = r":param\s+(\w+):\s+([\w\[\], ]+)\."
        for match in re.finditer(param_pattern, docstring):
            param_name, type_name = match.groups()
            inputs.append((param_name, type_name.strip()))
        
        return_pattern = r":return:\s+([\w\[\], ]+)\."
        match = re.search(return_pattern, docstring)
        if match:
            output = match.group(1).strip()
        
        return inputs, output

    def parse_generic_types(self, type_str: str) -> List[str]:
        """
        ，
        :
            "List[CustomClass]" -> ["CustomClass"]
            "Dict[String, CustomClass]" -> ["CustomClass"]
        """
        # 
        # 
        type_names = re.findall(r'\b([A-Z][A-Za-z0-9_]*)\b', type_str)
        return [t for t in type_names if t not in self.standard_types]

    def retrieve_custom_class_definitions_multi(
        self,
        docstring: str,
        file_path: str,
    ) -> Tuple[List[dict], List[dict]]:
        """
         docstring ，，
        
        
        Args:
            docstring:  docstring
            file_path: 
        
        Returns:
            Tuple[, ]， dict， class_name  code
        """
        input_results = []
        output_results = []
        
        inputs, output = self.extract_io_types(docstring)
        
        # 
        input_types = list(chain.from_iterable(self.parse_generic_types(t) for _, t in inputs))
        for class_name in input_types:
            snippet = self.retrieve_class_definition(class_name, file_path)
            if snippet:
                input_results.append({
                    "class_name": class_name,
                    "code": snippet.code
                })
        
        # 
        if output:
            output_types = self.parse_generic_types(output)
            for class_name in output_types:
                snippet = self.retrieve_class_definition(class_name, file_path)
                if snippet:
                    output_results.append({
                        "class_name": class_name,
                        "code": snippet.code
                    })
        
        return input_results, output_results
    
    def retrieve_class_definition(
        self,
        class_name: str,
        file_path: str
    ) -> Optional[CodeSnippet]:
        """
        
        
        Args:
            class_name: 
            file_path: 
            
        Returns:
            CodeSnippet ，， None
        """
        from pathlib import Path

        try:
            file = Path(file_path)
            if not file.exists():
                print(f"⚠️ : {file_path}")
                return None

            codes = load_jsonl(file)
            for c in codes:
                if c["class"] == class_name and c["name"] == "__init__":
                    return CodeSnippet(
                        node_id=c["src_file"].split("/", 3)[-1]+"::"+class_name+"::__init__",
                        file_path=c["src_file"].split("/", 3)[-1],
                        function_name="__init__",
                        code=f"class {class_name}:\n    {c['code_chunk']}"
                    )
            return None

        except Exception as e:
            print(f": {e}")
            import traceback
            traceback.print_exc()
            return CodeSnippet(
                node_id="",
                file_path="",
                function_name="",
                code=""
            )
