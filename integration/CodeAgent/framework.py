"""
 -  Agent 

：
1.  → 
2.  → （）
3.  → 
4.  → 
5.  → 
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = ""
from typing import List, Dict, Optional
from pathlib import Path
from agents.requirement_decomposer import RequirementDecomposer, Task
from agents.code_retriever import CodeRetriever
from agents.dependency_analyzer import DependencyAnalyzer
from agents.code_generator import CodeGenerator
from agents.test_agent import TestAgent
from agents.reflection import ReflectionAgent
import json
from utils.file_util import *


class CodeGenerationFramework:
    """"""
    
    def __init__(self, max_retries: int = 3, dataset: str = "DevEval", model_name: str = "gpt-4o-mini"):
        """"""
        self.max_retries = max_retries
        self.dataset = dataset
        self.model_name = model_name
        
        #  Agent
        self.requirement_decomposer = RequirementDecomposer(model_name=model_name)
        self.code_retriever = CodeRetriever(dataset=dataset)
        self.dependency_analyzer = DependencyAnalyzer()
        self.code_generator = CodeGenerator(model_name=model_name)
        self.test_agent = TestAgent(model_name=model_name)
        self.reflection_agent = ReflectionAgent(self.code_generator, self.code_retriever, model_name=model_name)
        
        # 
        self.execution_log = []
        
        # 
        self.dataset_path = Path("") / dataset
    
    def _load_repo_data(self, repo_name: str) -> Optional[Dict]:
        """
        
        
        Args:
            repo_name: 
            
        Returns:
            ， None
        """
        try:
            pkl_path = self.dataset_path / f"{repo_name}.pkl.gz"
            
            if not pkl_path.exists():
                print(f"❌ : {pkl_path}")
                return None
            
            print(f"📂 : {repo_name}.pkl.gz")
            data = load_pkl_gz(pkl_path)
            
            if not data:
                print(f"❌ {repo_name}.pkl.gz ")
                return None
            
            return data
            
        except Exception as e:
            print(f"❌ : {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _log(self, message: str):
        """"""
        print(f"[INFO] {message}")
        self.execution_log.append(message)
    
    def _log_step(self, step: str, content: str):
        """"""
        self._log(f"\n{'='*60}")
        self._log(f": {step}")
        self._log(f"{'='*60}\n{content}")
    
    def process_requirement(self, requirement: str, task_data: Dict = None) -> Dict:
        """
        ，：
        1.  → 
        2. （）
        3. 
        4. 
        
        Args:
            requirement: 
            task_data: （）
            
        Returns:
            
        """
        # 
        repo_name = task_data.get('repo_name')
        
        # ============  1:  ============
        self._log_step("1. ", requirement)
        
        tasks = self.requirement_decomposer.decompose(requirement)
        self._log(f" {len(tasks)} :\n")
        for task in tasks:
            self._log(f"  - [{task.id}] {task.name}")
            self._log(f"     {task.description}")
        print()
        
        # ============  2:  ============
        self._log_step("2. ", f" {len(tasks)} ")
        # （）
        repo_data = self._load_repo_data(repo_name)
        #  repo （GraphML -> dependency_graph），
        
        task_code_map = {}  # 
        
        for task in tasks:
            self._log(f"\n [{task.id}] {task.name} ...")
            code_snippets = self.code_retriever.retrieve(
                task_data,
                task.description, 
                top_k=1,
                repo_data=repo_data
            )
            upstreams = []
            # ： node_id（ file_path），
            for i, snippet in enumerate(code_snippets):
                self._log(f"  [{i+1}] \n {snippet.code}")

                node_id = getattr(snippet, 'node_id', None)
                dependency = self.dependency_analyzer.get_function_calls(task_data["input_code"], repo_name, repo_data, node_id, task_data["function_file_path"], task_data["function_name"], task_data["line_num"])
                upstream_chain = dependency.callers
                if upstream_chain != []:
                    for i in range(len(upstream_chain)):
                        self._log(f"  [{i+1}] :\n {upstream_chain[i]}")
                else:
                    self._log(f"   (node_id={node_id})")

                upstreams.append({"node_id": node_id, "upstream_calls": upstream_chain})

            task_code_map[task.id] = {
                "task": task,
                "code_snippets": code_snippets,
                "related_code": "\n".join([
                    f"# {getattr(snippet, 'node_id', getattr(snippet, 'file_path', 'unknown'))}\n{snippet.code}"
                    for snippet in code_snippets
                ]),
                "upstream_calls": upstreams
            }
        
        # ============  3:  ============
        self._log_step("3. ", "")
        
        all_related_code = "\n\n".join([
            f"=== Task {data['task'].id}: {data['task'].name} ===\n{data['related_code']}"
            for data in task_code_map.values()
        ])
        
        self._log(f" {sum(len(data['code_snippets']) for data in task_code_map.values())} ")
        self._log(f": {len(all_related_code)} ")
        print()
        
        # ============  4:  ============
        self._log_step("4. ", f" {len(tasks)} ")
        
        # 
        all_tasks_description = "\n\n".join([
            f"Task {task.id} - {task.name}:\n{task.description}"
            for task in tasks
        ])
        
        retry_count = 0
        generated_code = None
        test_passed = False
        
        while retry_count <= self.max_retries:
            self._log(f"\n ( {retry_count + 1}/{self.max_retries + 1})...")
            
            generated = self.code_generator.generate(
                function_name=repo_name,
                related_code=all_related_code,
                input_code=task_data["input_code"],
                retries=retry_count
            )
            
            generated_code = generated.code
            self._log(f"✓ ，: {len(generated_code)} ")
            self._log(f"\n:\n{generated_code}")
            
            # ============  5:  ============
            self._log("5. ...")
            test_result = self.test_agent.test(task_data["namespace"], generated_code)
            test_passed = test_result.passed
            
            # self._log(f": {'✓ ' if test_passed else '✗ '}")
            # if not test_passed:
            #     self._log(f": {test_result.test_output}")
            
            # ，
            if test_passed:
                self._log("\n✓ ，!")
                break
            # ：，
            if retry_count < self.max_retries:
                self._log(f"\n✗ ， ( {retry_count + 1})...")
                #  reflection agent 
                reflection = self.reflection_agent.reflect_and_regenerate(
                    function_name=repo_name,
                    input_code=task_data["input_code"],
                    related_code=all_related_code,
                    test_result=test_result,
                    retry_count=retry_count,
                )
                generated_code = reflection.code
                # （new generated_code ）
                retry_count += 1
                continue
            else:
                self._log("\n⚠️  ")
                break
        
        print()
        
        # ============  6:  ============
        self._log_step("", "")
        
        result_summary = {
            "overall_status": "SUCCESS" if test_passed else "INCOMPLETE",
            "tasks_count": len(tasks),
            "retrieval_snippets": sum(len(data['code_snippets']) for data in task_code_map.values()),
            "generated_code_length": len(generated_code) if generated_code else 0,
            "test_passed": test_passed,
            "retries": retry_count,
            "tasks": [{"id": task.id, "name": task.name} for task in tasks],
            "generated_code": generated_code,
            "test_result": {
                "passed": test_result.passed if test_result else False,
                "error_message": test_result.error_message if test_result else None
            },
            # "execution_log": self.execution_log
        }
        
        self._log(f": {result_summary['overall_status']}")
        self._log(f": {len(tasks)}")
        self._log(f": {result_summary['retrieval_snippets']}")
        self._log(f": {len(generated_code.split(chr(10))) if generated_code else 0}")
        self._log(f": {'PASS' if test_passed else 'FAIL'}")
        self._log(f": {retry_count}")
        
        print()
        
        return {"result": result_summary}
    
    def export_results(self, result_summary: Dict, output_file: str = "generation_results.json"):
        """JSON"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_summary, f, ensure_ascii=False, indent=2)
        
        self._log(f"\n: {output_file}")
        return result_summary
