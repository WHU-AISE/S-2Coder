import json
import os
from pathlib import Path
import re
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from agents.test_agent import TestResult
from llm_client import get_deepseek_client, get_openai_client

COMMENT_LIBRARY_ROOT = Path("")

class ReflectionResult(BaseModel):
    code: str = Field(description="Regenerated code")
    explanation: Optional[str] = Field(default=None, description="Reflection explanation")


class ReflectionAgent:
    """："""

    def __init__(self, code_generator, code_retriever, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        """
        Args:
            code_generator: ， `generate(function_name, input_code, related_code, retries)`
        """
        self.code_generator = code_generator
        self.code_retriever = code_retriever
        self.llm_client = get_openai_client(model=model_name, temperature=temperature)
        self.prompt_text = PromptTemplate(
            input_variables=[
                "code",
                "error_msg"
            ],
            template="""
You are a professional Python code analysis assistant.

Here is the result of an automated test for a Python function.

Please provide a concise summary that includes:
1. Possible root cause(s) of the failure
2. Suggested improvements or changes to the function implementation that would fix the issue
3. A list of class names that appear in the code or error message where the initialization (__init__) arguments are unclear. 
   These are classes whose __init__ signatures should be retrieved for further inspection.

Error code:
{code}

Error message:
{error_msg}

Return ONLY in the following JSON format:
{{
    "failure_summary": "...",
    "root_cause": "...",
    "suggested_improvements": [
        "...",
        "..."
    ],
    "example_fix": "...",
    "classes_to_check_init": ["ClassName1", "ClassName2", "..."]
}}
    """
            )

    def summarize_failure(self, code: str, test_result: "TestResult") -> str:
        """
        Use an LLM to summarize the test failure and provide suggestions for code improvement.

        Args:
            test_result: TestResult object containing test output and error message

        Returns:
            fail_summary: a concise summary suitable for reflection_generate
        """
        try:
            error_msg = getattr(test_result, 'error_message', '')

            if not error_msg:
                return {
                    "failure_summary": "No error information available.",
                    "root_cause": "",
                    "suggested_improvements": [],
                    "example_fix": "",
                    "classes_to_check_init": []
                }

            # Prepare English prompt for LLM
            code = code.replace("{", "{{").replace("}", "}}")
            error_msg = error_msg.replace("{", "{{").replace("}", "}}")
            
            prompt_text = self.prompt_text.format(code=code, error_msg=error_msg)

            # Call LLM
            response = self.llm_client.invoke(prompt_text)
            fail_summary = response.content.strip()
            if fail_summary.startswith("```json"):
                fail_summary = fail_summary[len("```json"):].lstrip()
            #  ```
            if fail_summary.endswith("```"):
                fail_summary = fail_summary[:-3].rstrip()
            print(fail_summary)
            result = json.loads(fail_summary)
            return result

        except Exception as e:
            print(f"⚠️ Enhanced summarize_failure error: {e}")
            # fallback
            return {
                "failure_summary": getattr(test_result, 'error_message', 'Unknown failure'),
                "root_cause": "",
                "suggested_improvements": [],
                "example_fix": "",
                "classes_to_check_init": []
            }

    def _resolve_dataset_name(self, data: dict) -> str:
        """Resolve dataset name for comment library lookup."""
        dataset = data.get("dataset") if isinstance(data, dict) else None
        if not dataset:
            dataset = getattr(self.code_retriever, "dataset", None)
        return dataset or "DevEval"

    def _resolve_functions_jsonl_path(self, data: dict) -> Optional[Path]:
        """Build `functions.jsonl` path by dataset and task metadata."""
        dataset = self._resolve_dataset_name(data)
        func_root = COMMENT_LIBRARY_ROOT / dataset / "Comment_Library"

        if dataset == "DevEval":
            project_key = data.get("project_path")
        else:
            project_key = data.get("repo_name") or data.get("project_path")

        if not project_key:
            print(f"⚠️ Missing project key when resolving functions.jsonl for dataset={dataset}")
            return None

        return func_root / project_key / "functions.jsonl"

    def reflect_and_regenerate(self, data: dict, function_name: str, input_code: str, class_definitions: str, test_result: TestResult, current_code: str, retry_count: int = 0, existing_imports: str = "", repo_data: Optional[dict] = None) -> ReflectionResult:
        """
        

        Args:
            function_name: /
            input_code: （）
            related_code: 
            test_result: ，
            retry_count: （）

        Returns:
            ReflectionResult 
        """
        #  related_code，
        failure_summary = self.summarize_failure(current_code, test_result)
        suggested_improvements = failure_summary["suggested_improvements"]
        classes_to_check_init = failure_summary["classes_to_check_init"]
        final_snippets = []
        node_ids = set()
        for task in suggested_improvements:
            #  retrieve ， dict， {"node_id": "...", "code": "...", ...}
            code_snippets = self.code_retriever.retrieve(
                data=data,
                task_description=task,
                top_k=1,
                repo_data=repo_data,
                seen_node_ids=node_ids
            )
            
            for snippet in code_snippets:
                #  node_id  code
                final_snippets.append({
                    "node_id": snippet.get("node_id"),
                    "code": snippet.get("code")
                })
                node_ids.add(snippet.get("node_id"))

        for cls_name in classes_to_check_init:
            #  __init__ 
            functions_path = self._resolve_functions_jsonl_path(data)
            if not functions_path:
                continue
            sig = self.code_retriever.retrieve_class_definition(cls_name, str(functions_path))
            if sig:
                print("[TestAgent] classes_to_check_init: \n" + sig.code)
                final_snippets.append({
                    "node_id": sig.node_id,
                    "code": sig.code
                })

        generated = self.code_generator.reflection_generate(
            function_name=function_name,
            input_code=input_code,
            class_definitions=class_definitions,
            pre_code=current_code,
            failed_test = test_result.failed_test,
            existing_imports=existing_imports,
            related_code=final_snippets,
            failure_summary=failure_summary,
            retries=retry_count
        )

        return ReflectionResult(code=generated.code, explanation=generated.explanation)
