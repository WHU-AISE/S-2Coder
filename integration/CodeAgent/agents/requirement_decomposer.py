from typing import List
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from llm_client import get_deepseek_client, get_openai_client


class Task(BaseModel):
    """"""
    id: str = Field(description="ID")
    name: str = Field(description="")
    description: str = Field(description="")


class RequirementDecomposer:
    """"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        self.llm = get_openai_client(model=model_name, temperature=temperature)
        self.prompt = PromptTemplate(
            input_variables=["requirement", "previous_feedback"],
            template="""
You are a professional requirements analyst. Please divide the given developer's natural language requirements, based on the principles of divide-and-conquer and code reuse, into the most relevant and useful sub-requirements to complete the code generation task. We will provide you with the developer's natural language requirements, parameter descriptions, and test cases for the code to be generated.

Division Guidelines:  
1. If the developer's requirements do not explicitly cover behaviors in the test cases, use the test cases to infer and supplement implicit requirements, ensuring the divided sub-requirements cover all necessary functionality.
2. Each sub-requirement represents the code summary of a function that may be invoked.
3. The granularity of decomposition should be appropriate: it should be specific enough to allow subsequent retrieval of corresponding implementations, but not so fine-grained that it introduces redundancy or unnecessary details.
4. When dividing requirements, adhere to the following software engineering principles. However, it does not need to be shown in sub-requirements. 
  - Clear Functional Goals: Requirements should clearly define the expected functionality, avoiding ambiguity, with unique and measurable goals.   
  - Boundary Conditions and Error Handling: Requirements should describe the function's behavior under boundary conditions (e.g., invalid inputs, null values) and specify error handling.    
  - Dependencies: Requirements should specify dependencies on other modules or functions, avoiding tight coupling, and clearly define interfaces for external dependencies. 
  - Maintainability and Extensibility: Requirements should account for potential future changes or extensions, avoiding code that is difficult to modify or extend. 
  - Scope Definition: Clearly define the function’s scope, avoiding side effects that impact external state, and ensuring the function remains independent.

Input Format:  
- Developer Requirements: {{Provide the specific natural language requirement description and the description of function parameters, including types and constraints here}}   

Return format:
[
  {{"id": "task_1", "name": "...", "description": "..."}},
  ...
]

Current Input:  
- Developer Requirements: {requirement}
{additional_feedback}
Based on the provided specific requirements, parameter description, and test cases, output the divided sub-requirements. 
"""
        )

        self.clarify_prompt = PromptTemplate(
            input_variables=["original_requirement", "task_list", "task_name", "task_description"],
            template="""
You are a professional requirements analyst.

Original Requirement:
{original_requirement}

The requirement was decomposed into several sub-requirements.
Existing Sub-Requirements:
{task_list}

The following sub-requirement needs clarification:
Task Name:{task_name}
Task Description:{task_description}

Problems with this sub-requirement:
- It may be unclear
- It may be inconsistent with the original requirement
- It may miss boundary conditions or edge cases

Important constraints:
1. Only clarify this task.
2. Do NOT introduce functionality already covered by other sub-requirements.
3. Maintain independence between tasks.
4. Do not merge multiple tasks into one.

Rewrite the task description to make it precise and complete.

Return ONLY the clarified description.

{{
"id": "task_1",
"name": "same task name",
"description": "clarified requirement description"
}}
"""
        )

        self.decompose_further_prompt = PromptTemplate(
            input_variables=[
                "original_requirement",
                "task_list",
                "task_name",
                "task_description"
            ],
            template="""
You are a professional requirements analyst.

Original Requirement:
{original_requirement}

The requirement has already been decomposed into several sub-requirements.
Existing Sub-Requirements:
{task_list}

The following sub-requirement needs to be further decomposed:
Task Name:{task_name}
Task Description:{task_description}

Your task is to further decompose this sub-requirement into exactly TWO smaller and clearer functional sub-requirements.

Important constraints:
1. Only decompose THIS task.
2. Do NOT introduce functionality already covered by other sub-requirements.
3. Maintain independence between tasks.
4. You MUST produce exactly TWO sub-requirements.
5. Do not output more or fewer than two items.
6. Each sub-requirement must describe a concrete functional behavior that can be implemented in code.
7. The two sub-requirements together should fully cover the functionality of the original task without overlap.

Return format:

[
  {{"id": "subtask_1", "name": "...", "description": "..."}},
  {{"id": "subtask_2", "name": "...", "description": "..."}}
]
"""
        )
    
    def decompose(self, requirement: str, previous_feedback: str = "") -> List[Task]:
        """
        ，
        
        Args:
            requirement: 
            previous_feedback: 
            
        Returns:
            
        """
        #  prompt
        if previous_feedback:
            additional_feedback = f"There are previous feedback or suggestions for improvement, please incorporate them into the task decomposition process: {previous_feedback}"
        else:
            additional_feedback = ""

        prompt_text = self.prompt.format(requirement=requirement, previous_feedback=previous_feedback, additional_feedback=additional_feedback)
        
        response = self.llm.invoke(prompt_text)
        
        try:
            import json
            import re
            # JSON
            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
            if json_match:
                tasks_data = json.loads(json_match.group())
                return [Task(**task) for task in tasks_data]
            print(f"[WARN] decompose no JSON array matched; fallback will be used. preview: {str(response.content)[:500]}")
        except Exception as e:
            print(f"Error parsing response: {e}")

        # fallback: parse failed or no JSON array detected
        return [
            Task(
                id="task_1",
                name="",
                description=requirement
            )
        ]

    def clarify(self, original_requirement: str, task_list: List[str], name: str, description: str) -> Task:
        """
        

        Args:
            original_requirement: 
            task: 

        Returns:
             Task
        """

        prompt_text = self.clarify_prompt.format(
            original_requirement=original_requirement,
            task_list=task_list,
            task_name=name,
            task_description=description
        )

        response = self.llm.invoke(prompt_text)

        try:
            import json
            import re

            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                task_data = json.loads(json_match.group())

                return Task(**task_data)

        except Exception as e:
            print(f"Clarify parse error: {e}")

        # fallback
        return Task(
            id="task_1",
            name="",
            description=original_requirement
        )
    
    def decompose_further(
        self,
        original_requirement: str,
        task_list: List[str],
        name: str,
        description: str
    ) -> List[Task]:
        """
        
        """

        prompt_text = self.decompose_further_prompt.format(
            original_requirement=original_requirement,
            task_list=task_list,
            task_name=name,
            task_description=description
        )

        response = self.llm.invoke(prompt_text)

        try:
            import json
            import re

            json_match = re.search(r'\[.*\]', response.content, re.DOTALL)

            if json_match:
                tasks_data = json.loads(json_match.group())

                return [Task(**task) for task in tasks_data]

        except Exception as e:
            print(f"Decompose further parse error: {e}")

        # fallback
        return [
            Task(
                id="task_1",
                name=name,
                description=description
            )
        ]
