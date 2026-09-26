"""
 Agent - 
"""
from typing import List, Optional
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from llm_client import get_openai_client


class GeneratedCode(BaseModel):
    """"""
    code: str = Field(description="")
    explanation: str = Field(description="")


class CodeGenerator:
    """"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", **kwargs):
        """
        
        
        Args:
            model_name: 
            **kwargs: （api_base, temperature ）
        """
        #  llm_client 
        self.llm = get_openai_client(model=model_name, **kwargs)
        self.generation_history = []
        self.memory: dict[str, dict] = {}  # key: function_name, value: info
    
    def generate(
        self,
        function_name: str,
        input_code: str,
        original_requirement: str,
        related_code: str = "",
        selected_libraries: str = "",
        existing_imports: str = "",
        retries: int = 0
    ) -> GeneratedCode:
        """
        
        
        Args:
            task_description: 
            related_code: 
            dependencies: 
            retries: 
            
        Returns:
            
         """
        prompt = PromptTemplate(
            input_variables=["function_name", "original_requirement", "libraries", "related_code", "existing_imports", "input_code", "retry_note"],
            template="""
You are a professional AI code generation assistant. Your task is to generate high-quality Python code for a single function.

Follow these instructions carefully:

---

### 1. Implementation Scope
- Only implement the function {function_name} as provided in the Input Code.
- The implementation must strictly follow the original requirement, which is described as follows:
  - Original Requirement: 
  {original_requirement}
- Do not modify or generate any code outside this function (no extra classes, functions, imports, or variables).
- Do not add extra imports or helper functions outside this function unless absolutely necessary. Use only the provided function signature.
- Follow exactly the parameter names, order, and logic structure in the input code.

---

### 2. References: Similar Code Snippets
- The retrieved code snippets illustrate patterns and approaches for solving similar tasks.
- Use them as high-level guidance only—follow the design patterns or approaches, but do not copy code verbatim or change the input function signature or parameters.
- The following are the reference code snippets:
{related_code}

---

### 3. Input Code
- Use the following input code as the base. Implement only the specified function.
Input Code:
{input_code}

---

### 4. Retry Note
{retry_note}

---

### 6. Output Format
- Return only the completed Python function code. Do not include any explanations, markdown, or extra text outside the code block.
- Wrap the code exactly with triple backticks and specify Python:

```python
# Generated code
```
"""
        )
        
#         prompt = PromptTemplate(
#             input_variables=["function_name", "original_requirement", "libraries", "related_code", "existing_imports", "input_code", "retry_note"],
#             template="""
# You are a professional AI code generation assistant. Your task is to generate high-quality Python code for a single function.

# Follow these instructions carefully:

# ---

# ### 1. Implementation Scope
# - Only implement the function {function_name} as provided in the Input Code.
# - The implementation must strictly follow the original requirement, which is described as follows:
#   - Original Requirement: 
#   {original_requirement}
# - Do not modify or generate any code outside this function (no extra classes, functions, imports, or variables).
# - Do not add extra imports or helper functions outside this function unless absolutely necessary. Use only the provided function signature.
# - Follow exactly the parameter names, order, and logic structure in the input code.

# ---

# ### 2. References: Libraries
# - You may use the following third-party libraries if needed:
# - The following are the reference code third-party libraries:
# {libraries}

# ---

# ### 3. References: Similar Code Snippets
# - The retrieved code snippets illustrate patterns and approaches for solving similar tasks.
# - Use them as high-level guidance only—follow the design patterns or approaches, but do not copy code verbatim or change the input function signature or parameters.
# - The following are the reference code snippets:
# {related_code}

# ---

# ### 4. Retry Note
# {retry_note}

# ---

# ### 5. Input Code
# - Use the following input code as the base. Complete only the specified function within it. 
# - Ensure the implementation strictly adheres to the parameter names and logic structure provided. 
# - Avoid introducing unrelated variables or unnecessary abstractions.
# Input Code:
# {input_code}
# ```
# """
#         )
        
        retry_note = f"This is the {retries}th regeneration, please improve the previous implementation." if retries > 0 else "This is the first generation."
        
        prompt_text = prompt.format(
            function_name=function_name,
            original_requirement = original_requirement,
            related_code=related_code or "None",
            libraries = selected_libraries or "None",
            # existing_imports=existing_imports or "None",
            input_code=input_code,
            retry_note=retry_note
        )
        print(f"prompt_text:\n{prompt_text}")
        
        response = self.llm.invoke(prompt_text)
        
        try:
            import re
            # 
            code_match = re.search(r'```python\n(.*?)\n```', response.content, re.DOTALL)
            code = code_match.group(1) if code_match else response.content
            
            generated = GeneratedCode(
                code=code,
                explanation=response.content,
            )
            self.generation_history.append({
                "task": input_code,
                "code": code,
                "retry": retries
            })
                    
            self.memory[function_name] = {
                "related_code": related_code or "",
                "last_generated": code
            }
            return prompt_text, generated
        except Exception as e:
            print(f"Error parsing generated code: {e}")
            return prompt_text, GeneratedCode(
                code="# Error generating code",
                explanation=response.content,
                imports=[]
            )

    
    def get_history(self):
        """"""
        return self.generation_history


    def reflection_generate(
        self,
        function_name: str,
        input_code: str,
        class_definitions: str,
        existing_imports: str,
        pre_code: str,  # 
        failed_test: str,
        related_code: str,
        failure_summary: str,
        retries: int = 0
    ) -> GeneratedCode:

        # 1. 
        if function_name not in self.memory:
            self.memory[function_name] = {
                "history": [], 
                "related_code": related_code
            }

        if "history" not in self.memory[function_name]:
            self.memory[function_name]["history"] = []

        # “(pre_code)”
        #  LLM  history_context 
        self.memory[function_name]["history"].append({
            "code": pre_code,
            "error": failure_summary or "Unknown error"
        })

        # 
        self.merge_related_code(function_name, related_code)
        
        mem_entry = self.memory[function_name]
        history = mem_entry.get("history", [])

        # 2. “” LLM
        history_context = ""
        if history:
            history_context = "\n--- Historical Attempts ---\n"
            for i, attempt in enumerate(history):
                history_context += f"Attempt #{i+1}:\n"
                history_context += f"[Code Snippet]:\n{attempt['code']}\n"
                history_context += f"[Error Summary]:\n{attempt['error']}\n"
                history_context += "---------------------------\n"
        else:
            history_context = "This is the first attempt."

        retry_note = f"This is the {retries}th regeneration. Please review ALL previous failures to avoid repeating the same mistakes." if retries > 0 else ""
        related_code_list = list(self.memory[function_name]["related_code_dict"].values())
        related_code_str = "\n".join([f"# node_id: {c['node_id']}\n{c['code']}" for c in related_code_list])

        # 3.  Prompt
        prompt = PromptTemplate(
            input_variables=["function_name", "related_code", "input_code", "class_definitions", "history_context", "existing_imports", "retry_note", "failed_test"],
            template="""
    You are a professional AI code generation assistant. You are tasked with reflecting on the errors from ALL previous attempts.

    1. History of Failures
    {history_context}

    2. Failed Test Function (Current Target):
    {failed_test}

    3. Reference: Relevant Code Snippets
    {related_code}

    4. Base Input Code:
    {input_code}

    5. Custom Class Definitions:
    The following classes are used in the parameters or return types of the target function. Use these as a reference for object attributes and methods:
    {class_definitions}

    {retry_note}

    Compare the previous attempts, identify why they failed, and generate a new, corrected version. 
    Return format, wrap the code with ```python and ```:
    ```python
    # Generated code
    """
        )
    #     # 3.  Prompt
    #     prompt = PromptTemplate(
    #         input_variables=["function_name", "related_code", "input_code", "class_definitions", "history_context", "existing_imports", "retry_note", "failed_test"],
    #         template="""
    # You are a professional AI code generation assistant. You are tasked with reflecting on the errors from ALL previous attempts.

    # 1. History of Failures
    # {history_context}

    # 2. Failed Test Function (Current Target):
    # {failed_test}

    # 3. Reference: Relevant Code Snippets
    # {related_code}

    # 4. Custom Class Definitions:
    # The following classes are used in the parameters or return types of the target function. Use these as a reference for object attributes and methods:
    # {class_definitions}

    # {retry_note}

    # 5. Input Code
    # - Use the following input code as the base. Complete only the specified function within it. 
    # - Ensure the implementation strictly adheres to the parameter names and logic structure provided. 
    # - Avoid introducing unrelated variables or unnecessary abstractions.
    # Input Code:
    # {input_code}
    # """
    #     )

        prompt_text = prompt.format(
            function_name=function_name,
            related_code=related_code_str,
            input_code=input_code,
            class_definitions=class_definitions,
            history_context=history_context,
            existing_imports=existing_imports,
            retry_note=retry_note,
            failed_test=failed_test
        )

        # 4.  LLM 
        response = self.llm.invoke(prompt_text, temperature=0.0)

        # 
        import re
        content = response.content
        # 
        code_match = re.search(r'```python\s*(.*?)\s*```', content, re.DOTALL)
        
        if code_match:
            new_code = code_match.group(1).strip()
        else:
            # ：，
            alt_match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
            new_code = alt_match.group(1).strip() if alt_match else content.strip()

        # 5. “”
        self.memory[function_name]["last_generated"] = new_code
        self.memory[function_name]["related_code"] = related_code

        # ： append  history，
        # ， new_code  pre_code  append

        return GeneratedCode(code=new_code, explanation=response.content)
    
    def merge_related_code(self, function_name: str, new_related_code: List[dict]):
        """
         new_related_code  memory  related_code_dict ，node_id 
        new_related_code  node_id
        """
        if function_name not in self.memory:
            self.memory[function_name] = {"history": [], "related_code_dict": {}}

        if "related_code_dict" not in self.memory[function_name]:
            self.memory[function_name]["related_code_dict"] = {}

        for code_info in new_related_code:
            node_id = code_info.get("node_id")
            if not node_id:
                continue  #  node_id 
            #  node_id ，
            self.memory[function_name]["related_code_dict"][node_id] = code_info
