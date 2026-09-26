import json
from typing import List, Dict
from pydantic import BaseModel, Field
from llm_client import get_deepseek_client, get_openai_client
from langchain_core.prompts import PromptTemplate


class RequirementReflectionResult(BaseModel):
    """，"""
    overall_score: float = Field(description=" (0-5)")
    need_human_intervention: bool = Field(description="")
    
    # 
    clarity_score: float = Field(description=" (0-5)")
    feasibility_score: float = Field(description=" (0-5)")
    completeness_score: float = Field(description=" (0-5)")
    comment: str = Field(description="")
    
    @classmethod
    def from_task_scores(cls, overall_score: float, need_human_intervention: bool, clarity_score: float, feasibility_score: float, completeness_score: float, comment: str):
        return cls(
            overall_score=overall_score,
            need_human_intervention=need_human_intervention,
            clarity_score=clarity_score,
            feasibility_score=feasibility_score,
            completeness_score=completeness_score,
            comment=comment
        )


class RequirementReflector:
    """"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0, decomposer=None, max_retries: int = 3):
        self.llm = get_openai_client(model=model_name, temperature=temperature)
        self.decomposer = decomposer  # RequirementDecomposer 
        self.max_retries = max_retries  # 
        self.prompt = PromptTemplate(
            input_variables=["tasks_json"],
            template="""
You are an expert requirements reviewer. 
Please evaluate the following list of tasks that were decomposed from a requirement:

Tasks:
{tasks_json}

For the entire set of tasks, provide an overall reflection including:
- clarity_score: an integer from 0 to 5, indicating how clearly the task is defined
- feasibility_score: an integer from 0 to 5, indicating how feasible it is to implement
- completeness_score: an integer from 0 to 5, indicating how complete the task is
- comment: suggestions for improvement (use concise and actionable feedback)

Scoring Criteria:
- 5: Fully meets the requirement; tasks are clear, feasible, and complete.
- 4: Mostly good; minor missing details or small ambiguity.
- 3: Some important details missing; may require review or clarification.
- 2: Many unclear or infeasible parts; significant gaps in completeness.
- 1: Task is poorly defined; almost unusable.
- 0: Completely wrong or irrelevant.

**Important Instructions:**
1. Use only integers from 0 to 5.
2. Assign low scores (0-3) if there are significant issues; do not default to high scores.
3. Provide concise and actionable comments for improvement.
4. Return the result in strict JSON format, like this:

{{
  "clarity_score": 3,
  "feasibility_score": 2,
  "completeness_score": 4,
  "comment": "The tasks are mostly clear, but task_2 is missing key implementation details. Consider specifying data structures and expected outputs more clearly."
}}
"""
        )

    def reflect(self, requirement: str, tasks: List[Dict], socketio=None) -> str:
        """
        ，3
        
        Args:
            tasks:  RequirementDecomposer （dict）
            
        Returns:
            JSON
        """
        import json
        retries = 0
        task_result = {}
        while retries < self.max_retries:
            # 
            tasks_dict_list = [task.model_dump() if hasattr(task, "model_dump") else task for task in tasks]
            prompt_text = self.prompt.format(tasks_json=json.dumps(tasks_dict_list, indent=2))
            response = self.llm.invoke(prompt_text)
            
            try:
                import re
                # JSON
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    reflection_data = json.loads(json_match.group())

                    # 
                    clarity_score = reflection_data["clarity_score"]
                    feasibility_score = reflection_data["feasibility_score"]
                    completeness_score = reflection_data["completeness_score"]
                    overall_score = (clarity_score + feasibility_score + completeness_score) / 3
                    comment = reflection_data["comment"]

                    # 
                    if overall_score <= 3:
                        reflection_result = RequirementReflectionResult.from_task_scores(
                            overall_score=overall_score,
                            need_human_intervention=False,
                            clarity_score=clarity_score,
                            feasibility_score=feasibility_score,
                            completeness_score=completeness_score,
                            comment=comment
                        )

                        print(""+str(retries)+"："+str(overall_score))
                        if socketio:
                            socketio.emit('task_reflection', {'reflection': reflection_result.model_dump()})

                        #  RequirementDecomposer 
                        print(f"Attempt {retries + 1}: Score is below 3, re-running RequirementDecomposer for task decomposition...")
                        #  RequirementDecomposer 
                        tasks = self.decomposer.decompose(requirement, comment)  #  decomposer 
                        task_result = {
                            'original_requirement': requirement,
                            'decomposed_tasks': [{'id': task.id, 'name': task.name, 'description': task.description} for task in tasks]
                        }
                        retries += 1
                        
                        # 
                        if socketio:
                            socketio.emit('task_divided', {'tasks': task_result})

                        continue  # 

                    elif 3 < overall_score <= 4:
                        # 34，
                        reflection_result = RequirementReflectionResult.from_task_scores(
                            overall_score=overall_score,
                            need_human_intervention=True,
                            clarity_score=clarity_score,
                            feasibility_score=feasibility_score,
                            completeness_score=completeness_score,
                            comment=comment
                        )
                        # 
                        if socketio:
                            socketio.emit('task_reflection', {'reflection': reflection_result.model_dump()})

                        break
                    else:
                        # 4，
                        reflection_result = RequirementReflectionResult.from_task_scores(
                            overall_score=overall_score,
                            need_human_intervention=False,
                            clarity_score=clarity_score,
                            feasibility_score=feasibility_score,
                            completeness_score=completeness_score,
                            comment=comment
                        )
                        # 
                        if socketio:
                            socketio.emit('task_reflection', {'reflection': reflection_result.model_dump()})
                        break

            except Exception as e:
                print(f"Error parsing reflection response: {e}")
                # 
                reflection_result = RequirementReflectionResult.from_task_scores(
                    overall_score=5.0,
                    need_human_intervention=True,
                    clarity_score=5.0,
                    feasibility_score=5.0,
                    completeness_score=5.0,
                    comment="，"
                )
                # 
                if socketio:
                    socketio.emit('task_reflection', {'reflection': reflection_result.model_dump()})
                break

        return task_result, reflection_result.model_dump()
