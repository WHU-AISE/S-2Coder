import requests
from typing import List, Dict
from pydantic import BaseModel, Field
from llm_client import get_deepseek_client, get_openai_client
from bs4 import BeautifulSoup
import re

class LibraryRecommendation(BaseModel):
    """"""
    name: str = Field(description="")
    description: str = Field(description="")
    url: str = Field(description="PyPI")
    summary: str = Field(description="")
    functions: List[str] = Field(description="")


class SearchAgent:
    """ Agent，Python"""

    def __init__(self, search_api_url: str = "https://pypi.org/pypi", model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        """
         Agent
        
        Args:
            search_api_url: PyPIAPI URL，
        """
        self.search_api_url = search_api_url
        self.llm = get_openai_client(model=model_name, temperature=temperature)

    def search_libraries(self, task_description: str) -> List[LibraryRecommendation]:
        """
        
        
        Args:
            task_description: 
            
        Returns:
            
        """
        # GPT
        search_keywords = self.generate_search_keywords(task_description)

        # PyPI
        search_results = self.search_pypi(search_keywords)
        
        return search_results, search_keywords

    def generate_search_keywords(self, task_description: str) -> str:
        """
        
        
        Args:
            task_description: 
            
        Returns:
            
        """
        try:
            # Construct the prompt for GPT
            prompt = f"""
                You are an expert software architect. Analyze the Python task description below and identify 3 generic technical keywords. 

                GOAL: Find external, third-party libraries (on PyPI) or standard frameworks that implement these functionalities.

                CRITICAL RULES:
                1. DO NOT use internal class names or project-specific terms.
                2. USE functional, industry-standard terms.
                3. FOCUS on the underlying technology (e.g., 'data validation', 'deep reflection', 'AST parsing').

                Task description: {task_description}

                Separate the 3 keywords with '\n' (Return ONLY the keywords):
                """
            # Call OpenAI API to generate keywords
            response = self.llm.invoke(prompt)
            
            # Get the generated keywords
            keywords = response.content.split('\n')
            print("search keywords: " + "\n".join(keywords))
            return keywords
        except Exception as e:
            print(f"Error generating keywords: {e}")
            return ""

    def search_pypi(self, queries: List[str]) -> List[LibraryRecommendation]:
        """
        PyPI API
        
        Args:
            queries: 
        
        Returns:
            
        """
        all_libraries = []
        
        # 
        for query in queries:
            url = f"{self.search_api_url}/{query}/json"
            response = requests.get(url)

            # 
            if response.status_code == 200:
                data = response.json()

                #  'info' 
                library_info = data['info']
                #  BeautifulSoup  HTML 
                soup = BeautifulSoup(library_info.get('description', 'No description available'), 'html.parser')

                # 
                text_description = soup.get_text()

                # 
                summary, functions = self.summarize_library_info(text_description)
                
                #  LibraryRecommendation 
                libraries = LibraryRecommendation(
                    name=library_info['name'],
                    description=text_description,
                    url=library_info.get('package_url', ''),
                    summary=summary,
                    functions=functions
                )
                all_libraries.append(libraries)
            else:
                # ，
                continue
        
        return all_libraries

    def summarize_library_info(self, description: str) -> (str, List[str]): # type: ignore
        """
        ，
        
        Args:
            description: 
            
        Returns:
            
        """
        try:
            #  prompt 
            prompt = f"""
Please summarize the following library description into a short sentence that explains what the library does. Also, list the main functions or methods that can be used with this library in the format below:

Description: {description}

Output format:
Summary: <summary>
Functions: <function1>, <function2>, <function3>, ...

Do not include any other information.
            """
            #  API
            response = self.llm.invoke(prompt)

            # 
            result = response.content.strip()

            #  summary  functions
            summary = ""
            functions = []

            # 
            if result.startswith("Summary:"):
                #  summary  functions 
                parts = result.split('Functions:')
                summary = parts[0].replace("Summary:", "").strip()

                #  functions 
                if len(parts) > 1:
                    functions_text = parts[1].strip()
                    functions = [func.strip() for func in functions_text.split(',')]

            return summary, functions
        except Exception as e:
            print(f"Error summarizing library info: {e}")
            return "Summary not available", []
