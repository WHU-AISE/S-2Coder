"""
 - 
"""
from typing import Dict, Any

# LLM 
LLM_CONFIG = {
    "model": "gpt-4",  # : gpt-4, gpt-3.5-turbo
    "temperature": {
        "decomposer": 0.7,      #  - 
        "retriever": 0.3,       #  - 
        "analyzer": 0.3,        #  - 
        "generator": 0.7,       #  - 
        "test_agent": 0.5,      #  - 
    },
    "max_tokens": {
        "decomposer": 2000,
        "generator": 3000,
        "test_agent": 2000,
    }
}

# 
FRAMEWORK_CONFIG = {
    "max_retries": 3,           # 
    "timeout": 5,               # （）
    "code_retrieval_top_k": 5,  # 
    "dependency_analysis_depth": 3,  # 
}

# 
GENERATION_CONFIG = {
    "include_docstrings": True,  # 
    "include_type_hints": True,  # 
    "include_error_handling": True,  # 
    "coding_style": "pep8",     # 
}

# 
TEST_CONFIG = {
    "run_syntax_check": True,       # 
    "run_execution_test": True,     # 
    "run_quality_check": True,      # 
    "min_confidence_score": 0.6,    # 
}

# 
LOG_CONFIG = {
    "log_level": "INFO",           # 
    "log_to_file": True,           # 
    "log_file": "agent_logs.txt",  # 
    "verbose": True,               # 
}

# 
OUTPUT_CONFIG = {
    "output_format": "json",       # 
    "output_file": "generation_results.json",  # 
    "include_code": True,          # 
    "include_logs": True,          # 
}


def get_config() -> Dict[str, Any]:
    """"""
    return {
        "llm": LLM_CONFIG,
        "framework": FRAMEWORK_CONFIG,
        "generation": GENERATION_CONFIG,
        "test": TEST_CONFIG,
        "log": LOG_CONFIG,
        "output": OUTPUT_CONFIG,
    }


def get_llm_config(agent_type: str) -> Dict[str, Any]:
    """ Agent  LLM """
    config = {
        "model": LLM_CONFIG["model"],
        "temperature": LLM_CONFIG["temperature"].get(agent_type, 0.5),
    }
    if agent_type in LLM_CONFIG["max_tokens"]:
        config["max_tokens"] = LLM_CONFIG["max_tokens"][agent_type]
    return config
