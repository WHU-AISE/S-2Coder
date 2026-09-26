"""
 - 
"""
import os
from dotenv import load_dotenv
from framework import CodeGenerationFramework


def main():
    # 
    load_dotenv()
    
    #  API Key
    if not os.getenv('OPENAI_API_KEY'):
        print(":  OPENAI_API_KEY ")
        print(" .env.example  .env， API Key")
        return
    
    # 
    framework = CodeGenerationFramework(max_retries=3)
    
    # 
    requirement = """
    ，：
    1. ，
    2. ，
    3. ，ID
    4. ，
    """
    
    print("...")
    print(f": {requirement}\n")
    
    # 
    results = framework.process_requirement(requirement)
    
    # 
    framework.export_results("generation_results.json")
    
    print("\n\n！")
    print(f": {sum(1 for r in results if r.test_passed)}/{len(results)} ")


if __name__ == "__main__":
    main()
