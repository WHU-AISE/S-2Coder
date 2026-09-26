"""
 -  DevEval 
Quick test script with the last item from DevEval dataset
"""

import json
import sys
import os
from pathlib import Path

# 
try:
    from dotenv import load_dotenv
    #  .env 
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)
except ImportError as e:
    print(f"⚠️  {e}")

def test_requirement():
    """"""
    
    # 
    print("📦  DevEval ...")
    with open("dataset/DevEval_dataset_all_infor_172.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    last_item = data[0]
    
    # 
    repo_name = last_item.get('repo_name')
    function_name = last_item.get('function_name')
    all_requirement = last_item.get('all_requirement')
    
    print(f"✓ ")
    print()
    
    # 
    print("=" * 70)
    print("🎯 ")
    print("=" * 70)
    print(f": {repo_name}")
    print(f": {function_name}")
    print(f": {last_item.get('language')}")
    print()
    
    # 
    print("=" * 70)
    print("📝  (all_requirement)")
    print("=" * 70)
    print(all_requirement)
    print()
    
    # 
    print("=" * 70)
    print("💻 ")
    print("=" * 70)
    input_code = last_item.get('input_code')
    if input_code:
        #  30 
        lines = input_code.split('\n')
        for i, line in enumerate(lines[:30], 1):
            print(f"{i:3d}: {line}")
        if len(lines) > 30:
            print(f"... ( {len(lines)} )")
    print()
    
    return last_item


def run_framework(task_info):
    """"""
    from framework import CodeGenerationFramework
    from llm_client import get_openai_client
    import numpy as np
    
    print("=" * 70)
    print("🚀 ")
    print("=" * 70)
    print()
    
    #  API 
    try:
        client = get_openai_client()
    except ValueError as e:
        print(f"⚠️  {e}")
    print()
    
    framework = CodeGenerationFramework()
    
    requirement = task_info['all_requirement']
    print(f": {requirement}")
    print()
    
    # 
    
    #  embedding ， numpy 
    if task_info['all_requirement_embedding'] is not None:
        if isinstance(task_info['all_requirement_embedding'], (list, tuple)):
            task_info['all_requirement_embedding'] = np.array(task_info['all_requirement_embedding'])
    
    try:
        results = framework.process_requirement(requirement, task_data=task_info)
        
        print()
        print("=" * 70)
        print("✅ ！")
        print("=" * 70)
        
        # 
        if results:
            print()
            print("📊 :")
            print(f"  - : {len(results.get('tasks', []))}")
            print(f"  - : {len(results.get('generated_code', ''))} ")
            
            # 
            generated_code = results.get('generated_code')
            if generated_code:
                print()
                print("=" * 70)
                print("📄 ")
                print("=" * 70)
                lines = generated_code.split('\n')
                for i, line in enumerate(lines[:40], 1):
                    print(f"{i:3d}: {line}")
                if len(lines) > 40:
                    print(f"... ( {len(lines)} )")
        
        return results
        
    except Exception as e:
        print(f"❌ : {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    try:
        # 
        task_info = test_requirement()
        
        # 
        results = run_framework(task_info)
        
        # 
        if results:
            output_file = "dataset_example_results.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'task_info': task_info["all_requirement"],
                    'results': results
                }, f, indent=2, ensure_ascii=False)
            print()
            print(f"💾 : {output_file}")
        
    except KeyboardInterrupt:
        print("\n⚠️  ")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
