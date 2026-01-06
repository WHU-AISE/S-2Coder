import tree_sitter_python as pyts
import tree_sitter_java as javats
from tree_sitter import Language, Parser

from argparse import ArgumentParser
import json

import config as config
import os

from openai import OpenAI

class TS_Parser_Load:
    def __init__(self, language):
        self.TS_LANGUAGE, self.parser = self.get_parser(language)
        self.queries = self.get_queries(language)

    def get_parser(self,language):
        if language == "python":
            TS_LANGUAGE = Language(pyts.language())
            parser = Parser(TS_LANGUAGE)
        elif language == "java":
            TS_LANGUAGE = Language(javats.language())
            parser = Parser(TS_LANGUAGE)
        else:
            raise ValueError(f"Unsupported language: {language}")
        return TS_LANGUAGE, parser

    def get_queries(self, language):
        if language == "python":
            function_definition_query = """
                (function_definition name: (identifier) @function_name)@function_def
            """
        elif language == "java":
            function_definition_query = """
                (method_declaration name: (identifier) @function_name) @function_def
            """
        else:
            raise ValueError(f"Unsupported language: {language}")
        return self.TS_LANGUAGE.query(function_definition_query)

class Repo_Processor:
    def __init__(self, repo_root_dir, language):
        self.repo_root_dir = repo_root_dir
        self.language = language
        self.ts_parser_load = TS_Parser_Load(language)
        self.TS_LANGUAGE, self.parser = self.ts_parser_load.get_parser(language)
        self.all_files = get_all_files(language, repo_root_dir)
        self.client = self.load_models()

    def get_repo_name(self):
        return self.repo_root_dir.split('/')[-1]

    def get_all_files(self, language, repo_root_dir):
        all_files_abs = []
        end_tag = {
            "java": ".java",
            "python": ".py"
        }
        for root, _, files in os.walk(repo_root_dir):
            if 'test' in root.lower():
                continue
            for file in files:
                if file.endswith(end_tag[language]):
                    all_files_abs.append(os.path.join(root, file))
        return all_files_abs

    def get_relative_path(self, file_path):
        return file_path.replace(repo_root_dir + '/', '')

    def open_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        return code

    def get_file_tree(self, code):
        tree = self.parser.parse(bytes(code, "utf8"))
        return tree.root_node
        
    def get_functions_from_file(self, file_path):
        code = self.open_file(file_path)
        tree = self.get_file_tree(code)
        query = self.ts_parser_load.get_queries(self.language)
        captures = query.captures(tree)
        functions = []
        for node, capture_name in captures:
            if capture_name == "function_def":
                start_line = node.start_point[0]
                end_line = node.end_point[0]
                function_code = "\n".join(code.splitlines()[start_line:end_line+1])
                summary = self.generate_summary(function_code)
                summary_embedding = self.generate_embedding(summary)
                functions.append({
                    "code": function_code,
                    "start": start_line,
                    "end": end_line,
                    "summary": summary,
                    "summary_embedding": summary_embedding
                })
        return functions

    def get_all_functions(self):
        all_functions = []
        for file_path in self.all_files:
            functions = self.get_functions_from_file(file_path)
            relative_path = self.get_relative_path(file_path)
            all_functions.append({
                "file_path": relative_path,
                "functions": functions
            })
        json.dump(all_functions, open(f"retrieve_corpus/{self.get_repo_name()}.json", "w", encoding="utf-8"), indent=4)
        return all_functions

    def load_models(self):
        client = OpenAI(api_key=config.GPT_API_KEY)
        return client

    def generate_summary(self, code):
        outputs = self.client.chat.completions.create(
            model=config.summary_generate_model,
            messages=[
                {"role": "system", "content": "You are experienced in software development."},
                {"role": "user", "content": config.summary_prompt + "\n" + code}
            ],
            max_tokens=max_tokens,
            temperature=0,
            top_p=1
        )
        return outputs.choices[0].message.content

    def generate_embedding(self, text):
        return self.client.embeddings.create(input=text, model=config.embedding_generate_model).data[0].embedding

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--repo_path', type=str)
    parser.add_argument('--language', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_parser()
    repo_root_dir = args.repo_path
    language = args.language
    processor = Repo_Processor(repo_root_dir, language)
    processor.get_all_functions()
