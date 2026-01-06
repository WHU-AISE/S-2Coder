import numpy as np
import config as config
from openai import OpenAI
from data_load import load_dataset
from utils import load_json
from argparse import ArgumentParser

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def generate_embedding(client, text):
        return client.embeddings.create(input=text, model=config.embedding_generate_model).data[0].embedding

def retrieve_similar_functions(req, corpus_path, dataset_file_path, dataset_function_name, dataset_line, top_k):
    corpus_json = load_json(corpus_path)
    client = OpenAI(api_key=config.GPT_API_KEY)
    req_emb = generate_embedding(client, req)
    similarities = []
    for file_data in corpus_json:
        file_path = file_data["file_path"]
        file_functions = file_data["functions"]
        for func in file_functions:
            if file_path == dataset_file_path and func["name"] == dataset_function_name and func["start"] <= dataset_line and dataset_line <= func["end"]:
                continue
            func_emb = func["summary_embedding"]
            sim = cosine_similarity(req_emb, func_emb)
            similarities.append({
                "file_path": file_path,
                "function_code": func["code"],
                "start_line": func["start"],
                "end_line": func["end"],
                "similarity": sim
            })
    similarities = sorted(similarities, key=lambda x: x["similarity"], reverse=True)
    return similarities[:top_k]

def retrieve_dataset(dataset_name, language, top_k):
    dataset = load_dataset(dataset_name, language)
    similarity_results = []
    for data in dataset:
        req = data["requirement"]
        file_path = data["file_path"]
        function_name = data["function_name"]
        line = data["line"]
        similar_functions = retrieve_similar_functions(
            req,
            corpus_path=f"retrieve_corpus/{data['repo']}.json",
            dataset_file_path=file_path,
            dataset_function_name=function_name,
            dataset_line=line,
            top_k=top_k
        )
        similarity_results.append(similar_functions)
    json.dump(similarity_results, open(f"retrieve_results/{dataset_name}_{language}_{top_k}.json", "w", encoding="utf-8"), indent=4)


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--language', type=str)
    parser.add_argument('--top_k', type=int, default=5)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_parser()
    dataset_name = args.dataset_name
    language = args.language
    top_k = args.top_k
    retrieve_dataset(dataset_name, language, top_k)