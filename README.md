# S<sup>2</sup>Coder
This repository contains the implementation of S<sup>2</sup>Coder, a syntax-constrained and structure-augmented code generation framework.

## Datasets
We use the two open source repository-level datasets:

+ [CoderEval](https://github.com/CoderEval/CoderEval) (Python and Java)
+ [DevEval](https://github.com/seketeam/DevEval) (Python)

## Baselines
We selected several representative and SOTA repository-level code generation approaches as baselines.

+ Zero-shot

Generate code solely based on the developer’s requirement without incorporating any retrieval.

+ SimpleRAG

Retrieve similar code snippets based on fixed-size overlapping slide window chunks and append them to the input prompt for code generation.

+ [RepoCoder](https://github.com/microsoft/CodeT/tree/main/RepoCoder)

Apply an iterative retrieval-generation pipeline to generate code at various levels of granularity.

+ [CodeGRAG](https://github.com/KounianhuaDu/CodeGRAG)

Propose a hard meta-graph prompt template to utilize the graphical view of code blocks based on the control flow and data flow. 

+ [DRACO]( https://github.com/nju-websoft/DraCo)

Leverage dataflow analysis to enhance the input prompt by integrating structurally relevant background knowledge.

+ [RLCoder]([https://github.com/DeepSoftwareAnalytics/RLCoder)

Employ a reinforcement learning-based framework that enables the retriever to learn useful content without labeled data for code generation.

## Setup
### Environment Requirements
```yaml
accelerate==1.11.0
anyio==4.11.0
astor==0.8.1
javalang==0.13.0
networkx==3.4.2
numpy==2.2.6
openai==2.7.1
safetensors==0.6.2
tiktoken==0.12.0
tokenizers==0.22.1
torch==2.9.0
torchvision==0.24.0
tqdm==4.67.1
transformers==4.57.1
tree-sitter==0.25.2
tree-sitter-java==0.23.5
tree-sitter-python==0.25.0
```


### Get Start
#### Prepare Data and Model
Following the instruction in [CoderEval](https://github.com/CoderEval/CoderEval) and [DevEval](https://github.com/seketeam/DevEval), download all repositories to the `./repositories` directory, downdload dataset file to the `./dataset` directory.

Load [Qwen-2.5-Coder](https://huggingface.co/collections/Qwen/qwen25-coder) (7B, 3B, 1.5B, and 0.5B) and [DeepSeek-Coder](https://huggingface.co/collections/deepseek-ai/deepseek-coder) (6.7B and 1.3B) from Hugging Face

#### Set API_key 
Set API_Key for GPT and DeepSeek in `config.py`

#### Build Retrieval Corpus
`./build_corpus.py --repo_path repo_path --language language(Python/Java)`

#### Retrieve
`./retrieve.py --dataset_name dataset_name --language language --top_k top_k`

#### Generate
`./main.py --dataset_name dataset_name --language language --top_k top_k --checker True --sr True -- coarse_flag True --fine_flag True --model_name model_name --n n`

#### Ablation Study
+ w/o syntactic checker

   `./main.py --dataset_name dataset_name --language language --top_k top_k --checker False --sr True -- coarse_flag True --fine_flag True --model_name model_name --n n`

+ w/o augmentation

   `./main.py --dataset_name dataset_name --language language --top_k top_k --checker True --sr False -- coarse_flag False --fine_flag False --model_name model_name --n n`

+ w/o key syntactic structure

   `./main.py --dataset_name dataset_name --language language --top_k top_k --checker True --sr True -- coarse_flag False --fine_flag False--model_name model_name --n n `

+ w/o coarse pruning
   
   `./main.py --dataset_name dataset_name --language language --top_k top_k --checker True --sr True -- coarse_flag False --fine_flag True --model_name model_name --n n`


+ w/o fine pruning

   `./main.py --dataset_name dataset_name --language language --top_k top_k --checker True --sr True -- coarse_flag True --fine_flag False --model_name model_name --n n`

