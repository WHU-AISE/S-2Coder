GPT_API_KEY = "your_api_key_here"
DeepSeek_API_KEY = "your_deepseek_api_key_here"

temperature = 0.6
top_p = 0.7
max_new_tokens = 512

summary_generate_model = "gpt-4o-mini"
embedding_generate_model = "text-embedding-3-small"

summary_prompt = """
Please generate a summary of no more than 100 words for the code above, only describing the overall functionality it implements.
"""