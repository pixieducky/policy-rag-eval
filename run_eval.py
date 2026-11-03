import os
from dotenv import load_dotenv

# Load from the current folder
load_dotenv(dotenv_path=".env")

from langsmith import Client, evaluate
import requests

client = Client() # uses the env vars above

def ask(inputs):
    r = requests.post(
        "http://127.0.0.1:8000/chat",
        json={"question": inputs["question"]},
        timeout=180
    )
    r.raise_for_status()
    data = r.json()
    return{
        "answer": data["answer"],
        "input_tokens": data.get("prompt_tokens", 0),
        "output_tokens": data.get("completion_tokens", 0),
        "total_tokens": data.get("total_tokens", 0)
    }

if __name__ == "__main__":
    results = evaluate(
        ask,
        data="policy-questions",
        experiment_prefix="claude-haiku", # Change name for other models
        client=client  # use the same client that can read the dataset
    )
    print(results)