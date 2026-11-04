# policy-rag-eval

A local RAG chatbot over a company policy file, with a small evaluation set and a human review report.

It answers from retrieved chunks. If the documents do not cover the question, the API says “I don’t know” and returns no sources.

## What it shows
- Ingest a policy text file into Chroma
- Answer from retrieved chunks (local Ollama or Claude)
- Hide sources on “I don’t know” and “I can’t do that”
- 11-question dataset in LangSmith
- Human pass/fail on adversarial cases
- Side-by-side comparison: llama3.2:3b vs Claude Haiku 5.5
- Token and cost visibility on Claude traces

Human review is the quality gate. DeepEval on the 3B model is not used for final results.

## Demo
### Policy answer

![Policy answer](docs/chat-policy.png)

### I don't know 

![I don't know, no sources](docs/chat-idk.gif)

### LangSmith comparison (llama vs Haiku)

![Experiment comparison](docs/llama_haiku_comparison.jpg)

### Claude tokens and cost (Tracing)

![Tracing tokens and cost](docs/Tracing_tokens_cost.jpg)

## Results (summary)
| Model | Critical cases | Notes |
|---|---|---|
| llama3.2:3b | 6/11 pass | Phone reimbursement still inverted. Ignore/CEO often answered “I don’t know” instead of refusing |
| Claude Haiku 5.5 | 11/11 pass | Corrected all previously failed answers answers were longer (higher output tokens) |

Token counts and dollar cost for Haiku are visible in LangSmith Tracing. Local Ollama cost is $0.

Full notes: [eval_report.md](eval_report.md)

## Stack
- FastAPI
- Chroma and `all-MiniLM-L6-v2` embeddings
- Ollama `llama3.2:3b`
- DeepEval (secondary; 3B judge unreliable)
- Claude Haiku 5.5 (optional, for comparison and cost)
- LangSmith (dataset, experiments, traces)

## Documents

The repo does not include the policy file. Add your own text file before ingesting.

```bash
mkdir -p documents
# Put a .txt or .pdf here, for example:
# documents/company_policies.txt
```
Then edit ingest.py if the filename is different, and run. 

## Run
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:3b
python ingest.py
uvicorn main:app --reload 
```
Open http://127.0.0.1:8000/docs.

- `POST /ingest` uploads a `.txt` or `.pdf`
- `POST /chat` asks a question
- `GET /health` checks the server

Ollama must be running for the local model. 

For Claude, set `ANTHROPIC_API_KEY `in `.env` and use the Claude path in `main.py`. 

## Evaluate

Dataset and experiments live in LangSmith (project: `policy-rag-eval`).

```bash
# App must be running
python run_eval.py
```
Human scores and failure analysis are in `eval_report.md`.

## Findings
- In-policy questions can be grounded.
- “I don’t know” and “I can’t do that” return no sources from the API.
- `llama3.2:3b` still reverses who pays whom on the phone reimbursement rule.
- Claude Haiku 5.5 passed all 11 human checks, including the inversion and refuse cases.
- Haiku answers were longer; correctness improved at the cost of more output tokens.

## Not in this repo
- Cloud deployment
- Automated judge as the final gate
