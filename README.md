# Agentic Healthcare Assistant for Medical Task Automation

This project implements the capstone problem statement as a Python and Streamlit application. It loads patient data from Excel and PDF files, retrieves medical history, books mock appointments, returns condition-level medical information, and logs agent planning, tool usage, and evaluation metrics.

## Features

- Patient record ingestion from `records.xlsx`
- PDF parsing for sample clinical reports
- FAISS-backed local patient retrieval using deterministic embeddings
- LangGraph-powered agent flow with OpenAI-backed planning and response generation
- LLM evaluation using LangChain QAEvalChain plus structured scoring
- Mock doctor schedule and appointment booking
- MedlinePlus medical information lookup with local trusted-source fallback
- Manual patient record add/update workflow backed by `data/user_records.json`
- Editable patient and doctor tables with Excel import/export
- Streamlit dashboard for chat, patients, appointments, traces, and evaluation
- JSONL logs for LLMOps-style monitoring

## Project Structure

```text
sourav_healthcare_capstone/
├── app.py
├── requirements.txt
├── data/
├── logs/
├── src/
└── tests/
```

## Setup on macOS

```bash
cd /Users/sourav/Documents/VSCode/GenAI/sourav_healthcare_capstone
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

Paste your OpenAI API key into the sidebar LLM settings, or set it before launch:

```bash
export OPENAI_API_KEY="your-key-here"
export OPENAI_MODEL="gpt-4.1-mini"
streamlit run app.py
```

## Demo Queries

```text
Summarize David Thompson's diabetes history.
```

```text
Ramesh has hypertension. Find his history, book a cardiologist, and summarize lifestyle advice.
```

```text
My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. Also, summarize latest treatment methods.
```

```text
What are Anjali Mehra's symptoms and treatment plan?
```

## Important Safety Note

The assistant is for administrative and educational support only. It does not diagnose, prescribe, or replace a licensed clinician.

## Notes

- The app attempts live MedlinePlus lookup when network access is available and falls back to local trusted summaries when offline.
- LLM planning and evaluation require `OPENAI_API_KEY`. The app does not store API keys in source code.
- Patient updates are persisted to `data/user_records.json`.
- The included FAISS retrieval uses deterministic local embeddings so the app works without downloading embedding models.
