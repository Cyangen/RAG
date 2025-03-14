# GADIO

## Setup

### Setting up Python 

Ensure that you are in a Python environment with Python 3.12. For example:

```bash
conda env create -f env.yml
conda activate chatbot
```

Alternatively...

```sh
pip install -r requirements.txt
```

### Setting up Ollama

1. Download Ollama
You can download Ollama from https://www.ollama.com.

2. Verify Ollama is Running
- Once installed, ensure that Ollama is running by accessing: http://localhost:11434/

3. Pull Required Models

- With Ollama running, you’ll need to pull the following three models:
```bash
ollama pull nomic-embed-text
ollama pull llama3.1
ollama pull deepseek-r1:8b
```
```bash
╔══════════════════╦═════════════════════════════════════════╗
║       Name       ║                  Usage                  ║
╠══════════════════╬═════════════════════════════════════════╣
║ nomic-embed-text ║ text embedding for RAG                  ║
║ llama3.1         ║ Simple Chat, Fast Response              ║
║ deepseek-r1:8b   ║ Complex Chat, Well thought out Response ║
╚══════════════════╩═════════════════════════════════════════╝
```

## Testing it Out

## Creating the Vector Store

```bash
python create_vs.py
```

## Running as FastAPI Backend

```bash
fastapi run main.py
```

By default, the backend runs at http://localhost:8000/.

## Running as Streamlit App

```bash
streamlit run app.py
```

## Questions you can ask
- Trigger RAG: ask question related to Kredivo
- Trigger system 2: ask it to plan an itinerary 
