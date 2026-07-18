# docuchat-rag
End-to-end RAG project: embeddings + FAISS + transformers + Gradio UI

## Overview

DocuChat RAG lets you ask questions about your own documents. It loads text, markdown, and PDF files from a folder, splits them into overlapping chunks, embeds each chunk with a sentence-transformer model, indexes the vectors with FAISS, and answers questions by retrieving the most relevant chunks and passing them to a HuggingFace text-to-text generation model.

## How it works

Ingestion happens in `src/ingest.py`, which reads `.txt`, `.md`, and `.pdf` files from a folder and splits their text into overlapping word chunks. Embedding happens in `src/embeddings.py`, which wraps `sentence-transformers` and defaults to the `all-MiniLM-L6-v2` model. Vector storage and search happen in `src/vectorstore.py`, which builds a FAISS IndexFlatIP index and keeps the chunk metadata alongside it so results can be traced back to their source file. The full pipeline is tied together in `src/rag_pipeline.py`, whose RAGPipeline class can build an index from a folder, save and reload that index, retrieve relevant chunks for a question, and generate a final answer using a HuggingFace text2text-generation pipeline (google/flan-t5-base by default). app.py exposes all of this through a small Gradio interface: point it at a folder to build an index, then ask questions and see the answer alongside the source chunks it was based on.

## Project structure

```
docuchat-rag/
├── app.py
├── requirements.txt
└── src/
    ├── ingest.py
    ├── embeddings.py
    ├── vectorstore.py
    └── rag_pipeline.py
```

## Setup

```
pip install -r requirements.txt
python app.py
```

Then open the local URL that Gradio prints, point "Documents folder path" at a folder containing your files, click "Build index", and start asking questions.

## Important note on how this project was built

This code was written by Claude (an AI assistant) through a browser-only automation session, at the repository owner's request, as a portfolio project. It was written carefully against the real APIs of sentence-transformers, faiss-cpu, transformers, and gradio, but it has not actually been run, since the environment used to build it had no Python runtime available, only a web browser. Please install the dependencies and run it locally (or in a notebook) at least once before relying on it for a demo, interview, or production use, and fix anything that comes up. The embedding and generation models will download automatically on first run and require an internet connection.
