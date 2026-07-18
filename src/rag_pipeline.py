"""End-to-end RAG pipeline: ingest, embed, retrieve, and generate answers."""

from transformers import pipeline

from src.embeddings import embed_query, embed_texts, load_embedder
from src.ingest import chunk_documents, load_documents
from src.vectorstore import build_index, load_index, save_index, search

DEFAULT_GENERATOR_MODEL = "google/flan-t5-base"

PROMPT_TEMPLATE = "Answer the question using only the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

def build_context(passages):
    texts = [p["text"] for p in passages]
    return "\n\n".join(texts)

def build_prompt(question, passages):
    context = build_context(passages)
    return PROMPT_TEMPLATE.format(context=context, question=question)

class RAGPipeline:
    def __init__(self, generator_model=DEFAULT_GENERATOR_MODEL, top_k=4):
        self.embedder = load_embedder()
        self.generator = pipeline("text2text-generation", model=generator_model)
        self.top_k = top_k
        self.index = None
        self.chunks = None

    def build_from_folder(self, folder_path):
        documents = load_documents(folder_path)
        chunks = chunk_documents(documents)
        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(self.embedder, texts)
        self.index = build_index(embeddings)
        self.chunks = chunks
        return len(chunks)

    def save(self, folder_path):
        save_index(self.index, self.chunks, folder_path)

    def load(self, folder_path):
        self.index, self.chunks = load_index(folder_path)

    def retrieve(self, question):
        query_embedding = embed_query(self.embedder, question)
        return search(self.index, self.chunks, query_embedding, self.top_k)

    def answer(self, question):
        passages = self.retrieve(question)
        prompt = build_prompt(question, passages)
        generated = self.generator(prompt, max_new_tokens=200)
        answer_text = generated[0]["generated_text"]
        return {"answer": answer_text, "sources": passages}
