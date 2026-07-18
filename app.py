"""Gradio UI for the DocuChat RAG pipeline."""

import gradio as gr

from src.rag_pipeline import RAGPipeline

pipeline_instance = RAGPipeline()

def ingest_folder(folder_path):
    count = pipeline_instance.build_from_folder(folder_path)
    return f"Indexed {count} chunks from {folder_path}"

def ask_question(question):
    result = pipeline_instance.answer(question)
    sources = "\n".join(s["source"] for s in result["sources"])
    return result["answer"], sources

def build_interface():
    with gr.Blocks(title="DocuChat RAG") as demo:
        gr.Markdown("# DocuChat RAG\nAsk questions about your own documents.")
        folder_input = gr.Textbox(label="Documents folder path")
        ingest_button = gr.Button("Build index")
        ingest_status = gr.Textbox(label="Index status", interactive=False)
        ingest_button.click(ingest_folder, inputs=folder_input, outputs=ingest_status)
        question_input = gr.Textbox(label="Your question")
        ask_button = gr.Button("Ask")
        answer_output = gr.Textbox(label="Answer", interactive=False)
        sources_output = gr.Textbox(label="Sources", interactive=False)
        ask_button.click(ask_question, inputs=question_input, outputs=[answer_output, sources_output])
    return demo

if __name__ == "__main__":
    app = build_interface()
    app.launch()
