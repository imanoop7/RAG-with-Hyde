import gradio as gr
from rag_hyde_faiss import load_web_content, create_or_load_faiss_index, generate_hypothetical_document, rag_with_hyde

# Load web content
web_documents = load_web_content()

# Create or load FAISS index (optimized)
vectorstore = create_or_load_faiss_index(web_documents)

def process_question(question):
    # Generate hypothetical document
    hyde_doc = generate_hypothetical_document(question)
    
    # Perform RAG with HyDE
    answer, _ = rag_with_hyde(question, vectorstore)
    
    return hyde_doc, answer

# Define Gradio interface
iface = gr.Interface(
    fn=process_question,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs=[
        gr.Textbox(label="Hypothetical Document (HyDE)"),
        gr.Textbox(label="Final Answer")
    ],
    title="RAG with HyDE Demo",
    description="This demo shows the Hypothetical Document Embedding (HyDE) process and the final output of a Retrieval-Augmented Generation (RAG) system.",
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()