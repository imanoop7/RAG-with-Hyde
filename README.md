# RAG with HyDE using FAISS

This project implements a Retrieval-Augmented Generation (RAG) system using the Hypothetical Document Embedding (HyDE) technique and FAISS for efficient vector storage and retrieval. It includes both a command-line interface and a Gradio web interface for easy interaction.

## Setup

1. Ensure you have Python 3.7+ installed on your system.

2. Clone this repository and navigate to the project directory.

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Install Ollama by following the instructions at https://ollama.ai/download

5. Download the required Ollama models:

   For the LLM (tinyllama):
   ```
   ollama pull tinyllama
   ```

   For the embeddings (nomic-embed-text):
   ```
   ollama pull nomic-embed-text
   ```

   You can verify the models are installed by running:
   ```
   ollama list
   ```

   This should show both `tinyllama` and `nomic-embed-text` in the list of available models.

6. Make sure Ollama is running. You can start it with:
   ```
   ollama serve
   ```

## Running the System

### Command-line Interface

To run the RAG system with a command-line interface:

```
python rag_hyde_faiss.py
```

This will process a sample question and display the results in the terminal.

### Gradio Web Interface

To run the system with a Gradio web interface:

```
python gradio_ui.py
```

This will start a local server and provide a URL where you can access the UI in your web browser. The interface allows you to:
- Enter your own questions
- View the generated hypothetical document (HyDE output)
- See the final answer produced by the RAG system

## Project Structure

- `rag_hyde_faiss.py`: Main implementation of the RAG system with HyDE and FAISS.
- `gradio_ui.py`: Gradio web interface for the RAG system.
- `requirements.txt`: List of Python dependencies.
- `README.md`: This file, containing project information and instructions.

## Notes

- The system requires an active internet connection to fetch web content and download necessary models.
- The FAISS index will be saved in the current directory as `faiss_index.faiss`.
- A hash of the processed documents is stored to avoid unnecessary reindexing.

## Customization

You can modify the `urls` list in `rag_hyde_faiss.py` to change the source documents for the RAG system. Remember to delete the existing `faiss_index.faiss` and `faiss_index_hash.txt` files to force reindexing when you change the source documents.
