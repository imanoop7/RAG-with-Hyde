import os
import hashlib
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_community.document_loaders import WebBaseLoader

# Initialize Ollama LLM with the smallest available model
llm = Ollama(model="tinyllama")

# Initialize Ollama Embeddings with Nomic model for efficient text embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

def load_web_content():
    """
    Load and process web content from predefined URLs.
    
    Returns:
        list: A list of Document objects containing processed web content.
    """
    urls = [
        "https://python.langchain.com/docs/get_started/introduction",
        "https://www.anthropic.com/index/retrieval-augmented-generation-rag",
        "https://arxiv.org/abs/2212.10496",  # HyDE paper
        "https://ollama.ai/",
        "https://blog.nomic.ai/"
    ]
    
    loader = WebBaseLoader(urls)
    documents = loader.load()
    
    processed_docs = []
    for doc in documents:
        chunks = doc.page_content.split('\n\n')
        for chunk in chunks:
            if len(chunk.strip()) > 100:  # Only keep substantial paragraphs
                processed_docs.append(Document(page_content=chunk.strip(), metadata=doc.metadata))
    
    return processed_docs

def get_documents_hash(documents):
    """
    Generate a hash of the documents to check for changes.
    """
    content = "".join(doc.page_content for doc in documents)
    return hashlib.md5(content.encode()).hexdigest()

def create_or_load_faiss_index(documents):
    """
    Create a new FAISS index or load an existing one if the documents haven't changed.
    
    Args:
        documents (list): List of Document objects to be indexed.
    
    Returns:
        FAISS: A FAISS vector store object.
    """
    index_name = "faiss_index"
    hash_file = f"{index_name}_hash.txt"
    
    current_hash = get_documents_hash(documents)
    
    if os.path.exists(f"{index_name}.faiss") and os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            stored_hash = f.read().strip()
        
        if stored_hash == current_hash:
            print("Loading existing FAISS index...")
            return FAISS.load_local(index_name, embeddings)
    
    print("Creating new FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    print("Saving FAISS index...")
    vectorstore.save_local(index_name)
    
    with open(hash_file, 'w') as f:
        f.write(current_hash)
    
    return vectorstore

def generate_hypothetical_document(question):
    """
    Generate a hypothetical document that answers the given question.
    
    Args:
        question (str): The input question.
    
    Returns:
        str: A hypothetical document answering the question.
    """
    hyde_prompt = PromptTemplate(
        input_variables=["question"],
        template="Please write a passage that answers the following question:\n\nQuestion: {question}\n\nPassage:"
    )
    hypothetical_doc = llm(hyde_prompt.format(question=question))
    return hypothetical_doc

def rag_with_hyde(question, vectorstore):
    """
    Perform Retrieval-Augmented Generation (RAG) with Hypothetical Document Embedding (HyDE).
    
    Args:
        question (str): The input question.
        vectorstore (FAISS): The FAISS vector store for document retrieval.
    
    Returns:
        tuple: A tuple containing the answer and the source documents.
    """
    hypothetical_doc = generate_hypothetical_document(question)
    augmented_query = f"Question: {question}\n\nContext: {hypothetical_doc}"
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    final_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Use the following context to answer the question. If the context doesn't contain relevant information, say so:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": final_prompt}
    )
    
    result = qa_chain({"query": question})
    return result["result"], result["source_documents"]

if __name__ == "__main__":
    # Load web content
    web_documents = load_web_content()

    # Create or load FAISS index
    vectorstore = create_or_load_faiss_index(web_documents)

    # Test the system with a sample question
    question = "How can HyDE improve RAG systems?"
    answer, sources = rag_with_hyde(question, vectorstore)

    # Print the results
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print("\nSources:")
    for doc in sources:
        print(f"- {doc.page_content[:100]}...")
        print(f"  Source: {doc.metadata['source']}\n")