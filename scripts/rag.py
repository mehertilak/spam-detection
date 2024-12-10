from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader, TextLoader
import os

class RAGSystem:
    def __init__(self, documents_path):
        self.documents_path = documents_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.embeddings = HuggingFaceEmbeddings()
        self.vector_store = None

    def load_documents(self):
        """Load documents from the specified path."""
        if os.path.isdir(self.documents_path):
            loader = DirectoryLoader(self.documents_path, glob="**/*.txt", loader_cls=TextLoader)
        else:
            loader = TextLoader(self.documents_path)
        documents = loader.load()
        return documents

    def process_documents(self):
        """Process documents and create vector store."""
        documents = self.load_documents()
        texts = self.text_splitter.split_documents(documents)
        self.vector_store = FAISS.from_documents(texts, self.embeddings)

    def query(self, query_text, k=4):
        """Query the vector store for relevant documents."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please process documents first.")
        
        docs = self.vector_store.similarity_search(query_text, k=k)
        return docs

    def save_vector_store(self, save_path):
        """Save the vector store to disk."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please process documents first.")
        self.vector_store.save_local(save_path)

    def load_vector_store(self, load_path):
        """Load a previously saved vector store."""
        self.vector_store = FAISS.load_local(load_path, self.embeddings)

# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag = RAGSystem("path/to/your/documents")
    
    # Process documents and create vector store
    rag.process_documents()
    
    # Save vector store
    rag.save_vector_store("vector_store")
    
    # Query the system
    results = rag.query("Your query here")
    for doc in results:
        print(doc.page_content)
