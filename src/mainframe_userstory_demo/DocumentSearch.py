import os
from typing import List, Dict, Tuple
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from datetime import datetime

class DocumentSearch:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the DocumentSearch class.
        
        Args:
            persist_directory (str): Directory to persist the Chroma database
        """
        # Initialize the embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize the vector store
        self.persist_directory = persist_directory
        self.vector_store = None
        
        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

    def load_and_split_document(self, file_path: str) -> List[Dict]:
        """
        Load and split a Python document into chunks.
        
        Args:
            file_path (str): Path to the document
            
        Returns:
            List[Dict]: List of document chunks with metadata
        """
        # Load the document
        loader = TextLoader(file_path)
        documents = loader.load()
        
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        splits = text_splitter.split_documents(documents)
        
        # Add custom metadata
        for split in splits:
            split.metadata.update({
                "source": file_path,
                "filename": os.path.basename(file_path),
                "date_added": datetime.now().isoformat(),
                "file_type": "python",
                "file_size": os.path.getsize(file_path)
            })
            
        return splits

    def create_vector_store(self, documents: List[Dict]):
        """
        Create a vector store from documents.
        
        Args:
            documents (List[Dict]): List of document chunks with metadata
        """
        # Initialize Chroma client
        client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Create or get collection
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            client=client,
            collection_name="python_docs"
        )
        self.vector_store.persist()

    def search_documents(self, 
                        query: str, 
                        n_results: int = 3):
        """
        Search the vector store for relevant documents.
        
        Args:
            query (str): Search query
            n_results (int): Number of results to return
            
        Returns:
            List[Tuple[Document, float]]: List of relevant documents with scores
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please create it first.")
            
        results = self.vector_store.similarity_search_with_relevance_scores(
            query,
            k=n_results
        )
        
        return results

    def process_directory(self, directory_path: str):
        """
        Process all Python files in a directory.
        
        Args:
            directory_path (str): Path to the directory containing Python files
        """
        all_documents = []
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        documents = self.load_and_split_document(file_path)
                        all_documents.extend(documents)
                        print(f"Processed: {file_path}")
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")
        
        if all_documents:
            self.create_vector_store(all_documents)
            print(f"Created vector store with {len(all_documents)} chunks")

def main():
    # Initialize the document search
    doc_search = DocumentSearch()
    
    # Directory containing Python files
    python_files_directory = "./python_files"  # Replace with your directory path
    
    # Process all Python files in the directory
    print("Processing Python files...")
    doc_search.process_directory(python_files_directory)
    
    # Example searches
    example_queries = [
        "Find the flyers name"
    ]
    
    # Perform searches
    for query in example_queries:
        print(f"\nSearching for: {query}")
        results = doc_search.search_documents(query)
        
        print("\nSearch Results:")
        for doc, score in results:
            print(f"\nRelevance Score: {score:.4f}")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")

if __name__ == "__main__":
    main()
