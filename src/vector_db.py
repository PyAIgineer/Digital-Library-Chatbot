from langchain_qdrant import QdrantVectorStore
import qdrant_client
from qdrant_client.http import models
import os
import json
from typing import List, Dict, Any
import warnings
import time
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore', category=RuntimeWarning)


def setup_vector_db(embeddings, collection_name="ebooks_library", local_path="./qdrant_db"):
    """Set up Qdrant vector database"""
    # Create a local Qdrant instance
    os.makedirs(local_path, exist_ok=True)
    
    max_retries = 5
    retry_delay = 1.5
    
    for attempt in range(max_retries):
        try:
            client = qdrant_client.QdrantClient(path=local_path)
            
            # Test connection by getting collections
            collections = client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            print(f"Successfully connected to vector database")
            print(f"Existing collections: {collection_names}")
            
            # Get vector size from embeddings model
            # Handle different embeddings types
            if hasattr(embeddings, 'client') and hasattr(embeddings.client, 'get_sentence_embedding_dimension'):
                # LangChain HuggingFaceEmbeddings
                vector_size = embeddings.client.get_sentence_embedding_dimension()
            elif isinstance(embeddings, SentenceTransformer):
                # Direct SentenceTransformer model
                vector_size = embeddings.get_sentence_embedding_dimension()
            else:
                # Fallback to checking the model name
                model_name = getattr(embeddings, 'model_name', "all-MiniLM-L6-v2")
                # Create a temporary model just to get the dimension
                temp_model = SentenceTransformer(model_name)
                vector_size = temp_model.get_sentence_embedding_dimension()
                del temp_model
            
            # Create or recreate collection if needed
            if collection_name not in collection_names:
                vector_config = models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
                
                # Create with optimized configuration
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=vector_config,
                    hnsw_config=models.HnswConfigDiff(
                        m=16,
                        ef_construct=128,  # Higher for better accuracy
                        full_scan_threshold=10000
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=20000,  # Larger batches for indexing
                        memmap_threshold=20000    # Use memory mapping for large collections
                    )
                )
                print(f"Created new collection: {collection_name}")
            
            # Create vector store
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embeddings
            )
            
            # Test vector store
            collection_info = client.get_collection(collection_name=collection_name)
            print(f"Collection info: {collection_info}")
            
            return vector_store
            
        except Exception as e:
            print(f"Connection attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            else:
                raise RuntimeError(f"Failed to connect to vector database after {max_retries} attempts")

def get_available_books(vector_db=None, library_dir="./ebooks_library"):
    """
    Get a list of all available books, simplified to just read from the library directory.
    This avoids the complexity of querying the vector database.
    """
    books = []
    
    # Simply get book names from PDF files in the library directory
    if os.path.exists(library_dir):
        books = [os.path.splitext(f)[0] for f in os.listdir(library_dir) if f.endswith('.pdf')]
        print(f"Found {len(books)} books in library directory: {books}")
    
    return books

def get_books_with_metadata(processed_data_dir="./processed_data"):
    """Get a list of all books with their metadata."""
    books = []
    
    if not os.path.exists(processed_data_dir):
        return books
    
    for file_hash in os.listdir(processed_data_dir):
        metadata_path = os.path.join(processed_data_dir, file_hash, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                books.append(metadata)
    
    return books

def clear_vector_db(vector_db):
    """Clear all data from vector database."""
    client = vector_db.client
    collection_name = vector_db.collection_name
    
    try:
        # Delete collection
        client.delete_collection(collection_name=collection_name)
        
        # Recreate collection with proper vector size
        # Get embeddings from vector_db
        embeddings = vector_db.embedding_function
        
        # Determine vector size based on embeddings type
        if hasattr(embeddings, 'client') and hasattr(embeddings.client, 'get_sentence_embedding_dimension'):
            vector_size = embeddings.client.get_sentence_embedding_dimension()
        elif isinstance(embeddings, SentenceTransformer):
            vector_size = embeddings.get_sentence_embedding_dimension()
        else:
            # Import directly from the new implementation
            from load_data import get_embedding_dimension
            vector_size = get_embedding_dimension()
        
        vector_config = models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        )
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vector_config,
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=128,
                full_scan_threshold=10000
            ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=20000,
                memmap_threshold=20000
            )
        )
        
        print(f"Vector database cleared and collection recreated")
        return True
    except Exception as e:
        print(f"Error clearing vector database: {e}")
        return False