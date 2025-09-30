import chromadb
import os
from typing import List, Dict, Any, Optional
from agentic_framework_pkg.logger_config import get_logger # Assuming you have a shared logger

# Assuming an embedding function is available, e.g., from LiteLLM or a direct provider
# from .embedding_utils import get_embeddings # Placeholder for your embedding logic

logger = get_logger(__name__)

# Default path for ChromaDB persistent storage for VectorStoreManager.
# Attempt to use CHROMA_DB_PATH from the environment (e.g., .env file),
# otherwise fall back to a default specific to VectorStoreManager or a general default.
CHROMA_DATA_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_vsm_store") # Default if CHROMA_DB_PATH is not set.

class VectorStoreManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
            logger.info(f"Initializing ChromaDB client with path: {CHROMA_DATA_PATH}")
            os.makedirs(CHROMA_DATA_PATH, exist_ok=True)
            cls._instance.client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
            # You'll need an embedding function. This is a placeholder.
            # Replace with your actual embedding model, e.g., SentenceTransformer, OpenAIEmbeddings, etc.
            # from chromadb.utils import embedding_functions
            # cls._instance.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            # For more control, integrate your chosen embedding model (e.g., via LiteLLM or directly)
            logger.info("ChromaDB client initialized.")
        return cls._instance

    def get_or_create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        try:
            # collection = self.client.get_or_create_collection(name=name, embedding_function=self.embedding_function, metadata=metadata)
            # If your embedding function is part of ChromaDB's default or you manage embeddings externally:
            collection = self.client.get_or_create_collection(name=name, metadata=metadata)
            logger.info(f"Accessed or created collection: {name}")
            return collection
        except Exception as e:
            logger.error(f"Error getting or creating collection '{name}': {e}", exc_info=True)
            raise

    def add_documents(self, collection_name: str, documents: List[str], embeddings: Optional[List[List[float]]] = None, metadatas: Optional[List[dict]] = None, ids: Optional[List[str]] = None):
        if not documents:
            logger.warning(f"No documents provided to add to collection '{collection_name}'.")
            return
        try:
            collection = self.get_or_create_collection(collection_name)
            # If you provide embeddings directly (recommended for consistency if generated elsewhere):
            if embeddings:
                collection.add(embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids)
            else:
                # If relying on ChromaDB's internal embedding function (ensure it's configured)
                collection.add(documents=documents, metadatas=metadatas, ids=ids)
            logger.info(f"Added {len(documents)} documents to collection '{collection_name}'.")
        except Exception as e:
            logger.error(f"Error adding documents to collection '{collection_name}': {e}", exc_info=True)
            raise

    def query_collection(self, collection_name: str, query_texts: Optional[List[str]] = None, 
                         query_embeddings: Optional[List[List[float]]] = None, n_results: int = 5, 
                         where: Optional[dict] = None, where_document: Optional[dict] = None,
                         include: Optional[List[str]] = None,
                         ) -> Dict[str, Any]:
        if not query_texts and not query_embeddings:
            raise ValueError("Either query_texts or query_embeddings must be provided.")
        try:
            collection = self.get_or_create_collection(collection_name)
            results = collection.query(
                query_texts=query_texts,
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            logger.info(f"Queried collection '{collection_name}' with {len(query_texts or query_embeddings)} queries, got {len(results.get('ids', [[]])[0])} results for the first query.")
            return results
        except Exception as e:
            logger.error(f"Error querying collection '{collection_name}': {e}", exc_info=True)
            raise