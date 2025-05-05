# For retriever_utils.py - Improved implementation

from langchain.schema import BaseRetriever
from langchain.schema.document import Document
from typing import List, Any, Optional, Dict
from pydantic import BaseModel, Field
from langchain.callbacks.manager import CallbackManagerForRetrieverRun

class RerankedRetriever(BaseRetriever, BaseModel):
    """Custom retriever that wraps the base retriever with reranking."""
    
    base_retriever: Any = Field(description="The base retriever to wrap")
    llm: Any = Field(description="The language model to use for reranking")
    use_reranker: bool = Field(default=True, description="Whether to use reranking")
    top_k: int = Field(default=5, description="Number of documents to return")
    
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Get relevant documents with proper run_manager handling."""
        # First try getting docs from base retriever
        docs = []
        try:
            # Try using get_relevant_documents if available
            if hasattr(self.base_retriever, 'get_relevant_documents'):
                try:
                    docs = self.base_retriever.get_relevant_documents(query)
                except TypeError as e:
                    # Handle case where get_relevant_documents requires run_manager
                    if "run_manager" in str(e):
                        # Try with explicit run_manager forwarding
                        docs = self.base_retriever.get_relevant_documents(
                            query, run_manager=run_manager
                        )
                    else:
                        raise
            # Fallback to _get_relevant_documents if needed
            elif hasattr(self.base_retriever, '_get_relevant_documents'):
                docs = self.base_retriever._get_relevant_documents(
                    query, run_manager=run_manager
                )
            # Try alternative retrieval methods in EnhancedRetriever
            elif hasattr(self.base_retriever, 'advanced_retrieval'):
                results = self.base_retriever.advanced_retrieval(query, top_k=self.top_k*2)
                docs = [Document(
                    page_content=r["content"],
                    metadata=r["metadata"]
                ) for r in results]
            elif hasattr(self.base_retriever, 'semantic_search'):
                results = self.base_retriever.semantic_search(query, top_k=self.top_k*2)
                docs = [Document(
                    page_content=r["content"],
                    metadata=r["metadata"]
                ) for r in results]
            else:
                raise ValueError("Base retriever doesn't have compatible retrieval methods")
        except Exception as e:
            import traceback
            print(f"Error in retrieval: {e}")
            traceback.print_exc()
            return []
        
        # Apply reranking if enabled and we have documents
        if self.use_reranker and docs:
            # Import within function to avoid circular imports
            from llm_interface import rerank_with_llm
            docs = rerank_with_llm(docs, query, self.llm, self.top_k)
        
        return docs[:self.top_k]