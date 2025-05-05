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
        """Get relevant documents with optional reranking."""
        # Get initial docs from base retriever
        docs = self.base_retriever.get_relevant_documents(query)
        
        # Apply reranking if enabled
        if self.use_reranker and docs:
            # Import here to avoid circular imports
            from llm_interface import rerank_with_llm
            docs = rerank_with_llm(docs, query, self.llm, self.top_k)
        
        return docs