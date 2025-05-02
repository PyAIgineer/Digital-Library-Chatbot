from langchain.schema import BaseRetriever

class RerankedRetriever(BaseRetriever):
    """Custom retriever that wraps the base retriever with reranking."""
    
    def __init__(self, base_retriever, llm, use_reranker=True, top_k=5):
        """Initialize the reranked retriever."""
        self.base_retriever = base_retriever
        self.llm = llm
        self.use_reranker = use_reranker
        self.top_k = top_k
        
    def get_relevant_documents(self, query):
        """Get relevant documents with optional reranking."""
        # Get initial docs from base retriever
        docs = self.base_retriever.get_relevant_documents(query)
        
        # Apply reranking if enabled
        if self.use_reranker and docs:
            # Import here to avoid circular imports
            from llm_interface import rerank_with_llm
            docs = rerank_with_llm(docs, query, self.llm, self.top_k)
        
        return docs