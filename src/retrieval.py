# retrieval.py

import os
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document
import re
from tqdm import tqdm
import time

class EnhancedRetriever:
    """Advanced retrieval system for semantic search and context building from PDF documents."""
    
    def __init__(self, vector_db, embedding_model=None, 
                 processed_data_dir="./processed_data",
                 top_k=5, min_relevance_score=0.6):
        """Initialize the retriever with vector database and configuration.
        
        Args:
            vector_db: The vector database (QdrantVectorStore instance)
            embedding_model: Optional embedding model for direct similarity computation
            processed_data_dir: Directory containing processed document data
            top_k: Default number of chunks to retrieve
            min_relevance_score: Minimum similarity score to consider a chunk relevant
        """
        self.vector_db = vector_db
        self.embedding_model = embedding_model
        self.processed_data_dir = processed_data_dir
        self.top_k = top_k
        self.min_relevance_score = min_relevance_score
        
        # Load available books metadata for additional context
        self.books_metadata = self._load_books_metadata()
        
        print(f"Initialized retriever with {len(self.books_metadata)} available books")
    
    def _load_books_metadata(self):
        """Load metadata for all processed books."""
        books = {}
        
        if not os.path.exists(self.processed_data_dir):
            print(f"Warning: Processed data directory {self.processed_data_dir} does not exist.")
            return books
        
        # Get books metadata from processed data directory
        for file_hash in os.listdir(self.processed_data_dir):
            metadata_path = os.path.join(self.processed_data_dir, file_hash, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                        books[metadata["book_title"]] = metadata
                except Exception as e:
                    print(f"Error loading metadata for {file_hash}: {str(e)}")
        
        return books
    
    def semantic_search(self, query, top_k=None, filters=None):
        """Perform semantic search over the vector database.
        
        Args:
            query: User query string
            top_k: Number of results to return (uses default if None)
            filters: Optional dictionary of metadata filters
            
        Returns:
            List of search results with metadata and similarity scores
        """
        if top_k is None:
            top_k = self.top_k
        
        # Get raw results from vector database
        raw_results = self._vector_search(query, top_k=top_k, filters=filters)
        
        # Process results into a more usable format
        processed_results = self._process_search_results(raw_results)
        
        # Apply additional relevance filtering if embedding model is available
        if self.embedding_model:
            processed_results = self.filter_by_relevance(processed_results, query)
        
        return processed_results
    
    def _vector_search(self, query, top_k=None, filters=None):
        """Perform vector search with the vector database.
        
        Args:
            query: User query string
            top_k: Number of results to return
            filters: Metadata filters to apply
            
        Returns:
            Raw search results
        """
        if top_k is None:
            top_k = self.top_k
        
        try:
            # Check if filters is a dict or a filter object
            if filters:
                # For Qdrant, convert dictionary filters to the correct format
                # Qdrant expects filters in a specific format, not just a plain dictionary
                filter_dict = {}
                for key, value in filters.items():
                    filter_dict[key] = {"$eq": value}
                
                # Perform search with filter
                results = self.vector_db.similarity_search_with_score(
                    query, k=top_k, filter=filter_dict
                )
            else:
                # Perform similarity search without filters
                results = self.vector_db.similarity_search_with_score(
                    query, k=top_k,
                )
            
            return results
        except Exception as e:
            print(f"Error in vector search: {str(e)}")
            return []
    
    def _process_search_results(self, results):
        """Process raw search results into a more usable format.
        
        Args:
            results: Raw search results from vector database
            
        Returns:
            List of processed results with enhanced metadata
        """
        processed_results = []
        
        for doc, score in results:
            # Create a processed result object
            processed_result = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "book_title": doc.metadata.get("book_title", "Unknown"),
                "page_num": doc.metadata.get("page_num", 0),
                "header": doc.metadata.get("current_header", ""),
                "chunk_type": "table" if doc.metadata.get("content_type") == "table" else "text",
                "score": score,
                "file_hash": doc.metadata.get("file_hash", "")
            }
            
            # Add additional context from book metadata if available
            if processed_result["book_title"] in self.books_metadata:
                book_meta = self.books_metadata[processed_result["book_title"]]
                processed_result["author"] = book_meta.get("author", "Unknown")
                processed_result["total_pages"] = book_meta.get("total_pages", 0)
            
            processed_results.append(processed_result)
        
        return processed_results
    
    def filter_by_relevance(self, results, query, threshold=None):
        """Filter results by computing direct relevance to query.
        
        Args:
            results: List of search results
            query: User query
            threshold: Minimum similarity score (uses default if None)
            
        Returns:
                Filtered list of results above threshold
        """
        if threshold is None:
            threshold = self.min_relevance_score
        
        if not self.embedding_model or not results:
            # If no embedding model available or no results, return all results
            return results
        
        try:
            # Compute query embedding - handle different types of embedding models
            if hasattr(self.embedding_model, 'embed_query'):
                # LangChain HuggingFaceEmbeddings
                query_embedding = self.embedding_model.embed_query(query)
            elif hasattr(self.embedding_model, 'encode'):
                # SentenceTransformer
                query_embedding = self.embedding_model.encode([query])[0]
            else:
                # Unknown embedding model type - return results without filtering
                print("Unknown embedding model type, skipping relevance filtering")
                return results
            
            filtered_results = []
            for result in results:
                # Compute text embedding and similarity
                text = result["content"] if "content" in result else result.get("page_content", "")
                
                if hasattr(self.embedding_model, 'embed_query'):
                    # LangChain HuggingFaceEmbeddings
                    text_embedding = self.embedding_model.embed_query(text)
                elif hasattr(self.embedding_model, 'encode'):
                    # SentenceTransformer
                    text_embedding = self.embedding_model.encode([text])[0]
                
                # Calculate similarity
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np
                
                # Convert to numpy arrays if needed
                query_embedding_np = np.array(query_embedding).reshape(1, -1)
                text_embedding_np = np.array(text_embedding).reshape(1, -1)
                
                similarity = cosine_similarity(query_embedding_np, text_embedding_np)[0][0]
                
                if similarity >= threshold:
                    result["similarity"] = float(similarity)
                    filtered_results.append(result)
            
            # Sort by similarity
            filtered_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
            
            return filtered_results
        except Exception as e:
            print(f"Error in relevance filtering: {str(e)}")
            return results
    
    def build_qa_context(self, query, top_k=None, max_tokens=4000):
        """Build context for question answering with the most relevant content.
        
        Args:
            query: User question
            top_k: Number of chunks to include (uses default if None)
            max_tokens: Maximum context size in tokens
            
        Returns:
            String containing formatted context for LLM consumption
        """
        if top_k is None:
            top_k = min(self.top_k * 2, 10)  # Use more chunks for context building
        
        # Get relevant chunks
        search_results = self.semantic_search(query, top_k=top_k)
        
        if not search_results:
            return "No relevant information found."
        
        # Format context
        context_parts = []
        total_tokens = 0  # Simple approximation, 4 chars ~ 1 token
        
        # Add book information
        books_included = set()
        for result in search_results:
            book_title = result["book_title"]
            if book_title not in books_included:
                books_included.add(book_title)
                
                book_info = f"BOOK: {book_title}"
                if "author" in result:
                    book_info += f" by {result['author']}"
                
                context_parts.append(book_info)
                total_tokens += len(book_info) // 4
        
        # Add context snippets
        for i, result in enumerate(search_results):
            # Format the snippet with metadata
            header = result.get("header", "")
            page_num = result.get("page_num", 0)
            content = result.get("content", "")
            
            snippet = f"\n--- SNIPPET {i+1} ---\n"
            if header:
                snippet += f"SECTION: {header}\n"
            snippet += f"PAGE: {page_num}\n"
            snippet += f"CONTENT: {content}\n"
            
            # Check token limit
            snippet_tokens = len(snippet) // 4
            if total_tokens + snippet_tokens > max_tokens:
                # If adding this would exceed token limit, stop
                context_parts.append("\n(Additional relevant content was found but omitted due to context length limits.)")
                break
            
            context_parts.append(snippet)
            total_tokens += snippet_tokens
        
        return "\n".join(context_parts)
    
    def search_by_book(self, query, book_titles=None, top_k=None):
        """Search within specific books.
        
        Args:
            query: User query
            book_titles: List of book titles to search within (None for all)
            top_k: Number of results per book
            
        Returns:
            Dictionary of results by book
        """
        if top_k is None:
            top_k = self.top_k
        
        if book_titles is None:
            # Get all available books
            book_titles = list(self.books_metadata.keys())
        
        results_by_book = {}
        
        for book_title in book_titles:
            # Create filter for this book
            filters = {
                "book_title": book_title
            }
            
            # Search with book filter
            book_results = self.semantic_search(query, top_k=top_k, filters=filters)
            
            if book_results:
                results_by_book[book_title] = book_results
        
        return results_by_book
    
    # Add this method to the EnhancedRetriever class in retrieval.py

    def get_relevant_documents(self, query, metadata_filters=None, top_k=None):
        """Get relevant documents based on the query."""
        if top_k is None:
            top_k = self.top_k
            
        try:
            # If filters are provided, we need to handle them properly
            if metadata_filters:
                # Get all documents from the vector DB and filter manually
                # This bypasses the filter compatibility issues
                all_results = self.vector_db.similarity_search(query, k=top_k*3)
                
                # Filter results manually based on metadata
                filtered_results = []
                for doc in all_results:
                    matches_all_filters = True
                    for key, value in metadata_filters.items():
                        if doc.metadata.get(key) != value:
                            matches_all_filters = False
                            break
                    
                    if matches_all_filters:
                        filtered_results.append(doc)
                        
                        # Stop once we have enough results
                        if len(filtered_results) >= top_k:
                            break
                            
                return filtered_results[:top_k]
            else:
                # No filters, use standard search
                return self.vector_db.similarity_search(query, k=top_k)
                
        except Exception as e:
            print(f"Error in get_relevant_documents: {str(e)}")
            # Return empty list on error to ensure graceful fallback
            return []
        
    def get_document_structure(self, book_title):
        """Get structural information about a document.
        
        Args:
            book_title: Title of the book
            
        Returns:
            Document structure with headers, chapters, etc.
        """
        # Find book hash from metadata
        book_metadata = self.books_metadata.get(book_title)
        if not book_metadata:
            return None
        
        file_hash = book_metadata["file_hash"]
        
        # Load TOC file generated by EnhancedPDFProcessor
        toc_path = os.path.join(self.processed_data_dir, file_hash, "toc.json")
        if os.path.exists(toc_path):
            with open(toc_path, "r") as f:
                toc = json.load(f)
            return toc
        
        return None
    
    def get_chapter_content(self, book_title, chapter_id):
        """Retrieve all content from a specific chapter.
        
        Args:
            book_title: Title of the book
            chapter_id: ID of the chapter to retrieve
            
        Returns:
            List of chunks belonging to the chapter
        """
        # Find book hash from metadata
        book_metadata = self.books_metadata.get(book_title)
        if not book_metadata:
            return []
        
        file_hash = book_metadata["file_hash"]
        
        # Load chunks file
        chunks_path = os.path.join(self.processed_data_dir, file_hash, "chunks.json")
        if not os.path.exists(chunks_path):
            return []
        
        with open(chunks_path, "r") as f:
            chunks_data = json.load(f)
        
        # Filter chunks by chapter ID
        chapter_chunks = []
        for chunk_data in chunks_data:
            metadata = chunk_data.get("metadata", {})
            if metadata.get("chapter_num") == chapter_id:
                chapter_chunks.append(chunk_data)
        
        return chapter_chunks
    
    def find_relevant_tables(self, query, top_k=3):
        """Find tables relevant to the query.
        
        Args:
            query: User query
            top_k: Number of tables to return
            
        Returns:
            List of relevant tables with context
        """
        # Add filter for tables
        filters = {
            "content_type": "table"
        }
        
        # Search for tables
        table_results = self.semantic_search(query, top_k=top_k, filters=filters)
        
        # Format table results
        formatted_tables = []
        for result in table_results:
            table_info = {
                "table_content": result["content"],
                "table_id": result["metadata"].get("table_id", "unknown"),
                "book_title": result["book_title"],
                "page_num": result["page_num"],
                "section": result["header"],
                "similarity": result.get("similarity", result["score"])
            }
            formatted_tables.append(table_info)
        
        return formatted_tables
    
    def answer_question(self, question, llm_client, max_context_tokens=4000):
        """Answer a question using context and an LLM.
        
        Args:
            question: User question
            llm_client: Language model client with generate_text method
            max_context_tokens: Maximum context size
            
        Returns:
            Answer with source information
        """
        # Build context from relevant chunks
        context = self.build_qa_context(question, max_tokens=max_context_tokens)
        
        # Create prompt with context and question
        prompt = f"""
        Answer the question based on the following context:
        
        CONTEXT:
        {context}
        
        QUESTION: {question}
        
        ANSWER:
        """
        
        # Get answer from LLM
        response = llm_client.generate_text(prompt)
        
        # Extract sources for citation
        sources = self._extract_sources_from_context(context)
        
        return {
            "answer": response,
            "sources": sources
        }
    
    def _extract_sources_from_context(self, context):
        """Extract source information from context.
        
        Args:
            context: Context string
            
        Returns:
            List of source information
        """
        sources = []
        
        # Extract book information
        book_pattern = re.compile(r"BOOK: ([^(]+)(?:\s+by\s+([^(]+))?")
        for match in book_pattern.finditer(context):
            book_title = match.group(1).strip()
            author = match.group(2).strip() if match.group(2) else "Unknown"
            
            sources.append({
                "book_title": book_title,
                "author": author
            })
        
        # Extract snippet information
        snippet_pattern = re.compile(r"--- SNIPPET \d+ ---\n(?:SECTION: ([^\n]+)\n)?PAGE: (\d+)")
        for match in snippet_pattern.finditer(context):
            section = match.group(1) if match.group(1) else ""
            page = match.group(2)
            
            source_entry = {
                "section": section,
                "page": page
            }
            
            # Find the associated book
            if sources:
                source_entry["book_title"] = sources[0]["book_title"]
                source_entry["author"] = sources[0]["author"]
            
            sources.append(source_entry)
        
        return sources
    
    def compare_information(self, query, book_titles):
        """Compare information across multiple books.
        
        Args:
            query: Topic to compare
            book_titles: List of books to compare
            
        Returns:
            Comparative analysis with sources
        """
        # Get results from each book
        results_by_book = self.search_by_book(query, book_titles, top_k=3)
        
        if not results_by_book:
            return {"comparison": "No relevant information found in the specified books."}
        
        # Format comparative results
        comparison_data = {
            "query": query,
            "books_compared": list(results_by_book.keys()),
            "book_results": {}
        }
        
        for book_title, results in results_by_book.items():
            book_data = {
                "top_excerpts": [],
                "relevance_score": 0.0,
                "total_matches": len(results)
            }
            
            # Extract top excerpts
            for result in results[:3]:  # Use top 3 results
                excerpt = {
                    "content": result["content"],
                    "page": result["page_num"],
                    "section": result["header"],
                    "relevance": result.get("similarity", result["score"])
                }
                book_data["top_excerpts"].append(excerpt)
            
            # Calculate average relevance
            if results:
                scores = [r.get("similarity", r["score"]) for r in results]
                book_data["relevance_score"] = sum(scores) / len(scores)
            
            comparison_data["book_results"][book_title] = book_data
        
        return comparison_data
    
    def get_book_chapters(self, book_title):
        """Get all chapters for a specific book.
        
        Args:
            book_title: Title of the book
            
        Returns:
            List of chapters with their metadata
        """
        structure = self.get_document_structure(book_title)
        if not structure:
            return []
        
        # Extract chapters (level 1 headers)
        chapters = [item for item in structure if item.get("level", 0) == 1]
        return chapters
    
    def search_fuzzy(self, query, min_score=0.5):
        """Perform fuzzy search for queries that may not have exact matches.
        
        Args:
            query: User query
            min_score: Minimum similarity score threshold
            
        Returns:
            List of search results with fuzzy matching
        """
        # First try normal search
        results = self.semantic_search(query)
        
        # If results are insufficient, try decomposing the query
        if not results or results[0].get("similarity", results[0]["score"]) < min_score:
            # Break query into key concepts
            concepts = self._extract_key_concepts(query)
            
            all_results = []
            for concept in concepts:
                concept_results = self.semantic_search(concept, top_k=2)
                all_results.extend(concept_results)
            
            # Deduplicate results
            seen_ids = set()
            unique_results = []
            for result in all_results:
                result_id = f"{result['file_hash']}_{result['page_num']}_{result['metadata'].get('chunk_index', '')}"
                if result_id not in seen_ids:
                    seen_ids.add(result_id)
                    unique_results.append(result)
            
            # Sort by relevance and return
            unique_results.sort(key=lambda x: x.get("similarity", x["score"]), reverse=True)
            return unique_results[:self.top_k]
        
        return results
    
    def _extract_key_concepts(self, query):
        """Extract key concepts from a complex query."""
        # Simple keyword extraction based on POS tagging
        # For more advanced extraction, use NLP libraries
        keywords = []
        
        # Simple splitting by common conjunctions and stop words
        simple_split = re.split(r'\s+(?:and|or|in|the|with|about)\s+', query)
        for phrase in simple_split:
            if len(phrase.strip()) > 3:  # Only keep meaningful phrases
                keywords.append(phrase.strip())
        
        # If no good keywords were found, default to the full query
        if not keywords:
            keywords = [query]
        
        return keywords
    
    def advanced_retrieval(self, query, retrieval_mode="standard", top_k=None, filters=None):
        """Perform advanced retrieval with different modes.
        
        Args:
            query: User query
            retrieval_mode: Mode of retrieval: "standard", "high_precision", "high_recall", "hybrid"
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of search results optimized according to the mode
        """
        if top_k is None:
            top_k = self.top_k
        
        if retrieval_mode == "standard":
            # Standard search with balanced precision/recall
            return self.semantic_search(query, top_k=top_k, filters=filters)
            
        elif retrieval_mode == "high_precision":
            # High precision mode focuses on most relevant results
            results = self.semantic_search(query, top_k=top_k*2, filters=filters)
            
            # Apply stricter relevance filtering
            if self.embedding_model:
                results = self.filter_by_relevance(results, query, threshold=self.min_relevance_score+0.1)
            
            return results[:top_k]
            
        elif retrieval_mode == "high_recall":
            # High recall mode tries to get more diverse results
            # First get standard results
            standard_results = self.semantic_search(query, top_k=top_k, filters=filters)
            
            # Then get some results from a fuzzy search
            fuzzy_results = self.search_fuzzy(query)
            
            # Combine and deduplicate
            combined_results = standard_results + fuzzy_results
            
            # Deduplicate by file hash and page
            seen_ids = set()
            unique_results = []
            for result in combined_results:
                result_id = f"{result['file_hash']}_{result['page_num']}_{result['metadata'].get('chunk_index', '')}"
                if result_id not in seen_ids:
                    seen_ids.add(result_id)
                    unique_results.append(result)
            
            # Sort by relevance and return
            unique_results.sort(key=lambda x: x.get("similarity", x["score"]), reverse=True)
            return unique_results[:top_k]
            
        elif retrieval_mode == "hybrid":
            # Use vector search but with semantic re-ranking
            # First get more results than needed
            results = self.semantic_search(query, top_k=top_k*3, filters=filters)
            
            if self.embedding_model and results:
                # Re-rank with direct semantic similarity
                query_embedding = self.embedding_model.encode([query])[0]
                
                for result in results:
                    text_embedding = self.embedding_model.encode([result["content"]])[0]
                    similarity = cosine_similarity([query_embedding], [text_embedding])[0][0]
                    result["hybrid_score"] = similarity * 0.7 + result["score"] * 0.3  # Weighted combination
                
                # Re-sort by hybrid score
                results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
            
            return results[:top_k]
        
        else:
            # Default to standard search if invalid mode
            return self.semantic_search(query, top_k=top_k, filters=filters)

    def retrieve_with_mmr(self, query, top_k=None, diversity_factor=0.3):
        """Retrieve using Maximum Marginal Relevance for diversity.
        
        Args:
            query: User query
            top_k: Number of results to return
            diversity_factor: How much to prioritize diversity (0-1)
            
        Returns:
            Diverse list of search results
        """
        if top_k is None:
            top_k = self.top_k
        
        # Step 1: Get initial results using vector DB's built-in search
        # Get more results than needed for diversity selection
        initial_k = min(top_k * 3, 50)  # Limit to reasonable number
        initial_results = self.semantic_search(query, top_k=initial_k)
        
        if not initial_results or len(initial_results) <= top_k:
            return initial_results
        
        # Step 2: Use MMR algorithm with vectors already in the database
        try:
            # Get vector IDs from results for efficient retrieval
            point_ids = [r["metadata"].get("langchain_id", "") for r in initial_results]
            valid_ids = [pid for pid in point_ids if pid]
            
            if len(valid_ids) <= top_k:
                # Not enough valid IDs for MMR, return initial results
                return initial_results[:top_k]
            
            # Get query vector from Qdrant (reusing the vector from initial search)
            search_params = {
                "query": query,
                "k": 1,  # We just need the vector itself
                "with_vectors": True
            }
            
            # Use built-in MMR directly if available (newer Qdrant versions)
            if hasattr(self.vector_db.client, "search_with_mmr"):
                mmr_results = self.vector_db.client.search_with_mmr(
                    collection_name=self.vector_db.collection_name,
                    query=query,
                    k=top_k,
                    diversity=diversity_factor
                )
                
                # Convert to our format
                processed_results = self._process_search_results(
                    [(self.vector_db.get_by_id(r.id), r.score) for r in mmr_results]
                )
                return processed_results
            
            # Otherwise implement MMR ourselves using the vectors from DB
            # (This is more efficient than recomputing all embeddings)
            selected_results = []
            remaining_results = initial_results.copy()
            
            # Select first by highest similarity
            remaining_results.sort(key=lambda x: x.get("similarity", x["score"]), reverse=True)
            selected_results.append(remaining_results.pop(0))
            
            # Select rest using MMR
            while len(selected_results) < top_k and remaining_results:
                best_mmr_score = -float('inf')
                best_idx = -1
                
                for i, result in enumerate(remaining_results):
                    # Relevance is already scored by the vector DB
                    relevance = result.get("similarity", result["score"])
                    
                    # Calculate diversity by comparing with already selected items
                    max_similarity = 0
                    for selected in selected_results:
                        # Use file/page/section as proxy for content similarity
                        # Exact same page/section = high similarity
                        if (result["file_hash"] == selected["file_hash"] and 
                            result["page_num"] == selected["page_num"]):
                            similarity = 0.9
                            if result.get("header") == selected.get("header"):
                                similarity = 0.95
                        # Same book but different page = medium similarity
                        elif result["file_hash"] == selected["file_hash"]:
                            similarity = 0.7
                        # Different books = low similarity
                        else:
                            similarity = 0.3
                            
                        max_similarity = max(max_similarity, similarity)
                    
                    # Calculate MMR score
                    mmr_score = (1 - diversity_factor) * relevance - diversity_factor * max_similarity
                    
                    if mmr_score > best_mmr_score:
                        best_mmr_score = mmr_score
                        best_idx = i
                
                if best_idx >= 0:
                    selected_results.append(remaining_results.pop(best_idx))
                else:
                    break
                
            return selected_results
            
        except Exception as e:
            print(f"Error in MMR retrieval: {str(e)}")
            # Fall back to standard results
            return initial_results[:top_k]
    
    def get_reading_suggestions(self, query, book_counts=3, excerpt_counts=2):
        """Get reading suggestions based on a query or topic.
        
        Args:
            query: User query or topic
            book_counts: Number of books to suggest
            excerpt_counts: Number of excerpts per book
            
        Returns:
            Reading suggestions with selected excerpts
        """
        # Search across all books
        results_by_book = self.search_by_book(query, top_k=excerpt_counts)
        
        if not results_by_book:
            return {"suggestions": []}
        
        # Get the top books by relevance
        book_relevance = {}
        for book_title, results in results_by_book.items():
            if results:
                avg_score = sum(r.get("similarity", r["score"]) for r in results) / len(results)
                book_relevance[book_title] = avg_score
        
        # Sort books by relevance
        top_books = sorted(book_relevance.items(), key=lambda x: x[1], reverse=True)[:book_counts]
        
        # Build suggestions
        suggestions = []
        for book_title, relevance in top_books:
            book_meta = self.books_metadata.get(book_title, {})
            
            suggestion = {
                "book_title": book_title,
                "author": book_meta.get("author", "Unknown"),
                "relevance": relevance,
                "excerpts": []
            }
            
            # Add excerpts
            for result in results_by_book[book_title][:excerpt_counts]:
                excerpt = {
                    "content": result["content"],
                    "page_num": result["page_num"],
                    "section": result["header"]
                }
                suggestion["excerpts"].append(excerpt)
                
            # Get chapters information if available
            chapters = self.get_book_chapters(book_title)
            if chapters:
                suggestion["relevant_chapters"] = chapters[:3]  # Top 3 chapters
                
            suggestions.append(suggestion)
        
        return {"suggestions": suggestions}

    def search_with_metadata_boost(self, query, metadata_boosts=None, top_k=None):
        """Search with boosting based on metadata fields.
        
        Args:
            query: User query
            metadata_boosts: Dictionary mapping metadata fields to boost values
            top_k: Number of results to return
            
        Returns:
            List of search results with boosted ranking
        """
        if top_k is None:
            top_k = self.top_k
            
        if metadata_boosts is None:
            # Default boosts
            metadata_boosts = {
                "header": 1.2,     # Boost results from important sections
                "page_num": 0.01,  # Small boost for earlier pages (introduction)
                "level": 0.5       # Boost by header level importance
            }
        
        # Get initial results
        results = self.semantic_search(query, top_k=top_k*2)
        
        if not results:
            return []
            
        # Apply metadata boosting
        for result in results:
            # Start with original score
            boosted_score = result.get("similarity", result["score"])
            
            # Apply boosts
            for field, boost_factor in metadata_boosts.items():
                if field == "page_num":
                    # Special handling for page numbers - earlier pages get slight boost
                    page = result.get("page_num", 0)
                    if page <= 20:  # Early pages (preface, intro, etc.)
                        boosted_score += boost_factor
                elif field == "level":
                    # Header importance boost - level 1 headers are most important
                    level = result["metadata"].get("level", 3)
                    if level <= 2:  # Important headers
                        boosted_score += boost_factor * (3 - level)
                elif field in result:
                    # Generic boost if field exists and is not empty
                    if result[field]:
                        boosted_score += boost_factor
                        
            # Store boosted score
            result["boosted_score"] = boosted_score
            
        # Sort by boosted score
        results.sort(key=lambda x: x.get("boosted_score", 0), reverse=True)
        
        return results[:top_k]

    def generate_contextual_summary(self, query, results=None):
        """Generate a contextual summary from retrieved results.
        
        Args:
            query: User query
            results: Optional pre-retrieved results (will retrieve if None)
            
        Returns:
            Contextual summary of the information
        """
        if results is None:
            results = self.semantic_search(query, top_k=5)
            
        if not results:
            return {"summary": "No relevant information found."}
            
        # Collect context from results
        context_parts = []
        for result in results:
            context_parts.append(result["content"])
            
        context = "\n\n".join(context_parts)
        
        # Extract key information (simplified approach)
        key_info = {
            "sources": [f"{r['book_title']} (p. {r['page_num']})" for r in results],
            "topics": self._extract_key_concepts(context),
            "query": query
        }
        
        # A real implementation would use an LLM to generate a summary here
        # For now, we'll just structure the source information
        summary = {
            "query": query,
            "source_count": len(results),
            "sources": key_info["sources"],
            "key_topics": key_info["topics"]
        }
        
        return summary