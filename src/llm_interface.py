import os
from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from langchain_core.messages import HumanMessage, AIMessage

from langchain.schema import BaseRetriever
from langchain.schema.document import Document
from langchain.schema.output_parser import StrOutputParser
from pydantic import Field

from retriever_utils import RerankedRetriever

from dotenv import load_dotenv
import os

# Load variables from .env file into environment
load_dotenv(dotenv_path=".env")

class EducationalLLM:
    """Singleton LLM instance that can be used across different files."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls, api_key=None, model_name="llama3-8b-8192"):
        """Get or create the LLM singleton instance."""
        if cls._instance is None:
            cls._instance = cls(api_key, model_name)
        return cls._instance
    
    def __init__(self, api_key=None, model_name="llama3-8b-8192"):
        """Initialize the LLM with API key."""
        # Use environment variable if API key not provided
        if not api_key:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError(
                    "No Groq API key provided. Either pass the key as an argument or set the GROQ_API_KEY environment variable."
                )
        
        # Set up API key
        os.environ["GROQ_API_KEY"] = api_key
        
        print(f"Setting up Groq LLM with model: {model_name}...")
        
        # Initialize ChatGroq
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=0.2,
            max_tokens=4096
        )

def setup_educational_llm(api_key=None, model_name="llama3-8b-8192"):
    """Set up and return the singleton LLM instance."""
    return EducationalLLM.get_instance(api_key, model_name).llm

class TopicConnectionFinder:
    """Finds connections between topics across chapters and books."""
    
    def __init__(self, llm, retriever):
        """Initialize with LLM and retriever."""
        self.llm = llm
        self.retriever = retriever
        
        # Define the connection-finding prompt
        self.connection_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an educational analysis system specialized in finding connections between topics.
            
            Analyze the following content retrieved from educational materials and identify meaningful connections,
            relationships, and interdependencies between concepts, topics, and ideas across different chapters,
            sections, or books.
            
            CONTENT TO ANALYZE:
            {context}
            
            ANALYSIS QUESTION: {question}
            
            Provide an analysis of the connections with the following structure:
            1. Key topics identified
            2. Cross-chapter connections
            3. Conceptual relationships
            4. Learning progression (how topics build on each other)
            
            CONNECTION ANALYSIS:"""
        )
        
        # Create the connection finder chain
        self.connection_chain = (
            self.connection_prompt 
            | self.llm 
            | StrOutputParser()
        )
    
    def find_connections(self, query, docs=None):
        """Find connections between topics in the retrieved documents."""
        
        # Retrieve relevant documents if not provided
        if docs is None:
            docs = self.retriever.get_relevant_documents(query)
            
        if not docs:
            return "No relevant content found to analyze connections."
        
        # Extract content from docs
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Find connections
        connections = self.connection_chain.invoke({
            "context": context,
            "question": query
        })
        
        return connections

def build_chatbot(retrieval_system, llm, use_reranker=True):
    """Build an enhanced RAG chatbot using the retrieval system with topic connections."""
    
    # Determine if we're receiving a direct retriever or a dictionary
    if isinstance(retrieval_system, dict) and "base_retriever" in retrieval_system:
        base_retriever = retrieval_system["base_retriever"]
    else:
        # Assume retrieval_system is the retriever itself
        base_retriever = retrieval_system
    
    # Initialize the topic connection finder
    connection_finder = TopicConnectionFinder(llm, base_retriever)

    # Create conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Create our enhanced retriever
    enhanced_retriever = RerankedRetriever(
        base_retriever=base_retriever, 
        llm=llm, 
        use_reranker=use_reranker
    )
    
    # Define a prompt template for extracting connections
    connection_prompt = PromptTemplate.from_template(
        """Analyze the following content and identify meaningful connections between topics:
        
        CONTENT:
        {context}
        
        QUESTION:
        {question}
        
        Identify key topics and their relationships:"""
    )
    
    # Define the main QA prompt
    qa_prompt = PromptTemplate.from_template(
        """You are an educational assistant specializing in textbooks and curriculum materials.
        Use the following context and connections to answer the question.
        
        CONTEXT:
        {context}
        
        TOPIC CONNECTIONS:
        {connections}
        
        CHAT HISTORY:
        {chat_history}
        
        QUESTION:
        {question}
        
        Provide a comprehensive answer that:
        1. Directly addresses the question
        2. Highlights relevant connections between concepts and chapters
        3. Explains how different topics relate to each other
        4. Shows the progression of ideas across the educational material
        
        COMPREHENSIVE ANSWER:"""
    )
    
    # Create a custom function for handling the conversation
    def process_query(query_dict):
        question = query_dict.get("question", "")
        chat_history = query_dict.get("chat_history", [])
        
        # Get relevant documents
        docs = enhanced_retriever.invoke(question)
        
        # Extract document content
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Find connections using connection finder
        connections = connection_finder.find_connections(question, docs)
        
        # Format chat history if present
        formatted_history = ""
        if chat_history:
            history_items = []
            for exchange in chat_history:
                if isinstance(exchange, tuple) and len(exchange) == 2:
                    history_items.append(f"Human: {exchange[0]}\nAssistant: {exchange[1]}")
                elif hasattr(exchange, "content"):
                    history_items.append(f"{exchange.type}: {exchange.content}")
            formatted_history = "\n".join(history_items)
        
        # Create input for the prompt
        prompt_input = {
            "context": context,
            "connections": connections,
            "chat_history": formatted_history,
            "question": question
        }
        
        # Generate the answer
        answer = llm.invoke(qa_prompt.format(**prompt_input))
        
        # Return the result in the expected format
        return {
            "answer": answer.content if hasattr(answer, "content") else str(answer),
            "source_documents": docs
        }
    
    return process_query
    
    # Create a function to analyze connections before answering
    def analyze_connections(inputs):
        # Docs already reranked by our custom retriever if enabled
        docs = enhanced_retriever.get_relevant_documents(inputs["question"])
        connections = connection_finder.find_connections(inputs["question"], docs)
        
        # Update inputs with context and connections
        inputs["context"] = "\n\n".join([doc.page_content for doc in docs])
        inputs["connections"] = connections
        inputs["source_documents"] = docs
        
        return inputs
    
    # Create the chatbot chain with connection analysis
    chain = (
        RunnablePassthrough() 
        | RunnableLambda(analyze_connections)
        | {
            "answer": qa_prompt | llm | StrOutputParser(),
            "source_documents": lambda x: x["source_documents"]
        }
    )
    
    # Wrap in ConversationalRetrievalChain for memory management
    qa_chain = ConversationalRetrievalChain(
        combine_docs_chain=chain,
        retriever=enhanced_retriever,  # Use our enhanced retriever here
        memory=memory,
        return_source_documents=True,
        verbose=False
    )
    
    return qa_chain

def answer_with_citations(response):
    """Format response with citations from source documents."""
    answer = response["answer"]
    source_docs = response["source_documents"]
    
    # Build citation string
    citations = []
    seen_sources = set()  # Track unique sources
    
    for i, doc in enumerate(source_docs):
        metadata = doc.metadata
        
        # Create a unique identifier for this source
        source_id = f"{metadata.get('book_title', 'Unknown')}_{metadata.get('page_num', '0')}_{metadata.get('current_header', '')}"
        
        # Only add unique sources
        if source_id not in seen_sources:
            seen_sources.add(source_id)
            
            citation = f"[{len(citations)+1}] {metadata.get('book_title', 'Unknown')} - "
            if metadata.get('current_header'):
                citation += f"{metadata.get('current_header')}, "
            citation += f"Page {metadata.get('page_num', 'Unknown')}"
            citations.append(citation)
    
    # Add citations to answer
    formatted_answer = f"{answer}\n\nSources:\n" + "\n".join(citations)
    return formatted_answer

def get_chat_response(chatbot, question):
    """Get a response from the chatbot for a question with topic connections."""
    try:
        # Note: chatbot is now a function, not a chain object
        response = chatbot({"question": question})
        return answer_with_citations(response)
    except Exception as e:
        return f"Error generating response: {str(e)}. Please try again with a different question."

from response_format import get_formatter, extract_format_request

# Add this to your llm_interface.py file

def get_chat_response_with_format(chatbot, question, llm=None):
    """
    Enhanced version of get_chat_response that supports formatting.
    
    Args:
        chatbot: The chatbot function
        question: User's question
        llm: LLM instance to use for formatting (optional)
        
    Returns:
        Formatted response with citations
    """
    # Check if question contains a formatting request
    cleaned_question, format_type, format_options = extract_format_request(question)
    
    # Use original query if no format detected
    query_to_use = cleaned_question if format_type else question
    
    try:
        # Get regular response - note chatbot is now a function
        response = chatbot({"question": query_to_use})
        
        # Apply formatting if requested
        if format_type:
            # Extract the answer (keep citations separate)
            answer = response["answer"]
            
            # Get formatter
            formatter = get_formatter(llm)
            
            # Format the main content
            formatted_content = formatter.format_response(answer, format_type, **format_options)
            
            # Create new response with formatted content
            formatted_response = {
                "answer": formatted_content,
                "source_documents": response.get("source_documents", [])
            }
            
            # Apply standard citation formatting
            return answer_with_citations(formatted_response)
        else:
            # Return standard response with citations
            return answer_with_citations(response)
    except Exception as e:
        return f"Error generating response: {str(e)}. Please try again with a different question."



# Add this function to your llm_interface.py file
def generate_book_summary_direct(book_title, retrieval_system, llm, processed_data_dir="./processed_data"):
    """Generate a book summary by directly reading processed data files."""
    print(f"Generating summary for book: {book_title}")
    try:
        # Find book data in processed directory
        book_data = None
        book_hash = None
        
        # Debug: Log available directories
        if os.path.exists(processed_data_dir):
            print(f"Processing directory exists. Contents: {os.listdir(processed_data_dir)}")
        else:
            print(f"WARNING: Processing directory '{processed_data_dir}' does not exist")
            return f"Processing directory not found. Please check your configuration."
        
        # Loop through all directories in processed_data
        for file_hash in os.listdir(processed_data_dir):
            metadata_path = os.path.join(processed_data_dir, file_hash, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                        print(f"Found metadata for book: {metadata.get('book_title')}")
                        if metadata.get("book_title") == book_title:
                            book_data = metadata
                            book_hash = file_hash
                            print(f"MATCHED book: {book_title}, hash: {book_hash}")
                            break
                except Exception as e:
                    print(f"Error reading metadata file {metadata_path}: {str(e)}")
        
        if not book_data:
            # Try a more flexible matching approach
            for file_hash in os.listdir(processed_data_dir):
                metadata_path = os.path.join(processed_data_dir, file_hash, "metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                            stored_title = metadata.get("book_title", "").strip()
                            input_title = book_title.strip()
                            
                            # More flexible matching
                            if (stored_title.lower() == input_title.lower() or
                                stored_title.lower() in input_title.lower() or
                                input_title.lower() in stored_title.lower()):
                                book_data = metadata
                                book_hash = file_hash
                                print(f"Matched book with flexible matching: {stored_title}")
                                break
                    except Exception as e:
                        print(f"Error in flexible matching: {str(e)}")
            
            if not book_data:
                return f"No content found for book '{book_title}'. Please check the book title."
        
        # Load chunks
        chunks_path = os.path.join(processed_data_dir, book_hash, "chunks.json")
        if not os.path.exists(chunks_path):
            return f"No chunks found for book '{book_title}'."
        
        with open(chunks_path, "r") as f:
            chunks_data = json.load(f)
        
        if not chunks_data:
            return f"Empty chunks for book '{book_title}'."
        
        print(f"Loaded {len(chunks_data)} chunks for book '{book_title}'")
        
        # Extract content from chunks
        contents = []
        for chunk in chunks_data:
            if isinstance(chunk, dict) and "page_content" in chunk:
                contents.append(chunk["page_content"])
        
        if not contents:
            # Try alternative field names if page_content isn't found
            for chunk in chunks_data:
                if isinstance(chunk, dict):
                    if "content" in chunk:
                        contents.append(chunk["content"])
                    elif "text" in chunk:
                        contents.append(chunk["text"])
        
        if not contents:
            return f"Could not extract content from chunks for book '{book_title}'."
        
        # Limit content to avoid token limits
        book_content = "\n\n".join(contents[:20])  # Take first 20 chunks
        
        # Create summarization prompt
        summary_prompt = PromptTemplate(
            input_variables=["book_title", "book_content"],
            template="""You are an expert educational book summarizer specialized in creating comprehensive and insightful summaries.
            
            Create a detailed summary of the book "{book_title}" based on the following extracted content.
            
            BOOK CONTENT:
            {book_content}
            
            Provide a comprehensive book summary with the following structure:
            1. Overview - What the book covers and its educational purpose
            2. Main Themes - The primary themes and concepts explored
            3. Chapter Breakdown - Brief overview of each chapter/section and its significance
            4. Key Takeaways - The most important lessons and knowledge from this book
            
            COMPREHENSIVE BOOK SUMMARY:"""
        )

        # Create a summarization chain
        summary_chain = (
            summary_prompt 
            | llm 
            | StrOutputParser()
        )

        # Generate summary
        summary = summary_chain.invoke({
            "book_title": book_title,
            "book_content": book_content
        })

        return summary
    except Exception as e:
        traceback.print_exc()
        return f"Error generating summary: {str(e)}"

def generate_book_summary_with_format(book_title, retrieval_system, llm, format_type=None, format_options=None):
    """
    Generate a formatted book summary.
    
    Args:
        book_title: Title of the book
        retrieval_system: Retrieval system
        llm: Language model
        format_type: Format type (optional)
        format_options: Format options (optional)
        
    Returns:
        Formatted book summary
    """
    # Generate summary using the original function
    summary = generate_book_summary(book_title, retrieval_system, llm)
    
    # Apply formatting if requested
    if format_type:
        formatter = get_formatter(llm)
        options = format_options or {}
        return formatter.format_response(summary, format_type, **options)
    
    return summary

def generate_chapter_summary_with_format(book_title, chapter_header, retrieval_system, llm, 
                                       format_type=None, format_options=None):
    """
    Generate a formatted chapter summary.
    
    Args:
        book_title: Title of the book
        chapter_header: Header of the chapter
        retrieval_system: Retrieval system
        llm: Language model
        format_type: Format type (optional)
        format_options: Format options (optional)
        
    Returns:
        Formatted chapter summary
    """
    # Generate summary using the original function
    summary = generate_chapter_summary(book_title, chapter_header, retrieval_system, llm)
    
    # Apply formatting if requested
    if format_type:
        formatter = get_formatter(llm)
        options = format_options or {}
        return formatter.format_response(summary, format_type, **options)
    
    return summary


import json
from langchain.schema.output_parser import StrOutputParser

from langchain.schema.document import Document
import traceback

def generate_book_summary(book_title, retrieval_system, llm, processed_data_dir="./processed_data"):
    """Generate a book summary by directly reading processed data files."""
    print(f"Generating summary for book: {book_title}")
    try:
        # Find book data in processed directory
        book_data = None
        book_hash = None
        
        # Debug: Log available directories
        if os.path.exists(processed_data_dir):
            print(f"Processing directory exists. Contents: {os.listdir(processed_data_dir)}")
        else:
            print(f"WARNING: Processing directory '{processed_data_dir}' does not exist")
            return f"Processing directory not found. Please check your configuration."
        
        # Loop through all directories in processed_data
        for file_hash in os.listdir(processed_data_dir):
            metadata_path = os.path.join(processed_data_dir, file_hash, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                        print(f"Found metadata for book: {metadata.get('book_title')}")
                        if metadata.get("book_title") == book_title:
                            book_data = metadata
                            book_hash = file_hash
                            print(f"MATCHED book: {book_title}, hash: {book_hash}")
                            break
                except Exception as e:
                    print(f"Error reading metadata file {metadata_path}: {str(e)}")
        
        if not book_data:
            # Try a more flexible matching approach
            for file_hash in os.listdir(processed_data_dir):
                metadata_path = os.path.join(processed_data_dir, file_hash, "metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                            stored_title = metadata.get("book_title", "").strip()
                            input_title = book_title.strip()
                            
                            # More flexible matching
                            if (stored_title.lower() == input_title.lower() or
                                stored_title.lower() in input_title.lower() or
                                input_title.lower() in stored_title.lower()):
                                book_data = metadata
                                book_hash = file_hash
                                print(f"Matched book with flexible matching: {stored_title}")
                                break
                    except Exception as e:
                        print(f"Error in flexible matching: {str(e)}")
            
            if not book_data:
                return f"No content found for book '{book_title}'. Please check the book title."
        
        # Load chunks
        chunks_path = os.path.join(processed_data_dir, book_hash, "chunks.json")
        if not os.path.exists(chunks_path):
            return f"No chunks found for book '{book_title}'."
        
        with open(chunks_path, "r") as f:
            chunks_data = json.load(f)
        
        if not chunks_data:
            return f"Empty chunks for book '{book_title}'."
        
        print(f"Loaded {len(chunks_data)} chunks for book '{book_title}'")
        
        # Extract content from chunks
        contents = []
        for chunk in chunks_data:
            if isinstance(chunk, dict) and "page_content" in chunk:
                contents.append(chunk["page_content"])
        
        if not contents:
            # Try alternative field names if page_content isn't found
            for chunk in chunks_data:
                if isinstance(chunk, dict):
                    if "content" in chunk:
                        contents.append(chunk["content"])
                    elif "text" in chunk:
                        contents.append(chunk["text"])
        
        if not contents:
            return f"Could not extract content from chunks for book '{book_title}'."
        
        # Limit content to avoid token limits
        book_content = "\n\n".join(contents[:20])  # Take first 20 chunks
        
        # Create summarization prompt
        summary_prompt = PromptTemplate(
            input_variables=["book_title", "book_content"],
            template="""You are an expert educational book summarizer specialized in creating comprehensive and insightful summaries.
            
            Create a detailed summary of the book "{book_title}" based on the following extracted content.
            
            BOOK CONTENT:
            {book_content}
            
            Provide a comprehensive book summary with the following structure:
            1. Overview - What the book covers and its educational purpose
            2. Main Themes - The primary themes and concepts explored
            3. Chapter Breakdown - Brief overview of each chapter/section and its significance
            4. Key Takeaways - The most important lessons and knowledge from this book
            
            COMPREHENSIVE BOOK SUMMARY:"""
        )

        # Create a summarization chain
        summary_chain = (
            summary_prompt 
            | llm 
            | StrOutputParser()
        )

        # Generate summary
        summary = summary_chain.invoke({
            "book_title": book_title,
            "book_content": book_content
        })

        return summary
    except Exception as e:
        traceback.print_exc()
        return f"Error generating summary: {str(e)}"

def generate_chapter_summary(book_title, chapter_header, retrieval_system, llm):
    """Generate a comprehensive summary of a specific chapter with topic connections."""
    
    # The retriever should be passed directly, no need to extract it
    retriever = retrieval_system
    
    # Create a function to search for content with manual filtering
    def search_chapter_content(query, book_title, chapter_header=None, max_results=10):
        try:
            # Get raw results from vector DB
            raw_results = retriever.semantic_search(query, top_k=max_results*2)
            
            # Filter results manually to only include this book and chapter
            filtered_results = []
            for result in raw_results:
                # Check if it's the right book
                if result.get("book_title") != book_title:
                    continue
                    
                # If chapter header provided, check if it matches
                if chapter_header and result.get("header") != chapter_header:
                    continue
                
                # Convert to Document object if needed
                if not isinstance(result, Document):
                    doc = Document(
                        page_content=result.get("content", ""),
                        metadata=result.get("metadata", {})
                    )
                    filtered_results.append(doc)
                else:
                    filtered_results.append(result)
                    
                # Limit results
                if len(filtered_results) >= max_results:
                    break
            
            return filtered_results
        except Exception as e:
            print(f"Error in chapter content search: {str(e)}")
            return []  # Return empty list on error
    
    # Retrieve content from the specific chapter
    chapter_docs = search_chapter_content(
        f"Complete content and concepts from {chapter_header}",
        book_title,
        chapter_header
    )
    
    # Also retrieve some book-level context to place this chapter in perspective
    book_context_docs = search_chapter_content(
        f"Where does the chapter '{chapter_header}' fit in the overall structure of {book_title}",
        book_title
    )
    
    # Combine all docs
    all_docs = chapter_docs + book_context_docs
    
    if not chapter_docs:
        return f"No content found for chapter '{chapter_header}' in book '{book_title}'."
    
    # Extract content
    chapter_content = "\n\n".join([doc.page_content for doc in chapter_docs 
                                 if hasattr(doc, 'page_content')])
    
    # Find internal topic connections within the chapter
    from llm_interface import TopicConnectionFinder
    connection_finder = TopicConnectionFinder(llm, retriever)
    internal_connections = connection_finder.find_connections(
        f"How do concepts connect within chapter '{chapter_header}'", 
        chapter_docs
    )
    
    # Find connections between this chapter and the rest of the book
    book_connections = connection_finder.find_connections(
        f"How does chapter '{chapter_header}' connect to other chapters in {book_title}",
        all_docs
    )
    
    # Create summarization prompt with connections
    summary_prompt = PromptTemplate(
        input_variables=["book_title", "chapter_header", "chapter_content", "internal_connections", "book_connections"],
        template="""You are an expert educational content summarizer. Create a comprehensive summary of the chapter "{chapter_header}" 
        from the book "{book_title}" based on the following extracted content and connection analyses.

        CHAPTER CONTENT:
        {chapter_content}
        
        INTERNAL TOPIC CONNECTIONS:
        {internal_connections}
        
        CONNECTIONS TO OTHER CHAPTERS:
        {book_connections}

        Provide a comprehensive chapter summary with the following structure:
        1. Chapter Overview - Main focus and purpose
        2. Key Concepts - Primary ideas and principles introduced
        3. Internal Structure - How topics within the chapter relate to each other
        4. Position in Book - How this chapter connects to previous and upcoming chapters
        5. Study Focus - What students should concentrate on from this chapter
        
        COMPREHENSIVE CHAPTER SUMMARY:"""
    )

    # Create a summarization chain
    summary_chain = (
        summary_prompt 
        | llm 
        | StrOutputParser()
    )

    # Generate summary
    summary = summary_chain.invoke({
        "book_title": book_title,
        "chapter_header": chapter_header,
        "chapter_content": chapter_content,
        "internal_connections": internal_connections,
        "book_connections": book_connections
    })

    return summary

def rerank_with_llm(docs, query, llm, top_k=3):
    """Rerank retrieved documents using the LLM to identify most relevant ones."""
    
    if not docs or len(docs) <= top_k:
        return docs
    
    # Extract contents and basic metadata
    doc_contents = []
    for i, doc in enumerate(docs):
        source = f"{doc.metadata.get('book_title', 'Unknown')}"
        if doc.metadata.get('current_header'):
            source += f" - {doc.metadata.get('current_header')}"
        
        doc_contents.append(f"Document {i+1} from {source}:\n{doc.page_content}")
    
    all_docs_text = "\n\n".join(doc_contents)
    
    # Create reranking prompt
    rerank_prompt = PromptTemplate(
        input_variables=["documents", "query"],
        template="""You are an educational content retrieval expert. Your task is to rank documents by their relevance 
        to the query, considering educational value, accuracy, and conceptual depth.
        
        QUERY: {query}
        
        DOCUMENTS:
        {documents}
        
        Analyze each document and rank the top {top_k} most relevant documents for answering this educational query.
        For each document, explain why it's relevant in 1-2 sentences.
        
        FORMAT YOUR RESPONSE AS:
        RANKING:
        1. Document [number]
        2. Document [number]
        3. Document [number]
        
        EXPLANATION:
        Document [number]: [brief explanation of relevance]
        Document [number]: [brief explanation of relevance]
        Document [number]: [brief explanation of relevance]
        """
    )
    
    # Run reranking
    rerank_chain = rerank_prompt | llm | StrOutputParser()
    
    try:
        rerank_result = rerank_chain.invoke({
            "documents": all_docs_text,
            "query": query,
            "top_k": top_k
        })
        
        # Parse rankings (simple approach - extract doc numbers)
        ranking_section = rerank_result.split("EXPLANATION:")[0] if "EXPLANATION:" in rerank_result else rerank_result
        doc_numbers = []
        
        for line in ranking_section.split("\n"):
            if "Document" in line and any(char.isdigit() for char in line):
                # Extract the document number
                try:
                    num = int(''.join(filter(str.isdigit, line.split("Document")[1])))
                    if 1 <= num <= len(docs):  # Valid document number
                        doc_numbers.append(num)
                except:
                    continue
        
        # If successful parsing, return reranked docs
        if doc_numbers:
            # Convert to 0-based indexing and get unique doc numbers
            doc_indices = [num-1 for num in doc_numbers[:top_k]]
            return [docs[i] for i in doc_indices]
    
    except Exception as e:
        print(f"Error in LLM reranking: {str(e)}")
    
    # Fallback - return original docs
    return docs[:top_k]