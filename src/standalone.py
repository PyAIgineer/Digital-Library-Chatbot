import os
import gradio as gr
import time
from dotenv import load_dotenv
import json
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq  # Import directly

# Import only what we need to avoid problematic imports
from vector_db import setup_vector_db
from load_data import EnhancedPDFProcessor, PDFProcessorConfig, setup_embeddings

# Configuration
VECTOR_DB_PATH = "./qdrant_db"
LIBRARY_DIR = "./ebooks_library"  
PROCESSED_DIR = "./processed_data"

# Global system variable
system = None

# Load environment variables
load_dotenv()

# Setup LLM directly
def setup_llm_direct(api_key=None, model_name="llama3-8b-8192"):
    """Set up the LLM without using the llm_interface module."""
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("No API key found. Please set GROQ_API_KEY or OPENROUTER_API_KEY in .env file")
    
    print(f"Setting up Groq LLM with model: {model_name}...")
    
    # Initialize ChatGroq directly
    llm = ChatGroq(
        model_name=model_name,
        temperature=0.7,
        max_tokens=4096,
        api_key=api_key
    )
    
    return llm

# Create a chatbot directly
def create_direct_chatbot(vector_db, llm):
    """Create a chatbot directly without using the problematic RerankedRetriever class."""
    
    # Basic conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Educational prompt template
    qa_prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template="""You are an educational assistant specializing in textbooks.
        
        Use the following pieces of context to answer the question at the end.
        
        CONTEXT:
        {context}
        
        CHAT HISTORY:
        {chat_history}
        
        QUESTION:
        {question}
        
        Provide a comprehensive answer that:
        1. Directly addresses the question
        2. Explains concepts clearly and accurately 
        3. Uses examples when helpful
        
        COMPREHENSIVE ANSWER:"""
    )
    
    # Create the chatbot using ConversationalRetrievalChain
    chatbot = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
        return_generated_question=False
    )
    
    return chatbot

# Initialize the system
def initialize_system():
    global system
    if system is None:
        print("Initializing system and processing books...")

        # Setup embeddings
        embeddings = setup_embeddings()
        
        # Setup vector database
        vector_db = setup_vector_db(embeddings, local_path=VECTOR_DB_PATH)
        
        # Setup LLM directly
        llm = setup_llm_direct()
        
        # Setup PDF processor
        config = PDFProcessorConfig(
            embedding_model=embeddings,
            library_dir=LIBRARY_DIR,
            processed_dir=PROCESSED_DIR,
            use_semantic_chunking=True
        )
        pdf_processor = EnhancedPDFProcessor(config)
        
        # Build chatbot directly
        chatbot = create_direct_chatbot(vector_db, llm)
        
        system = {
            "chatbot": chatbot,
            "llm": llm,
            "vector_db": vector_db,
            "pdf_processor": pdf_processor,
            "embeddings": embeddings
        }

        process_existing_books(vector_db, pdf_processor)
        time.sleep(2)

    return system

# Process existing books
def process_existing_books(vector_db, pdf_processor):
    # Ensure library directory exists
    os.makedirs(LIBRARY_DIR, exist_ok=True)
    
    # Get list of PDFs in the directory
    pdf_files = [f for f in os.listdir(LIBRARY_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {LIBRARY_DIR}")
        return
    
    print(f"Found {len(pdf_files)} PDF files in {LIBRARY_DIR}")
    
    # Process each PDF
    for filename in pdf_files:
        pdf_path = os.path.join(LIBRARY_DIR, filename)
        print(f"Processing {filename}...")
        
        try:
            # Process PDF with enhanced processor
            pdf_data, chunks = pdf_processor.process_pdf(pdf_path, vector_db)
            print(f"✓ Processed {len(chunks)} chunks from {filename}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Get available books
def get_books():
    # Get books from processed data directory
    books = []
    if os.path.exists(PROCESSED_DIR):
        for file_hash in os.listdir(PROCESSED_DIR):
            metadata_path = os.path.join(PROCESSED_DIR, file_hash, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                        books.append(metadata["book_title"])
                except Exception as e:
                    print(f"Error loading metadata for {file_hash}: {e}")
    
    # Fallback to listing PDFs if no processed books found
    if not books:
        if os.path.exists(LIBRARY_DIR):
            books = [os.path.splitext(f)[0] for f in os.listdir(LIBRARY_DIR) if f.endswith('.pdf')]
    
    print(f"Found {len(books)} books: {books}")
    return books

# Function to refresh the book list
def refresh_books():
    books = get_books()
    default_value = books[0] if books else None
    return gr.Dropdown(choices=books, value=default_value)

# Function to filter books
def filter_books(search_text):
    all_books = get_books()
    if not search_text:
        return all_books
    
    filtered_books = [book for book in all_books if search_text.lower() in book.lower()]
    return filtered_books

# Chat function for continuous conversation
# Function to chat with the book
def chat(message, history, book_title):
    if not book_title:
        # Return in the format expected by Gradio chatbot (list of tuples)
        return history + [(message, "Please select a book first")]
    
    # Initialize system if needed
    system = initialize_system()
    
    # Modify query to focus on the selected book
    focused_query = f"About the book '{book_title}': {message}"
    print(f"Focused query: '{focused_query}'")
    
    try:
        # Get response from chatbot
        result = system["chatbot"].invoke({"question": focused_query})
        response = result["answer"]
        
        # Format response with source citations if available
        if "source_documents" in result and result["source_documents"]:
            sources = []
            for i, doc in enumerate(result["source_documents"][:3]):  # Limit to top 3 sources
                metadata = doc.metadata
                source = f"[{i+1}] {metadata.get('book_title', 'Unknown')}"
                if metadata.get('current_header'):
                    source += f" - {metadata.get('current_header')}"
                if metadata.get('page_num'):
                    source += f", Page {metadata.get('page_num')}"
                sources.append(source)
            
            if sources:
                response += "\n\nSources:\n" + "\n".join(sources)
    except Exception as e:
        print(f"Error in chat: {e}")
        response = f"Sorry, I encountered an error: {str(e)}"
    
    # Return updated history with the new message and response as a list of tuples
    # This is the format expected by Gradio's chatbot component
    return history + [(message, response)]

# Get book summary function
def get_book_summary(book_title):
    if not book_title:
        return "Please select a book first"
    
    # Initialize system if needed
    system = initialize_system()
    
    try:
        # Create a direct query to get relevant content from the book
        results = system["vector_db"].similarity_search(
            f"Summary of the book {book_title}", k=10
        )
        
        # Extract content from the search results
        content = "\n\n".join([doc.page_content for doc in results])
        
        # Create summary prompt
        summary_prompt = PromptTemplate(
            input_variables=["book_title", "content"],
            template="""You are an expert book summarizer.
            Create a comprehensive summary of the book "{book_title}" based on the following content:
            
            CONTENT:
            {content}
            
            Your summary should include:
            1. Main themes and topics
            2. Key concepts and ideas
            3. Overall structure and flow
            4. Learning outcomes and significance
            
            BOOK SUMMARY:"""
        )
        
        # Generate summary
        summary_input = {
            "book_title": book_title,
            "content": content
        }
        
        summary = summary_prompt.format(**summary_input)
        result = system["llm"].invoke(summary)
        
        return result.content
    except Exception as e:
        print(f"Error generating summary: {e}")
        return f"Sorry, I encountered an error generating the summary: {str(e)}"

# Function to upload and process a new PDF
def upload_pdf(file):
    if not file:
        return "No file uploaded"
    
    # Initialize system if needed
    system = initialize_system()
    
    # Save the file to the library directory
    filename = os.path.basename(file.name)
    file_path = os.path.join(LIBRARY_DIR, filename)
    
    try:
        # Copy the uploaded file to the library directory
        with open(file_path, "wb") as f:
            f.write(file.read())
        
        # Process the PDF
        pdf_data, chunks = system["pdf_processor"].process_pdf(file_path, system["vector_db"])
        
        return f"Successfully uploaded and processed {filename} with {len(chunks)} chunks."
    except Exception as e:
        return f"Error processing {filename}: {str(e)}"

# Initialize the system
try:
    initialize_system()
    time.sleep(2)
except Exception as e:
    print(f"Error during initialization: {e}")

# Get books after system is initialized
initial_books = get_books()
print(f"Available books: {initial_books}")
initial_value = initial_books[0] if initial_books else None

# Create the Gradio interface
with gr.Blocks(title="Book Chat") as demo:
    gr.Markdown("# Chat with Your Books")
    
    with gr.Row():
        with gr.Column(scale=1):
            # Book selection area
            book_search = gr.Textbox(label="Search Books", placeholder="Type to search...")
            book_selection = gr.Dropdown(
                label="Select a Book", 
                choices=initial_books,
                value=initial_value,
                allow_custom_value=True
            )
            
            # PDF upload area
            upload_button = gr.UploadButton("Upload PDF", file_types=[".pdf"])
            upload_output = gr.Textbox(label="Upload Status")
            refresh_btn = gr.Button("Refresh Book List")
            
            # Book summary
            summary_btn = gr.Button("Get Book Summary")
            summary_output = gr.Textbox(label="Book Summary", lines=8)
        
        with gr.Column(scale=2):
            # Chat interface - using messages format
            chatbot = gr.Chatbot(
                label="Conversation", 
                height=500, 
                show_label=True
            )
            msg = gr.Textbox(
                label="Your Message", 
                placeholder="Ask about the selected book...",
                show_label=True
            )
            clear = gr.Button("Clear Chat")
    
    # Event handlers
    book_search.change(filter_books, inputs=book_search, outputs=book_selection)
    refresh_btn.click(refresh_books, inputs=[], outputs=book_selection)
    upload_button.upload(upload_pdf, inputs=[upload_button], outputs=[upload_output])
    
    # Conversation with proper message formatting
    msg.submit(chat, inputs=[msg, chatbot, book_selection], outputs=[chatbot]).then(
        lambda: "", None, msg
    )
    
    # Updated clear function that returns an empty list
    clear.click(lambda: [], None, chatbot)
    summary_btn.click(get_book_summary, inputs=book_selection, outputs=summary_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

