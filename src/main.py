import os
import uvicorn
import threading
from fastapi import FastAPI, HTTPException, Query, Depends, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import shutil
from pathlib import Path
import tempfile
import time
from sentence_transformers import SentenceTransformer
import subprocess
import sys
import asyncio
import signal

# Import our custom modules
from llm_interface import (
    setup_educational_llm,
    build_chatbot,
    get_chat_response,
    generate_book_summary,
    generate_chapter_summary
)
from load_data import (
    EnhancedPDFProcessor,
    PDFProcessorConfig,
    setup_embeddings
)
from retrieval import EnhancedRetriever
from vector_db import setup_vector_db, get_books_with_metadata, clear_vector_db

from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables at the beginning of the file
load_dotenv()

# Add this context manager at the top of your main.py file
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code that runs on startup
    
    print("Starting lifespan - initializing system components...")
    
    # Start the library watcher thread
    start_library_watcher()
    
    # Auto-initialize the system if environment variables are set
    if os.environ.get("GROQ_API_KEY"):
        try:
            print("Auto-initializing system...")
            
            # Initialize the system with environment variables
            # Create a request object for initialization
            request = InitializeRequest()
            await initialize_system(request)
            
            print("System auto-initialized successfully")
        except Exception as e:
            print(f"Auto-initialization failed: {e}")
    else:
        print("No GROQ_API_KEY found in environment variables. Auto-initialization skipped.")
    
    # Add a longer delay to ensure the API server is fully started
    print("API server starting up - waiting to ensure it's ready...")
    await asyncio.sleep(5)
    
    # Start the UI process
    print("Starting Gradio UI process...")
    start_ui_process()
    
    # Yield control back to FastAPI
    yield
    
    # Code that runs on shutdown
    print("Shutting down API server...")
    
    # Terminate UI process if running
    if global_state["ui_process"]:
        print(f"Shutting down UI process (PID: {global_state['ui_process'].pid})")
        try:
            global_state["ui_process"].terminate()
            global_state["ui_process"].wait(timeout=5)
        except:
            # Force kill if termination doesn't work
            if sys.platform != "win32":  # Not supported on Windows
                os.kill(global_state["ui_process"].pid, signal.SIGKILL)
            else:
                os.system(f"taskkill /F /PID {global_state['ui_process'].pid}")



# Initialize FastAPI app
app = FastAPI(
    title="Digital Library Chatbot",
    description="API for processing, analyzing, and retrieving educational content using LLMs",
    version="1.0.0",
    lifespan=lifespan,
    # Add root_path for proxying on Railway if needed
    root_path=os.environ.get("ROOT_PATH", "")
)

# Add CORS middleware with specific origin for Gradio UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set this to specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Setup directories - using same structure as the Flask version
LIBRARY_DIR = "./ebooks_library"
PROCESSED_DIR = "./processed_data"
VECTOR_DB_PATH = "./qdrant_db"
UPLOAD_DIR = "./uploads"

os.makedirs(LIBRARY_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Create global variables for state
global_state = {
    "embedding_model": None,
    "vector_db": None,
    "llm": None,
    "pdf_processor": None,
    "retriever": None,
    "chatbot": None,
    "is_initialized": False,
    "processing_status": {
        "is_processing": False,
        "processed_books": [],
        "current_book": None
    },
    "ui_process": None
}

# Thread lock for processing status
processing_lock = threading.Lock()

# Pydantic models for API requests and responses
class InitializeRequest(BaseModel):
    groq_api_key: Optional[str] = None
    llm_model: Optional[str] = "llama3-8b-8192"

class QueryRequest(BaseModel):
    query: str
    book_title: Optional[str] = None
    top_k: Optional[int] = 5
    retrieval_mode: Optional[str] = "standard"

class SummaryRequest(BaseModel):
    book_title: str
    chapter_header: Optional[str] = None

class ConnectionRequest(BaseModel):
    query: str
    book_titles: List[str]

class BookInfo(BaseModel):
    title: str
    author: Optional[str] = "Unknown"
    total_pages: Optional[int] = 0
    file_hash: Optional[str] = None

class InitResponse(BaseModel):
    status: str
    message: str
    books: List[BookInfo]

# Dependency to check if system is initialized
def get_initialized_state():
    if not global_state["is_initialized"]:
        # Check if we have the critical components
        if (global_state["retriever"] is not None and 
            global_state["llm"] is not None and
            global_state["chatbot"] is not None):
            print("System components are available but is_initialized flag not set. Setting it now.")
            global_state["is_initialized"] = True
        else:
            raise HTTPException(status_code=400, detail="System not initialized. Call /api/initialize first.")
    return global_state

# Initialize the system components
@app.post("/api/initialize", response_model=InitResponse)
async def initialize_system(request: InitializeRequest = None):
    try:
        # Check if already initialized
        if global_state["is_initialized"] and global_state["retriever"] is not None:
            print("System already initialized, returning current state")
            # Get available books
            books = get_books_with_metadata(PROCESSED_DIR)
            book_info_list = []
            for book in books:
                book_info_list.append(BookInfo(
                    title=book["book_title"],
                    author=book.get("author", "Unknown"),
                    total_pages=book.get("total_pages", 0),
                    file_hash=book.get("file_hash", "")
                ))
            
            return InitResponse(
                status="success",
                message="System already initialized",
                books=book_info_list
            )
            
        # Use environment variables or default values if not provided in request
        groq_api_key = os.environ.get("GROQ_API_KEY", "")
        llm_model = os.environ.get("LLM_MODEL", "llama3-8b-8192")
        
        # If a request is provided with values, they override environment variables
        if request:
            if request.groq_api_key:
                groq_api_key = request.groq_api_key
            if request.llm_model:
                llm_model = request.llm_model
        
        # Validate API key
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is required. Set it in the environment or provide it in the request.")
        
        print("Setting up embeddings...")
        global_state["embedding_model"] = setup_embeddings()
        
        print("Setting up vector database...")
        global_state["vector_db"] = setup_vector_db(
            global_state["embedding_model"], 
            collection_name="ebooks_library", 
            local_path=VECTOR_DB_PATH
        )
        
        if global_state["vector_db"] is None:
            raise RuntimeError("setup_vector_db returned None instead of a valid vector database")
        
        print("Setting up PDF processor...")
        config = PDFProcessorConfig(
            embedding_model=global_state["embedding_model"],
            library_dir=LIBRARY_DIR,
            processed_dir=PROCESSED_DIR,
            use_semantic_chunking=True
        )
        global_state["pdf_processor"] = EnhancedPDFProcessor(config)
        
        print("Setting up retrieval system...")
        global_state["retriever"] = EnhancedRetriever(
            global_state["vector_db"],
            global_state["embedding_model"],
            processed_data_dir=PROCESSED_DIR
        )
        
        print("Setting up LLM...")
        global_state["llm"] = setup_educational_llm(groq_api_key, llm_model)
        
        print("Building chatbot...")
        # Pass the retriever directly
        global_state["chatbot"] = build_chatbot(global_state["retriever"], global_state["llm"])
        
        # Get available books
        books = get_books_with_metadata(PROCESSED_DIR)
        book_info_list = []
        for book in books:
            book_info_list.append(BookInfo(
                title=book["book_title"],
                author=book.get("author", "Unknown"),
                total_pages=book.get("total_pages", 0),
                file_hash=book.get("file_hash", "")
            ))
            
            # Add to processed books list for the watcher
            if book.get("filename") and book.get("filename") not in global_state["processing_status"]["processed_books"]:
                global_state["processing_status"]["processed_books"].append(book.get("filename"))
        
        # SET THE FLAG BEFORE RETURNING - this was missing in your version
        global_state["is_initialized"] = True
        
        print("System initialized successfully")
        
        return InitResponse(
            status="success",
            message="System initialized successfully",
            books=book_info_list
        )
    except Exception as e:
        print(f"Error initializing system: {e}")
        
        # Provide more specific error messages
        if 'vector_db' not in global_state or global_state["vector_db"] is None:
            error_message = "Failed to initialize vector database. Please check the database directory permissions."
        elif 'llm' not in global_state or global_state["llm"] is None:
            error_message = "Failed to initialize language model. Please check your API key and internet connection."
        else:
            error_message = f"Initialization failed: {str(e)}"
            
        raise HTTPException(status_code=500, detail=error_message)
    
# Function to start the UI process
def start_ui_process():
    try:
        ui_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "standalone.py")
        env = os.environ.copy()
        # Make sure we use 127.0.0.1 consistently 
         # For Railway deployment, set environment variables
        if os.environ.get("RAILWAY_STATIC_URL"):
            # If on Railway, the API is at the same URL
            env["API_BASE_URL"] = os.environ.get("RAILWAY_STATIC_URL")
        else:
            # For local development
            env["API_BASE_URL"] = "http://localhost:8000"
            
        # Make sure Gradio binds to all interfaces
        env["GRADIO_SERVER_NAME"] = "0.0.0.0"
        
        # Start the UI process
        print(f"Starting UI process from: {ui_script_path}")
        global_state["ui_process"] = subprocess.Popen(
            [sys.executable, ui_script_path],
            env=env,
            # Don't redirect output so we can see errors directly
        )
        
        print(f"UI process started with PID: {global_state['ui_process'].pid}")
        print(f"Access the Gradio UI at: http://0.0.0.0:7860 or through Railway proxy")
            
    except Exception as e:
        print(f"Error starting UI process: {e}")
            

# Upload a PDF file
@app.post("/api/upload_pdf")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    state: Dict = Depends(get_initialized_state)
):
    try:
        # Validate file is a PDF
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Check if already processing
        if global_state["processing_status"]["is_processing"]:
            raise HTTPException(
                status_code=409, 
                detail=f"Already processing a PDF: {global_state['processing_status']['current_book']}"
            )
            
        # Save file to library directory (using same directory as the Flask version)
        file_path = os.path.join(LIBRARY_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process PDF in background
        background_tasks.add_task(
            process_pdf_in_background,
            file_path,
            state["pdf_processor"],
            state["vector_db"]
        )
        
        return {"status": "success", "message": f"PDF uploaded and processing in background: {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# Background task for PDF processing
def process_pdf_in_background(file_path, pdf_processor, vector_db):
    try:
        print(f"Starting background processing of {file_path}")
        pdf_data, chunks = pdf_processor.process_pdf(file_path, vector_db)
        print(f"Completed processing {file_path}: {len(chunks)} chunks extracted")
        return True
    except Exception as e:
        print(f"Error processing PDF in background: {str(e)}")
        return False

# List all available books
@app.get("/api/books")
async def list_books():
    try:
        # Allow accessing books even if system is not yet initialized
        # This is useful for the UI to check what books are available
        books = get_books_with_metadata(PROCESSED_DIR)
        return {"books": books}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list books: {str(e)}")

# Search for information
@app.post("/api/search")
async def search(request: QueryRequest, state: Dict = Depends(get_initialized_state)):
    try:
        # Create filters if book_title is specified
        filters = None
        if request.book_title:
            filters = {"book_title": request.book_title}
        
        # Perform search
        results = state["retriever"].advanced_retrieval(
            query=request.query,
            retrieval_mode=request.retrieval_mode,
            top_k=request.top_k,
            filters=filters
        )
        
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Chat with the educational material
@app.post("/api/chat")
async def chat_with_content(request: QueryRequest, state: Dict = Depends(get_initialized_state)):
    try:
        # Get response from chatbot
        response = get_chat_response(state["chatbot"], request.query)
        
        return {"query": request.query, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

# Generate book or chapter summary
# Update just the method implementation while keeping the function name and parameters the same
@app.post("/api/summary")
async def generate_summary(request: SummaryRequest, state: Dict = Depends(get_initialized_state)):
    try:
        # Check if book exists
        books_metadata = get_books_with_metadata(PROCESSED_DIR)
        exact_match = False
        for book in books_metadata:
            if book["book_title"] == request.book_title:
                exact_match = True
                break
                
        if not exact_match:
            # Try more flexible matching
            found = False
            for book in books_metadata:
                stored_title = book["book_title"].strip()
                input_title = request.book_title.strip()
                if (stored_title.lower() == input_title.lower() or
                    stored_title.lower() in input_title.lower() or
                    input_title.lower() in stored_title.lower()):
                    # Use the correct book title from metadata
                    request.book_title = book["book_title"]
                    found = True
                    break
            
            if not found:
                raise HTTPException(status_code=404, detail=f"Book not found: {request.book_title}")
        
        # Get the retriever 
        retriever = state["retriever"]
        
        # Generate appropriate summary
        # Import our direct summary function
        from llm_interface import generate_book_summary_direct
        
        if request.chapter_header:
            # Chapter summary
            summary = generate_chapter_summary(
                request.book_title,
                request.chapter_header,
                retriever,  # Pass retriever directly
                state["llm"]
            )
            summary_type = "chapter"
        else:
            # Use our direct access function instead
            summary = generate_book_summary_direct(
                request.book_title,
                retriever,  # Pass retriever for compatibility 
                state["llm"],
                processed_data_dir=PROCESSED_DIR
            )
            summary_type = "book"
        
        return {
            "book_title": request.book_title,
            "chapter": request.chapter_header,
            "summary_type": summary_type,
            "summary": summary
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Summary generation error: {str(e)}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")

# Find connections between topics
@app.post("/api/connections")
async def find_connections(request: ConnectionRequest, state: Dict = Depends(get_initialized_state)):
    try:
        # Check if specified books exist
        available_books = [book["book_title"] for book in get_books_with_metadata(PROCESSED_DIR)]
        for book in request.book_titles:
            if book not in available_books:
                raise HTTPException(status_code=404, detail=f"Book not found: {book}")
        
        # Compare information across books
        comparison = state["retriever"].compare_information(request.query, request.book_titles)
        
        return comparison
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Finding connections failed: {str(e)}")

# Get document structure (TOC)
@app.get("/api/book_structure/{book_title}")
async def get_book_structure(book_title: str, state: Dict = Depends(get_initialized_state)):
    try:
        toc = state["retriever"].get_document_structure(book_title)
        if not toc:
            raise HTTPException(status_code=404, detail=f"Structure not found for book: {book_title}")
            
        return {"book_title": book_title, "structure": toc}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get book structure: {str(e)}")

# Get reading suggestions
@app.post("/api/reading_suggestions")
async def get_reading_suggestions(request: QueryRequest, state: Dict = Depends(get_initialized_state)):
    try:
        suggestions = state["retriever"].get_reading_suggestions(request.query)
        return suggestions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get reading suggestions: {str(e)}")

# Get processing status
@app.get("/api/status")
async def get_processing_status():
    # Allow checking status even if not initialized
    return global_state["processing_status"]

# Reset system (clear vector database)
@app.post("/api/reset")
async def reset_system(state: Dict = Depends(get_initialized_state)):
    try:
        # Clear vector database
        success = clear_vector_db(state["vector_db"])
        
        # Reset global state
        global_state["is_initialized"] = False
        
        # Reset processing status
        with processing_lock:
            global_state["processing_status"] = {
                "is_processing": False,
                "processed_books": [],
                "current_book": None
            }
        
        return {"status": "success" if success else "partial", "message": "System reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

# Add a health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "initialized": global_state["is_initialized"],
        "processing": global_state["processing_status"]["is_processing"]
    }

# Add a watcher function similar to the Flask version
def start_library_watcher():
    """Start a background thread that watches for new PDFs in the library directory"""
    def watch_library_folder():
        """Monitor the library folder for new PDFs and process them."""
        while True:
            # Only process if system is initialized and not already processing
            if (global_state["is_initialized"] and 
                not global_state["processing_status"]["is_processing"]):
                
                try:
                    # Get PDF processor and vector DB
                    pdf_processor = global_state["pdf_processor"]
                    vector_db = global_state["vector_db"]
                    
                    # Get list of already processed books
                    processed_books = global_state["processing_status"]["processed_books"]
                    
                    # Look for new PDFs
                    for filename in os.listdir(LIBRARY_DIR):
                        if filename.endswith('.pdf') and filename not in processed_books:
                            # Found a new PDF - process it
                            with processing_lock:
                                global_state["processing_status"]["is_processing"] = True
                                global_state["processing_status"]["current_book"] = filename
                            
                            pdf_path = os.path.join(LIBRARY_DIR, filename)
                            print(f"Watcher detected new PDF: {filename}")
                            
                            try:
                                pdf_data, chunks = pdf_processor.process_pdf(pdf_path, vector_db)
                                print(f"Watcher processed {filename}: {len(chunks)} chunks extracted")
                                
                                # Update processed books list
                                with processing_lock:
                                    global_state["processing_status"]["processed_books"].append(filename)
                            except Exception as e:
                                print(f"Error processing PDF in watcher: {str(e)}")
                            finally:
                                with processing_lock:
                                    global_state["processing_status"]["is_processing"] = False
                                    global_state["processing_status"]["current_book"] = None
                            
                            # Process one file at a time
                            break
                                
                except Exception as e:
                    print(f"Error in library folder watcher: {e}")
            
            # Sleep to prevent high CPU usage
            time.sleep(60)  # Check every minute
    
    # Start watcher thread
    watcher_thread = threading.Thread(target=watch_library_folder, daemon=True)
    watcher_thread.start()
    print("PDF processing watcher thread started successfully")

# Function to monitor UI process output
async def monitor_ui_process():
    if global_state["ui_process"]:
        # Non-blocking read from subprocess stdout and stderr
        for line in iter(global_state["ui_process"].stdout.readline, b''):
            if line:
                print(f"UI> {line.decode('utf-8').strip()}")
        
        for line in iter(global_state["ui_process"].stderr.readline, b''):
            if line:
                print(f"UI Error> {line.decode('utf-8').strip()}")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add this function near your other API endpoints in main.py
@app.get("/")
async def root():
    """Root endpoint that provides basic API information and UI redirect info"""
    return {
        "app": "Educational Content Analysis API",
        "version": "1.0.0",
        "status": "running",
        "ui_url": "http://localhost:7860",  # UI is running on this port
        "endpoints": {
            "api/initialize": "Initialize the system",
            "api/upload_pdf": "Upload a PDF file",
            "api/books": "List all available books",
            "api/search": "Search for information",
            "api/chat": "Chat with the educational material",
            "api/summary": "Generate book or chapter summary",
            "api/connections": "Find connections between topics",
            "api/book_structure/{book_title}": "Get document structure (TOC)",
            "api/reading_suggestions": "Get reading suggestions",
            "api/status": "Get processing status",
            "api/reset": "Reset system",
            "health": "Health check endpoint"
        }
    }

# Entry point for running the application
# if __name__ == "__main__":
#     # Run with reload enabled as requested
#     import uvicorn
#     print("Starting FastAPI server with hot reload on 127.0.0.1:8000...")
#     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, log_level="debug")

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    # For Railway, we need to bind to 0.0.0.0
    host = "0.0.0.0"
    
    print(f"Starting FastAPI server on {host}:{port}...")
    
    # Disable reload in production (on Railway)
    reload_mode = False if os.environ.get("RAILWAY_STATIC_URL") else True
    
    uvicorn.run("main:app", host=host, port=port, reload=reload_mode, log_level="info")