import os
import gradio as gr
import time
import json
import requests
from dotenv import load_dotenv

# Load environment variables (to get API key from .env file)
load_dotenv()

# Configuration
API_BASE_URL = "http://localhost:7860"  # Your FastAPI server address
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Global variables
system_initialized = False

# Check if the API is available
def check_api_available():
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        return response.status_code == 200
    except Exception:
        return False

# Get available books from FastAPI or local directory
def get_books():
    try:
        # First try the API
        if check_api_available():
            response = requests.get(f"{API_BASE_URL}/api/books")
            if response.status_code == 200:
                books_data = response.json().get("books", [])
                return [book["book_title"] for book in books_data]
    except Exception as e:
        print(f"Error getting books from API: {e}")
    
    # Fallback to local files
    books = []
    processed_dir = "./processed_data"
    library_dir = "./ebooks_library"
    
    # Check processed directory first
    if os.path.exists(processed_dir):
        for file_hash in os.listdir(processed_dir):
            metadata_path = os.path.join(processed_dir, file_hash, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                        books.append(metadata["book_title"])
                except Exception as e:
                    print(f"Error loading metadata for {file_hash}: {e}")
    
    # Fallback to listing PDFs if no processed books found
    if not books:
        if os.path.exists(library_dir):
            books = [os.path.splitext(f)[0] for f in os.listdir(library_dir) if f.endswith('.pdf')]
    
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

# Chat function that works with either API or direct
def chat(message, history, book_title):
    if not book_title:
        return history + [(message, "Please select a book first")]
    
    try:
        # Modify query to focus on the selected book (just like in standalone.py)
        focused_query = f"About the book '{book_title}': {message}"
        print(f"Focused query: '{focused_query}'")
        
        # First try to use the API if available
        if check_api_available():
            response = requests.post(
                f"{API_BASE_URL}/api/chat",
                json={
                    "query": focused_query,
                    "book_title": book_title
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return history + [(message, result["response"])]
        
        # If API fails, fallback to standalone mode logic
        # In this case, inform the user to use standalone.py directly
        return history + [(message, "Could not connect to API. Please run standalone.py directly.")]
        
    except Exception as e:
        print(f"Error in chat: {e}")
        return history + [(message, f"Sorry, I encountered an error: {str(e)}")]

# Get book summary function
def get_book_summary(book_title):
    if not book_title:
        return "Please select a book first"
    
    try:
        # Try to use the API if available
        if check_api_available():
            response = requests.post(
                f"{API_BASE_URL}/api/summary",
                json={
                    "book_title": book_title
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["summary"]
        
        # If API fails, fallback to standalone mode logic
        return "Could not connect to API. Please run standalone.py directly."
        
    except Exception as e:
        print(f"Error generating summary: {e}")
        return f"Sorry, I encountered an error generating the summary: {str(e)}"

# Function to upload and process a new PDF
def upload_pdf(file):
    if not file:
        return "No file uploaded"
    
    try:
        # Try to use the API if available
        if check_api_available():
            # Create multipart form data
            files = {"file": (os.path.basename(file.name), open(file.name, "rb"), "application/pdf")}
            
            # Call FastAPI upload endpoint
            response = requests.post(f"{API_BASE_URL}/api/upload_pdf", files=files)
            
            if response.status_code == 200:
                result = response.json()
                return result["message"]
            elif response.status_code == 400:
                error_msg = response.json().get("detail", "Unknown error")
                if "System not initialized" in error_msg:
                    # Initialize the system if needed
                    init_response = requests.post(
                        f"{API_BASE_URL}/api/initialize", 
                        json={
                            "groq_api_key": GROQ_API_KEY,
                            "llm_model": "llama3-8b-8192"
                        }
                    )
                    
                    if init_response.status_code == 200:
                        # Try upload again
                        response = requests.post(f"{API_BASE_URL}/api/upload_pdf", files=files)
                        if response.status_code == 200:
                            result = response.json()
                            return result["message"]
                
                return f"Upload failed: {error_msg}"
        
        # If API fails, fallback to standalone mode logic
        return "Could not connect to API. Please run standalone.py directly."
        
    except Exception as e:
        return f"Error uploading file: {str(e)}"

# Get initial books
initial_books = get_books()
initial_book = initial_books[0] if initial_books else None

# Try to initialize the system if API is available and we have an API key
if check_api_available() and GROQ_API_KEY:
    try:
        init_response = requests.post(
            f"{API_BASE_URL}/api/initialize", 
            json={
                "groq_api_key": GROQ_API_KEY,
                "llm_model": "llama3-8b-8192"
            }
        )
        if init_response.status_code == 200:
            print("Successfully initialized API")
        else:
            print(f"API initialization failed: {init_response.json().get('detail', 'Unknown error')}")
    except Exception as e:
        print(f"Could not initialize API: {e}")

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
                value=initial_book,
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
    # Use a different port than your FastAPI server
    print(f"Starting Gradio interface on port 7861, connecting to API at {API_BASE_URL}")
    print(f"API {'is' if check_api_available() else 'is not'} available")
    demo.launch(server_name="0.0.0.0", server_port=7861)