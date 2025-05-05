import os
import gradio as gr
import time
import json
import requests
from dotenv import load_dotenv
from urllib.parse import urljoin

# Load environment variables
load_dotenv()

# Configuration - Use localhost instead of 0.0.0.0
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

# Print for debugging
print(f"Connecting to backend at: {API_BASE_URL}")

# Function to make API calls to FastAPI backend with improved error handling
def api_call(endpoint, method="GET", data=None, files=None, max_retries=3, retry_delay=2):
    url = urljoin(API_BASE_URL, endpoint)
    
    print(f"Making API call to: {url} (method: {method})")
    
    for attempt in range(max_retries):
        try:
            if method == "GET":
                print(f"Sending GET request to {url}")
                response = requests.get(url, timeout=15)
            elif method == "POST":
                if files:
                    print(f"Sending POST request with files to {url}")
                    response = requests.post(url, files=files, timeout=15)
                else:
                    print(f"Sending POST request with data to {url}")
                    print(f"Data: {data}")
                    response = requests.post(url, json=data, timeout=15)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()  # Raise exception for HTTP errors
            print(f"Received response from {url}: Status code {response.status_code}")
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API call attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            else:
                print(f"All {max_retries} attempts failed for {url}")
                return {"error": str(e)}

# Get available books from the API
def get_books():
    try:
        response = api_call("/api/books")
        if "error" in response:
            print(f"Error getting books: {response['error']}")
            return []
        return [book["book_title"] for book in response["books"]]
    except Exception as e:
        print(f"Error getting books: {e}")
        return []

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

# Initialize system with retry logic
def initialize_system():
    max_retries = 3
    retry_delay = 3
    
    for attempt in range(max_retries):
        try:
            # Attempt to initialize the system using backend-configured credentials
            response = api_call("/api/initialize", method="POST", data={})
            
            if "error" in response:
                if attempt < max_retries - 1:
                    print(f"Initialization attempt {attempt+1} failed, retrying...")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5  # Exponential backoff
                    continue
                return f"Initialization failed: {response['error']}"
            
            num_books = len(response.get("books", []))
            return f"System initialized successfully. Found {num_books} books."
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Initialization attempt {attempt+1} failed, retrying...")
                time.sleep(retry_delay)
                retry_delay *= 1.5
                continue
            return f"Initialization failed: {str(e)}"

# Chat function to communicate with the API
def chat(message, history, book_title):
    """Chat function that communicates with the API and returns tuples for Gradio."""
    if not book_title:
        return history + [(message, "Please select a book first")]
    
    try:
        data = {
            "query": message,
            "book_title": book_title
        }
        response = api_call("/api/chat", method="POST", data=data)
        
        if "error" in response:
            return history + [(message, f"Error: {response['error']}")]
        
        # Return in tuple format (user_message, bot_message)
        return history + [(message, response.get("response", "No response received"))]
    except Exception as e:
        print(f"Error in chat: {e}")
        return history + [(message, f"Error: {str(e)}")]

# Get book summary function
def get_book_summary(book_title):
    if not book_title:
        return "Please select a book first"
    
    try:
        data = {
            "book_title": book_title
        }
        response = api_call("/api/summary", method="POST", data=data)
        
        if "error" in response:
            return f"Error: {response['error']}"
        
        return response.get("summary", "No summary available")
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Check API health and get system status with more robust connection checking
def check_api_health():
    global API_BASE_URL
    
    print("Checking API health...")
    
    # Try the connection
    try:
        health_url = urljoin(API_BASE_URL, "/health")
        print(f"Testing connection to: {health_url}")
        
        response = requests.get(health_url, timeout=15)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            print(f"Successfully connected to API at: {API_BASE_URL}")
            health_data = response.json()
            status_message = f"API Status: {health_data.get('status', 'unknown')}, Initialized: {health_data.get('initialized', False)}"
            return True, status_message
    except Exception as e:
        print(f"Error connecting to API: {e}")
    
    return False, "Could not connect to API backend - please make sure the server is running"

# Auto-initialize with connection retry logic
def auto_initialize_on_startup():
    # Add a delay to ensure the API server has started
    print("Waiting for API server to start...")
    time.sleep(5)  # Give the API server time to start
    
    # First, check if API is responsive with multiple retries
    for attempt in range(3):
        healthy, health_message = check_api_health()
        if healthy:
            print(f"API health check succeeded on attempt {attempt+1}")
            break
        print(f"API health check attempt {attempt+1} failed, retrying...")
        time.sleep(3)
    
    if not healthy:
        print(f"API health check failed after multiple attempts: {health_message}")
        return f"Waiting for API to become available. Please wait and click 'Check Connection'."
    
    # If API is healthy, try to initialize
    try:
        init_message = initialize_system()
        print(f"Auto-initialization result: {init_message}")
        return init_message
    except Exception as e:
        return f"Auto-initialization failed: {str(e)}"

# Manual connection check that updates the UI
def manual_health_check():
    healthy, message = check_api_health()
    books = get_books() if healthy else []
    default_value = books[0] if books else None
    
    # Return multiple values to update multiple UI components
    return message, gr.Dropdown(choices=books, value=default_value), "System connected" if healthy else "System not connected"

# Give API some time to start before checking health the first time
print("Waiting for API to start before first health check...")
time.sleep(3)

# Get initial books (if API is available)
is_api_healthy, health_message = check_api_health()
initial_books = get_books() if is_api_healthy else []
initial_value = initial_books[0] if initial_books else None

# Auto-initialize on startup
init_status_message = auto_initialize_on_startup() if is_api_healthy else "API connection failed - click 'Check Connection'"

# Create the Gradio interface
with gr.Blocks(title="Book Chat") as demo:
    gr.Markdown("# Chat with Your Books")
    
    # API Status with refresh button
    with gr.Row():
        api_status = gr.Markdown(f"*API Status: {health_message}*")
        check_connection_btn = gr.Button("Check Connection")
    
    with gr.Row():
        with gr.Column(scale=1):
            # System status display
            gr.Markdown("## System Status")
            init_status = gr.Textbox(label="System Status", value=init_status_message)
            
            # Book selection area
            gr.Markdown("## Book Selection")
            book_search = gr.Textbox(label="Search Books", placeholder="Type to search...")
            book_selection = gr.Dropdown(
                label="Select a Book", 
                choices=initial_books,
                value=initial_value,
                allow_custom_value=False
            )
            
            # Refresh button
            refresh_btn = gr.Button("Refresh Book List")
            
            # Book summary
            gr.Markdown("## Book Summary")
            summary_btn = gr.Button("Get Book Summary")
            summary_output = gr.Textbox(label="Book Summary", lines=8)
        
        with gr.Column(scale=2):
            # Chat interface (updated to use the new message format)
            gr.Markdown("## Chat with your Book")
            chatbot = gr.Chatbot(
                label="Conversation", 
                height=500, 
                show_label=True,
                avatar_images=["👤", "🤖"]  
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
    check_connection_btn.click(manual_health_check, inputs=[], outputs=[api_status, book_selection, init_status])
    
    # Conversation with proper message formatting
    msg.submit(chat, inputs=[msg, chatbot, book_selection], outputs=[chatbot]).then(
        lambda: "", None, msg
    )
    
    # Clear function that returns an empty list
    clear.click(lambda: [], None, chatbot)
    summary_btn.click(get_book_summary, inputs=book_selection, outputs=summary_output)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    
    # Print connection information for troubleshooting
    print(f"Gradio UI starting on port: {port}")
    print(f"Attempting to connect to backend API at: {API_BASE_URL}")
    
    # Launch the UI with increased timeout
    demo.launch(server_name="127.0.0.1", server_port=port, debug=True)