# Digital Library Chatbot

An intelligent chatbot system that helps users navigate, search, and interact with educational books in a digital library. The system processes PDF documents and enables natural language conversations about their content.

## Features

- **PDF Library**: Upload and process educational books and documents
- **Intelligent Chat**: Have natural conversations about book content
- **Semantic Search**: Quickly find specific information across your digital library
- **Book Summaries**: Generate concise summaries of entire books or individual chapters
- **Topic Exploration**: Discover connections between concepts across different books
- **Reading Recommendations**: Get personalized reading suggestions
- **Visual Exploration**: Navigate book structure through a user-friendly interface

## System Overview

The Digital Library Chatbot combines several technologies:

- **LLM Integration**: Leverages Groq's LLM for natural language understanding and generation
- **Vector Search**: Uses embeddings to find semantically relevant content
- **Document Processing**: Extracts structure and content from educational PDFs
- **Interactive UI**: Provides a conversational interface for library exploration

## Prerequisites

- Python 3.10+
- Docker and Docker Compose (for containerized deployment)
- Groq API key (or other supported LLM provider)
- 8GB+ RAM recommended for PDF processing

## Installation

### Using Docker (Recommended)

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/digital-library-chatbot.git
   cd digital-library-chatbot
   ```

2. Create and configure your environment file:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

3. Run the start script to set up and launch the application:
   ```bash
   chmod +x start.sh
   ./start.sh
   ```

4. Access the chatbot interface at http://localhost:7860

### Manual Installation

1. Clone the repository and create a virtual environment:
   ```bash
   git clone https://github.com/yourusername/digital-library-chatbot.git
   cd digital-library-chatbot
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

4. Create required directories:
   ```bash
   mkdir -p ebooks_library processed_data qdrant_db uploads src/static
   ```

5. Run the application:
   ```bash
   python src/main.py
   ```

6. Access the chatbot interface at http://localhost:7860

## User Guide

### Adding Books to Your Library

1. Access the chatbot interface and click on "Refresh Book List" to see your current library
2. Upload new PDFs using the upload function in the interface
3. The system will automatically process the books and make them available for chat

### Chatting with Your Books

1. Select a book from your library using the dropdown menu
2. Type your questions or queries in the chat box
3. The chatbot will respond with relevant information and citations to specific pages

### Example Queries

- "What are the main themes of this book?"
- "Summarize Chapter 3 for me"
- "Explain the concept of [specific topic] from this book"
- "How does this book approach [specific subject]?"
- "Find information about [specific topic] across all my books"
- "What does the author say about [specific topic]?"
- "Compare how different books in my library discuss [topic]"

### Getting Book Summaries

1. Select the book you want to learn about
2. Click "Get Book Summary" to generate a comprehensive overview

## API Reference

The system provides a RESTful API with the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/initialize` | POST | Initialize the system |
| `/api/upload_pdf` | POST | Upload a PDF file to the library |
| `/api/books` | GET | List all available books |
| `/api/search` | POST | Search for information |
| `/api/chat` | POST | Chat with books in the library |
| `/api/summary` | POST | Generate book or chapter summary |
| `/api/connections` | POST | Find connections between topics |
| `/api/book_structure/{book_title}` | GET | Get document structure (TOC) |
| `/api/reading_suggestions` | POST | Get reading suggestions |
| `/api/status` | GET | Get processing status |
| `/api/reset` | POST | Reset system |
| `/health` | GET | Health check endpoint |

## Project Structure

```
digital-library-chatbot/
├── .env                        # Environment variables
├── .env.example                # Example environment file
├── .gitignore                  # Git ignore file
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├                 # Helper script for starting the app
│
├── docker/                     # Docker configuration files
│   ├── .dockerignore           # Docker ignore file
│   ├── docker-compose.yml      # Docker Compose configuration
│   ├── Dockerfile              # Docker build configuration
│   └── Dockerfile.multistage   # Optimized multi-stage Docker build
│   |── start.sh   
|
├── src/                        # Source code
│   ├── load_data.py            # PDF processing and data loading
│   ├── llm_interface.py        # LLM interaction functionality
│   ├── main.py                 # FastAPI application and endpoints
│   ├── response_format.py      # Output formatting utilities
│   ├── retrieval.py            # Vector search and content retrieval
│   ├── retriever_utils.py      # Helper utilities for retrieval
│   ├── standalone.py           # Gradio UI implementation
│   ├── vector_db.py            # Vector database integration
│   └── static/                 # Static files for the web interface
│
├── ebooks_library/             # Storage for PDF files
├── processed_data/             # Processed document data
├── qdrant_db/                  # Vector database storage
└── uploads/                    # Temporary upload storage
```

## Configuration

The chatbot can be configured through environment variables in the `.env` file:

- `GROQ_API_KEY`: Your Groq API key for LLM access
- `LLM_MODEL`: Model to use (default: llama3-8b-8192)
- `PORT`: Port for the API backend (default: 8000)
- `UI_PORT`: Port for the chatbot UI (default: 7860)
- `API_BASE_URL`: Base URL for API access
- `LIBRARY_DIR`: Directory for storing PDF files
- `PROCESSED_DIR`: Directory for storing processed data
- `VECTOR_DB_PATH`: Path for vector database storage
- `UPLOAD_DIR`: Directory for temporary file uploads
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)

## Development

### Extending the Chatbot

The modular architecture makes it easy to extend with new capabilities:

1. Implement new processing logic in appropriate modules
2. Add API endpoints in `src/main.py`
3. Update the UI in `src/standalone.py` if needed

### Building Custom Docker Images

```bash
docker build -t digital-library-chatbot:custom -f docker/Dockerfile .

# Or using the optimized multi-stage build:
docker build -t digital-library-chatbot:optimized -f docker/Dockerfile.multistage .
```

## Troubleshooting

### Common Issues

- **Connection Error**: Ensure all services are running with `docker-compose -f docker/docker-compose.yml ps`
- **PDF Processing Error**: Check if the PDF is password-protected or corrupted
- **Memory Issues**: For large PDFs, increase the container memory limit in docker-compose.yml
- **API Key Issues**: Verify your Groq API key is correctly set in the .env file

### Logs

To view application logs:

```bash
# Docker deployment
docker-compose -f docker/docker-compose.yml logs -f

# Manual deployment
tail -f app.log
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PyMuPDF for PDF processing capabilities
- LangChain for the LLM integration framework
- Qdrant for vector storage
- FastAPI and Gradio for the backend and UI