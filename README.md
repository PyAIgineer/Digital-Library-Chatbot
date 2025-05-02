# Digital Library Chatbot for Students

A sophisticated digital library assistant that helps students in their studies by providing intelligent access to educational content through natural language conversations.

## 🌟 Overview

This project provides a comprehensive digital library chatbot designed specifically to support students in their academic activities. The system uses advanced Natural Language Processing (NLP) techniques to understand educational content from PDF textbooks, allowing students to have natural conversations about their study materials, find connections between topics, generate summaries, and get answers to specific questions - all in a conversational interface.

Key features:
- Study assistance through natural language conversations about textbook content
- Helps students find information across their textbooks without manual searching
- Generates concise summaries of chapters and books to aid in exam preparation
- Identifies connections between topics across different courses and textbooks
- Provides accurate answers with citations to textbook sources
- Supports multiple interfaces (web API and user-friendly interface) for flexibility

## 📋 Requirements

- Python 3.9+
- PyMuPDF (for PDF processing)
- Sentence Transformers (for semantic understanding)
- LangChain (for LLM orchestration)
- Qdrant (vector database for efficient information retrieval)
- Groq API (for fast and affordable LLM access)
- FastAPI/Uvicorn (for web API implementation)
- Gradio (for student-friendly user interface)

## 🚀 Getting Started

### Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file with your API keys:
```
GROQ_API_KEY=your_groq_api_key
```

### Project Structure

- `main.py` - FastAPI server for the web API
- `standalone.py` - Gradio interface for direct interaction
- `load_data.py` - PDF processing and data extraction
- `llm_interface.py` - Interface to LLM models for text generation
- `retrieval.py` - Advanced retrieval system for document search
- `retriever_utils.py` - Utilities for reranking search results
- `vector_db.py` - Vector database setup and management
- `response_format.py` - Output formatting utilities

### Usage

#### Option 1: Web Interface for Students

The simplest way to use the digital library chatbot:
```bash
python standalone.py
```

This starts a user-friendly interface at http://localhost:7860 where students can:
1. Upload their textbooks and study materials
2. Choose which book to study from their digital library
3. Chat naturally with the system about the content
4. Generate quick summaries for exam preparation
5. Ask questions and get answers with proper citations

#### Option 2: API Server for Institutional Integration

For educational institutions wanting to integrate with existing systems:
```bash
python main.py
```

This starts an API server on http://localhost:7860 that can:
1. Be integrated with learning management systems
2. Connect to existing digital library infrastructure
3. Support custom institutional applications
4. Enable programmatic uploads of department-approved textbooks
5. Allow tracking of usage analytics for educational insights

## 📘 Technical Architecture

### PDF Processing (`load_data.py`)
Behind the scenes, this component:
- Intelligently extracts text and structure from your textbooks
- Identifies tables, diagrams, and educational content blocks
- Maintains chapter and section organization for proper context
- Breaks content into meaningful chunks while preserving semantic relationships

### Smart Retrieval (`retrieval.py`)
The brain of the study assistant that:
- Understands what information you're looking for, even if you don't use exact keywords
- Finds the most relevant passages across all your uploaded textbooks
- Ensures diverse information when needed for comprehensive understanding
- Discovers connections between topics that might not be obvious

### Conversation Engine (`llm_interface.py`)
Enables the chatbot to:
- Format educational content for optimal understanding
- Identify and highlight connections between concepts
- Ensure answers include proper citations to your textbooks
- Rerank search results to prioritize the most helpful information

### Knowledge Storage (`vector_db.py`)
Efficiently stores and retrieves information:
- Creates semantic representations of your textbooks
- Enables lightning-fast retrieval of relevant content
- Optimizes memory usage for large educational libraries
- Ensures accurate retrieval of information

## 📝 Example Use Cases

### Study Assistance
Ask questions like "What are the key principles of thermodynamics?" and get comprehensive answers with textbook citations.

### Exam Preparation
Request summaries of chapters or entire textbooks to quickly review material before tests.

### Connecting Concepts
Ask "How does photosynthesis relate to cellular respiration?" to find connections across biology topics or even between different courses.

### Research Help
Upload multiple textbooks and research papers, then have conversations to gather information for assignments.

### Clarification of Difficult Concepts
Ask for explanations of complex topics in simpler terms to enhance understanding.

### Study Planning
Get suggestions on what to read next based on your current focus and curriculum requirements.

## 🔧 Customization for Educational Institutions

### Course-Specific Configurations

Educational institutions can customize the chatbot for specific courses:
1. Upload course-specific textbooks and materials
2. Modify prompts in `llm_interface.py` to align with course learning objectives
3. Add course-specific terminology and concepts

### Integration with Learning Management Systems

The system can be integrated with existing LMS platforms:
1. Use the FastAPI endpoints to connect to systems like Canvas, Moodle, or Blackboard
2. Enable single sign-on for student authentication
3. Track student interactions for personalized learning analytics

### Supporting Different Document Types

Beyond textbooks, the system can be customized to work with:
1. Lecture notes and slides
2. Academic papers
3. Lab manuals and procedural documents
4. Student-generated study materials

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.