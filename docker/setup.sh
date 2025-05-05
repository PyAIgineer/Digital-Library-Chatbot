#!/bin/bash

# Create necessary directories if they don't exist
mkdir -p ebooks_library processed_data qdrant_db uploads models static

# Check if .env file exists, if not create from example
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo "Creating .env file from .env.example"
        cp .env.example .env
        echo "Please edit the .env file and add your API keys"
    else
        echo "WARNING: No .env or .env.example file found. Creating a basic .env file."
        cat > .env << EOL
# API Keys
GROQ_API_KEY=your_groq_api_key_here

# LLM Model settings
LLM_MODEL=llama3-8b-8192

# Application settings
PORT=8000
UI_PORT=7860
API_BASE_URL=http://localhost:8000
LIBRARY_DIR=./ebooks_library
PROCESSED_DIR=./processed_data
VECTOR_DB_PATH=./qdrant_db
UPLOAD_DIR=./uploads
EOL
        echo "Please edit the .env file and add your API keys"
    fi
fi

# Build and start Docker containers
echo "Starting application with Docker Compose..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 5

# Check if the API is running
echo "Checking if API is running..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API is running! Access the UI at: http://localhost:7860"
else
    echo "⚠️  API may not be running yet. Check the logs with: docker-compose logs -f"
    echo "Once ready, access the UI at: http://localhost:7860"
fi

echo "To view logs: docker-compose logs -f"
echo "To stop the application: docker-compose down"