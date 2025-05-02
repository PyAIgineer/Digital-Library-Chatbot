#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Educational Content Analysis RAG System Setup${NC}"
echo "---------------------------------------------"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p ebooks_library processed_data qdrant_db models uploads

# Check for .env file and create if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    cp .env.example .env
    echo -e "${RED}Please edit the .env file and add your API keys.${NC}"
    exit 1
fi

# Build and start Docker container
echo -e "${YELLOW}Building and starting Docker container...${NC}"
docker-compose up -d --build

# Check if container is running
if [ "$(docker ps -q -f name=educational-rag)" ]; then
    echo -e "${GREEN}Container is running!${NC}"
    echo -e "API is available at: ${GREEN}http://localhost:8000${NC}"
    echo -e "Run ${YELLOW}docker exec -it educational-rag python standalone.py${NC} to start the Gradio UI"
    
    # Show API health status
    echo -e "${YELLOW}Checking API health...${NC}"
    sleep 5 # Give the API a moment to start
    if curl -s http://localhost:8000/health | grep -q "healthy"; then
        echo -e "${GREEN}API is healthy!${NC}"
    else
        echo -e "${RED}API health check failed. Check docker logs for errors.${NC}"
    fi
else
    echo -e "${RED}Container failed to start. Check docker logs for errors.${NC}"
    echo -e "Run: ${YELLOW}docker-compose logs${NC}"
fi