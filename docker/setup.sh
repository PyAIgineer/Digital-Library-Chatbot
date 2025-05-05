#!/bin/bash

# Print environment information for debugging
echo "Starting application on Railway..."
echo "PORT: $PORT"
echo "RAILWAY_STATIC_URL: $RAILWAY_STATIC_URL"

# Make sure PORT is set
export PORT="${PORT:-8000}"
echo "Using PORT=$PORT"

# Ensure Gradio binds to 0.0.0.0
export GRADIO_SERVER_NAME=0.0.0.0

# Modify the API_BASE_URL if needed for the Gradio app
# For Railway deployment, we need to make sure Gradio knows where to find the API
if [ -n "$RAILWAY_STATIC_URL" ]; then
  # If deployed on Railway, the API is at the same domain
  export API_BASE_URL="${RAILWAY_STATIC_URL}"
  echo "Setting API_BASE_URL=${API_BASE_URL}"
fi

# Execute uvicorn directly with the correct host binding
# The 'exec' command replaces the current process with uvicorn
exec uvicorn main:app --host 0.0.0.0 --port "$PORT"