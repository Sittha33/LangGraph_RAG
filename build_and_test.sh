#!/bin/bash

# Exit on any error
set -e

# Variables
IMAGE_NAME="v3"
CONTAINER_NAME="multi-agent-qna-test"
PORT="5001"

# Step 1: Build the Docker image
echo "Building Docker image: $IMAGE_NAME"
docker build -t $IMAGE_NAME -f Dockerfile .

# Step 2: Test the image by running a container
echo "Running container to test the image..."
docker run --name $CONTAINER_NAME -d -p $PORT:5000 --env-file .env -v $(pwd)/rag_docs:/app/rag_docs $IMAGE_NAME

# Wait for the container to start (give Gunicorn time to initialize)
sleep 5

# Step 3: Test the Flask API
echo "Testing Flask API..."
if curl --fail http://localhost:$PORT; then
    echo "API is running successfully!"
else
    echo "Failed to reach the API. Checking container logs..."
    docker logs $CONTAINER_NAME
    exit 1
fi

# Step 4: Clean up
echo "Stopping and removing test container..."
docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME

echo "Docker image $IMAGE_NAME built and tested successfully!"