LangGraph_RAG: Multi-Agent Question Answering System
Overview
LangGraph_RAG is a sophisticated, containerized question-answering system built with a multi-agent architecture. It intelligently routes user queries to the most appropriate agent—whether it's searching a local knowledge base (RAG), browsing the web, or asking for clarification based on conversational context. This project aims to provide accurate, context-aware responses by dynamically selecting the best tool for the job.

Features
Multi-Agent Architecture: Utilizes LangGraph to orchestrate different AI agents (RAG, Web Search, Clarification).

Retrieval-Augmented Generation (RAG): Integrates a local knowledge base (PDF documents) for in-depth, domain-specific answers.

Web Search Integration: Leverages the Tavily API for general knowledge, recent events, or topics outside the local knowledge base.

Conversational Memory: Maintains chat history within sessions, allowing for natural follow-up questions and personalized context.

Intelligent Routing: A router agent dynamically decides the best course of action for each query based on its content and conversational history.

Dockerized: Easy setup and deployment using Docker.

Technologies Used
Python 3.10

Flask: Web framework for the API.

Gunicorn: WSGI HTTP Server for production deployment.

LangChain & LangGraph: For building and orchestrating the AI agents.

Google Gemini Pro: Large Language Model for agent intelligence.

FAISS: For efficient similarity search in the RAG pipeline.

Tavily API: For web search capabilities.

Docker: For containerization.

Setup Instructions
Follow these steps to get the LangGraph_RAG application up and running on your local machine.

Prerequisites
Git: For cloning the repository.

Docker Desktop: Ensure Docker is installed and running on your system.

API Keys:

Google Gemini API Key: Obtain one from Google AI Studio.

Tavily API Key: Obtain one from Tavily AI.

1. Clone the Repository
First, clone the project repository to your local machine:

git clone https://github.com/Sittha33/LangGraph_RAG.git
cd LangGraph_RAG

2. Place RAG Documents (Optional but Recommended)
For the RAG functionality to work, place your PDF documents in the rag_docs directory within the cloned repository. If this directory doesn't exist, create it:

mkdir rag_docs

Example path on Windows: C:\Users\YourUser\LangGraph_RAG\rag_docs

3. Build the Docker Image
Navigate to the root directory of the LangGraph_RAG project (where Dockerfile is located) and build the Docker image:

docker build -t langgraph-rag:latest .

This process might take a few minutes as Docker downloads base images and installs dependencies.

4. Run the Docker Container
Run the Docker container, exposing port 5001 on your host machine to port 5000 inside the container. Crucially, you must pass your API keys as environment variables and mount the rag_docs volume.

Important: Replace "YOUR_GEMINI_API_KEY" and "YOUR_TAVILY_API_KEY" with your actual API keys. Adjust the volume path C:/Users/YourUser/LangGraph_RAG/rag_docs to the absolute path where your rag_docs folder is located on your machine.

docker run -d -p 5001:5000 --name langgraph-rag-app \
  -e GOOGLE_API_KEY="YOUR_GEMINI_API_KEY" \
  -e TAVILY_API_KEY="YOUR_TAVILY_API_KEY" \
  -v C:/Users/YourUser/LangGraph_RAG/rag_docs:/app/rag_docs \
  langgraph-rag:latest

-d: Runs the container in detached mode (in the background).

-p 5001:5000: Maps host port 5001 to container port 5000. The application will be accessible via http://localhost:5001.

--name langgraph-rag-app: Assigns a name to your container for easy management.

-e: Passes environment variables (your API keys).

-v: Mounts your local rag_docs directory into the container, allowing the RAG pipeline to access your PDF documents.

API Usage
Once the Docker container is running, you can interact with the application via its API endpoints.

The base URL for the API will be http://localhost:5001.

1. Ask a Question
Send a POST request to the /ask endpoint with your question and a session_id. The session_id is used to maintain conversation history.

Endpoint: POST /ask
Headers: Content-Type: application/json
Body:

{
  "question": "Your question here.",
  "session_id": "unique_session_id_for_this_conversation"
}

Example curl command:

curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the primary goal of the LangGraph project?", "session_id": "my_first_conversation"}' \
     http://localhost:5001/ask

Example follow-up question in the same session:

curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"question": "How does it achieve this?", "session_id": "my_first_conversation"}' \
     http://localhost:5001/ask

2. Clear Session Memory
To clear the chat history for a specific session, send a POST request to the /clear_memory endpoint:

Endpoint: POST /clear_memory
Headers: Content-Type: application/json
Body:

{
  "session_id": "session_id_to_clear"
}

Example curl command:

curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"session_id": "my_first_conversation"}' \
     http://localhost:5001/clear_memory

3. Health Check
To check if the application is running and if the RAG retriever is enabled:

Endpoint: GET /health

Example curl command:

curl http://localhost:5001/health

Testing Scenarios
You can test the application's capabilities with various types of questions:

RAG-specific questions: Ask about content directly from the PDF documents you placed in rag_docs (e.g., "What are the natural language challenges in Text-to-SQL according to Katsogiannis-Meimarakis and Koutrika (2023)?").

Web-augmented questions: Ask general knowledge questions or questions about very recent events not covered in your local documents (e.g., "What is the capital of France?").

Conversational questions: Engage in a multi-turn conversation, asking follow-up questions that rely on previous context (e.g., "My name is [Your Name]. What is my name?").

Troubleshooting
docker run errors: Double-check your API keys and ensure the volume mount path for rag_docs is correct and absolute.

API not responding: Verify the Docker container is running (docker ps) and check its logs (docker logs langgraph-rag-app).

RAG not working: Ensure your rag_docs folder contains valid PDF files and that the volume mount is correctly configured. Check the container logs for RAG initialization warnings.
