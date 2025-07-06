import pytest
from unittest.mock import MagicMock
from src.main import app_flask
import src.main  # Import the module to allow monkeypatching

# Fixture to provide a clean test client for each test function
@pytest.fixture
def client():
    with app_flask.test_client() as client:
        yield client

# Fixture to clear chat sessions before each test to ensure isolation
@pytest.fixture(autouse=True)
def clear_sessions():
    src.main.chat_sessions.clear()

def test_health_check_endpoint(client):
    """Tests that the /health endpoint is working."""
    response = client.get('/health')
    assert response.status_code == 200
    assert response.get_json()['status'] == 'ok'

def test_rag_agent_scenario(monkeypatch, client):
    """Tests the API response for a question intended for the RAG agent."""
    # Mock the entire graph application to simulate a RAG agent response
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        'answer': "According to the document, the test-suite accuracy is 55.1%.",
        'sources': [{'type': 'pdf', 'filename': 'Rajkumar et al. - 2022.pdf', 'page': 2}],
        'confidence_score': 0.95,
        'confidence_explanation': 'High confidence from RAG source.'
    }
    monkeypatch.setattr(src.main, 'graph_app', mock_graph)

    # Send a PDF-grounded question to the API
    payload = {
        "session_id": "rag_test_1",
        "question": "What is the test-suite accuracy of davinci-codex on Spider with the Create Table + Select 3 prompt?"
    }
    response = client.post('/ask', json=payload)
    data = response.get_json()

    assert response.status_code == 200
    assert "55.1%" in data['answer']
    assert data['sources'][0]['type'] == 'pdf'

def test_web_search_agent_scenario(monkeypatch, client):
    """Tests the API response for a question intended for the Web Search agent."""
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        'answer': "The most recent FIFA World Cup was won by Argentina.",
        'sources': [{'type': 'web', 'url': 'https://www.fifa.com'}],
        'confidence_score': 1.0,
        'confidence_explanation': 'Answer from web search.'
    }
    monkeypatch.setattr(src.main, 'graph_app', mock_graph)

    # Send a general knowledge question to the API
    payload = {
        "session_id": "web_test_1",
        "question": "Who won the most recent FIFA World Cup?"
    }
    response = client.post('/ask', json=payload)
    data = response.get_json()

    assert response.status_code == 200
    assert "Argentina" in data['answer']
    assert data['sources'][0]['type'] == 'web'

def test_clarification_agent_scenario(monkeypatch, client):
    """Tests the API response for a question intended for the Clarification agent."""
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        'answer': "Could you please specify which models you are comparing?",
        'sources': [],
        'confidence_score': 0.0,
        'confidence_explanation': 'Query is ambiguous.'
    }
    monkeypatch.setattr(src.main, 'graph_app', mock_graph)

    # Send an ambiguous question to the API
    payload = {
        "session_id": "clarify_test_1",
        "question": "Which model is better?"
    }
    response = client.post('/ask', json=payload)
    data = response.get_json()

    assert response.status_code == 200
    assert "Could you please specify" in data['answer']

def test_clear_memory_endpoint(monkeypatch, client):
    """Tests that the /clear_memory endpoint successfully removes a session."""
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {'answer': 'LangChain is a framework for developing applications powered by language models.', 'sources': []}
    monkeypatch.setattr(src.main, 'graph_app', mock_graph)
    
    session_id = "mem_clear_test"

    # Step 1: Create a session
    client.post('/ask', json={"session_id": session_id, "question": "What is LangChain?"})
    assert session_id in src.main.chat_sessions

    # Step 2: Clear the session
    response = client.post('/clear_memory', json={"session_id": session_id})
    assert response.status_code == 200
    assert "cleared" in response.get_json()['message']
    
    # Verify the session is gone
    assert session_id not in src.main.chat_sessions
