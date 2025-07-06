import json
from unittest.mock import MagicMock
from src.main import app_flask
import src.main # Import the module to allow monkeypatching

def test_health_check():
    """Tests the /health endpoint."""
    with app_flask.test_client() as client:
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'ok'

def test_ask_missing_params():
    """Tests the /ask endpoint with missing parameters."""
    with app_flask.test_client() as client:
        response = client.post('/ask', json={})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert "Missing 'question' or 'session_id'" in data['error']

def test_ask_endpoint(monkeypatch):
    """Tests a successful call to the /ask endpoint."""
    # Create a mock graph object that simulates the real one
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        'answer': 'This is a mock answer.',
        'sources': [],
        'confidence_score': 0.95,
        'confidence_explanation': 'Mocked confidence.'
    }
    
    # Replace the real graph_app in the main module with our mock
    monkeypatch.setattr(src.main, 'graph_app', mock_graph)
    
    with app_flask.test_client() as client:
        payload = {"session_id": "test123", "question": "What is RAG?"}
        response = client.post('/ask', json=payload)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['answer'] == 'This is a mock answer.'
        assert data['confidence_score'] == 0.95
        
        # Verify session history was updated
        from src.main import chat_sessions
        assert chat_sessions['test123'][-1] == ("ai", "This is a mock answer.")

def test_clear_memory_endpoint(monkeypatch):
    """Tests the /clear_memory endpoint."""
    # Create and set up the mock graph object
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {'answer': 'test'}
    
    # Replace the real graph_app in the main module
    monkeypatch.setattr(src.main, 'graph_app', mock_graph)
    
    with app_flask.test_client() as client:
        # First, populate a session by calling the /ask endpoint
        payload = {"session_id": "mem_test", "question": "Test question"}
        client.post('/ask', json=payload)

        # Now, clear the memory for that session
        clear_payload = {"session_id": "mem_test"}
        response = client.post('/clear_memory', json=clear_payload)
        assert response.status_code == 200
        
        # Verify the session is gone
        from src.main import chat_sessions
        assert "mem_test" not in chat_sessions

def test_clear_memory_not_found():
    """Tests clearing a non-existent session."""
    with app_flask.test_client() as client:
        clear_payload = {"session_id": "non_existent_session"}
        response = client.post('/clear_memory', json=clear_payload)
        assert response.status_code == 404