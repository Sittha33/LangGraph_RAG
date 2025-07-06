import traceback
from flask import Flask, request, jsonify
from src import config
from src.rag_pipeline import initialize_retriever
from src.agent_graph import create_graph

print("🚀 Initializing application...")
retriever = initialize_retriever()
if retriever is None:
    print("⚠️ Warning: RAG retriever not initialized. Proceeding without RAG.")
graph_app = create_graph(retriever)
print("✅ Application initialized successfully.")

app_flask = Flask(__name__)
chat_sessions = {}

@app_flask.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        if not data or "question" not in data or "session_id" not in data:
            return jsonify({"error": "Missing 'question' or 'session_id' in request"}), 400

        session_id = data["session_id"]
        question = data["question"]

        if session_id not in chat_sessions:
            chat_sessions[session_id] = []

        initial_state = {
            "question": question,
            "chat_history": chat_sessions[session_id],
        }
        
        result = graph_app.invoke(initial_state)

        chat_sessions[session_id].append(("human", question))
        chat_sessions[session_id].append(("ai", result["answer"]))

        return jsonify({
            "answer": result.get('answer'),
            "sources": result.get('sources'),
            "confidence_score": result.get('confidence_score'),
            "confidence_explanation": result.get('confidence_explanation')
        })
    except Exception as e:
        print(f"❌ An error occurred in /ask: {e}")
        print(traceback.format_exc())
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500

@app_flask.route("/clear_memory", methods=["POST"])
def clear_memory():
    data = request.get_json()
    if not data or "session_id" not in data:
        return jsonify({"error": "Missing 'session_id' in request"}), 400

    session_id = data["session_id"]
    if session_id in chat_sessions:
        del chat_sessions[session_id]
        return jsonify({"status": "success", "message": f"Memory for session {session_id} cleared."})
    else:
        return jsonify({"status": "not_found", "message": f"Session {session_id} not found."}), 404

@app_flask.route("/health", methods=["GET"])
def health_check():
    rag_status = "enabled" if retriever else "disabled"
    return jsonify({"status": "ok", "rag_status": rag_status}), 200

if __name__ == "__main__":
    print("🚀 Starting Flask development server on port 5000...")
    app_flask.run(host='0.0.0.0', port=5000)