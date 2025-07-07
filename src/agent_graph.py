from typing import List, TypedDict, Any, Dict
import os
import numpy as np

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# Define the state for the graph, including chat history, answer, and sources.
class GraphState(TypedDict):
    question: str
    chat_history: List
    answer: str
    sources: List[Dict[str, Any]]
    next: str
    confidence_score: float
    confidence_explanation: str

# Initialize the Large Language Model (LLM)
# Using gemini-1.5-pro-latest for its advanced capabilities.
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0, convert_system_message_to_human=True)

# Initialize the web search tool for external information retrieval.
web_search_tool = TavilySearchResults(k=3)

# Helper function to format chat history into a readable string for the prompt.
def format_chat_history(chat_history: List[tuple]) -> str:
    if not chat_history:
        return "No conversation history."
    # Format each turn as "role: text"
    return "\n".join([f"{role}: {text}" for role, text in chat_history])

# Agent responsible for RAG (Retrieval Augmented Generation) based on local documents.
def rag_agent(state: GraphState, retriever):
    print("🤖 Calling RAG Agent")
    question = state["question"]
    chat_history = state["chat_history"]

    # Retrieve relevant documents from the vector store based on the question.
    retrieved_docs_with_scores = retriever.vectorstore.similarity_search_with_score(question, k=4)
    context_docs = [doc for doc, score in retrieved_docs_with_scores]
    
    # Define the prompt for the RAG agent.
    # Emphasize using conversation history for direct answers, especially for follow-ups.
    prompt = PromptTemplate(
        template="""You are an expert assistant for question-answering over academic papers.
Considering the entire conversation history and the provided context, directly answer the user's current question.
If the current question is a follow-up, ensure your answer builds upon the previous turns.
If you don't know the answer from the context, state that you don't know.

<CONVERSATION_HISTORY>
{chat_history}
</CONVERSATION_HISTORY>

<RETRIEVED_CONTEXT>
{context}
</RETRIEVED_CONTEXT>

Question: {question}
Answer:""",
        input_variables=["chat_history", "question", "context"]
    )

    # Format the retrieved documents into a single string for the prompt.
    context_str = "\n\n---\n\n".join([f"Source Page: {doc.metadata.get('page', 'N/A')}\n\n{doc.page_content}" for doc in context_docs])
    
    # Create a chain for the LLM call.
    chain = prompt | llm | StrOutputParser()
    
    # Format the chat history for the prompt.
    formatted_history = format_chat_history(chat_history)
    
    # Invoke the LLM to get the answer.
    answer = chain.invoke({"question": question, "context": context_str, "chat_history": formatted_history})
    
    # Process sources to avoid duplicates and extract relevant information.
    sources = []
    seen_sources = set()
    for doc, score in retrieved_docs_with_scores:
        source_key = (doc.metadata.get('source'), doc.metadata.get('page'))
        if source_key not in seen_sources:
            sources.append({"type": "pdf", "filename": os.path.basename(doc.metadata.get('source', 'Unknown')), "page": doc.metadata.get('page', -1) + 1})
            seen_sources.add(source_key)

    # Calculate confidence score based on retrieval scores.
    scores = [score for doc, score in retrieved_docs_with_scores]
    avg_score = np.mean(scores) if scores else 0.0
    confidence_score = max(0.0, 1.0 - avg_score) # Simple inverse of average L2 distance
    explanation = f"Confidence based on the average relevance of {len(scores)} retrieved documents. Average L2 distance: {avg_score:.4f}."
    
    return {"answer": answer, "sources": sources, "confidence_score": float(confidence_score), "confidence_explanation": explanation}

# Agent responsible for performing web searches.
def web_search_agent(state: GraphState):
    print("🤖 Calling Web Search Agent")
    question = state["question"]
    chat_history = state["chat_history"]
    
    # Perform a web search using the TavilySearchResults tool.
    search_results = web_search_tool.invoke({"query": question})
    
    # Format web search results into a single string for the prompt.
    context_str = "\n\n---\n\n".join([res["content"] for res in search_results])
    
    # Define the prompt for the web search agent.
    # Emphasize using conversation history for direct answers, especially for follow-ups.
    prompt = PromptTemplate(
        template="""You are a helpful research assistant.
Considering the entire conversation history and the web search results, directly answer the user's current question.
If the current question is a follow-up, ensure your answer builds upon the previous turns.

<CONVERSATION_HISTORY>
{chat_history}
</CONVERSATION_HISTORY>

<WEB_SEARCH_RESULTS>
{context}
</WEB_SEARCH_RESULTS>

Question: {question}
Answer:""",
        input_variables=["chat_history", "question", "context"]
    )
    
    # Create a chain for the LLM call.
    chain = prompt | llm | StrOutputParser()
    
    # Format the chat history for the prompt.
    formatted_history = format_chat_history(chat_history)
    
    # Invoke the LLM to get the answer.
    answer = chain.invoke({"question": question, "context": context_str, "chat_history": formatted_history})
    
    # Extract URLs as sources.
    sources = [{"type": "web", "url": res["url"]} for res in search_results]
    
    return {"answer": answer, "sources": sources, "confidence_score": 1.0, "confidence_explanation": "Answer derived from web search results."}

# Agent responsible for asking clarifying questions when the user's query is ambiguous.
def clarification_agent(state: GraphState):
    print("🤖 Calling Clarification Agent")
    question = state["question"]
    chat_history = state["chat_history"]

    # Define the prompt for the clarification agent.
    # Added instruction to check chat history first for personal context.
    prompt = PromptTemplate(
        template="""The user's question is ambiguous, even given the conversation history.
Before asking for clarification, check if the question can be answered from the conversation history itself (e.g., remembering a name or preference). If so, answer directly.
Otherwise, ask a targeted question to help them clarify their intent.

<CONVERSATION_HISTORY>
{chat_history}
</CONVERSATION_HISTORY>

User's ambiguous question: {question}
Your clarifying question or direct answer:""", # Changed to allow direct answer
        input_variables=["chat_history", "question"]
    )
    
    # Create a chain for the LLM call.
    chain = prompt | llm | StrOutputParser()
    
    # Format the chat history for the prompt.
    formatted_history = format_chat_history(chat_history)
    
    # Invoke the LLM to get the clarifying question or direct answer.
    clarification_or_answer = chain.invoke({"question": question, "chat_history": formatted_history})
    
    # If the model gives a direct answer, its confidence should be higher.
    # This is a heuristic; a more robust solution might involve a separate "self-correction" step.
    if "clarifying question" not in clarification_or_answer.lower():
        return {"answer": clarification_or_answer, "sources": [], "confidence_score": 0.9, "confidence_explanation": "Answer derived directly from chat history."}
    else:
        return {"answer": clarification_or_answer, "sources": [], "confidence_score": 0.0, "confidence_explanation": "Query was ambiguous. Asking for clarification."}


# Agent responsible for routing the user's question to the appropriate agent.
def router_agent(state: GraphState, retriever):
    print("🧠 Routing question...")
    # If RAG retriever is not initialized (e.g., no PDF docs), default to web search.
    if not retriever:
        print("↳ No RAG retriever. Defaulting to web search.")
        return {"next": "web_search_agent"}

    # Define the prompt for the router agent.
    # It decides whether to use vectorstore (RAG), web search, or ask for clarification.
    # Added a specific instruction for personal context/chat history questions.
    prompt = PromptTemplate(
        template="""You are an expert at routing a user question to 'vectorstore', 'web_search', 'chat_history_only', or 'clarification'.
Based on the conversation history and current question, choose the most appropriate tool.

- `vectorstore`: Use for specific, technical questions about generative AI or text-to-SQL found in academic papers.
- `web_search`: Use for general knowledge, recent events, or topics outside the scope of the provided papers.
- `chat_history_only`: Use if the question can be answered directly and entirely from the conversation history, without needing external tools (e.g., "What is my name?").
- `clarification`: Use if the question is ambiguous, using vague terms like 'good' or 'best' without specific metrics, AND cannot be answered from chat history.

<CONVERSATION_HISTORY>
{chat_history}
</CONVERSATION_HISTORY>

Current Question: {question}

Give a JSON output with a key 'datasource' and one of the four string values.
""",
        input_variables=["chat_history", "question"]
    )
    
    # Create a chain for the LLM call.
    chain = prompt | llm | JsonOutputParser()
    
    # Format the chat history for the prompt.
    formatted_history = format_chat_history(state["chat_history"])
    
    # Invoke the LLM to get the routing decision.
    result = chain.invoke({"question": state["question"], "chat_history": formatted_history})
    
    print(f"↳ Routing decision: '{result['datasource']}'")
    
    # Return the next agent based on the routing decision.
    if result['datasource'] == 'vectorstore':
        return {"next": 'rag_agent'}
    elif result['datasource'] == 'web_search':
        return {"next": 'web_search_agent'}
    elif result['datasource'] == 'chat_history_only':
        # If it's a chat_history_only question, we can route it directly to a simpler agent
        # or even have the router itself provide a direct answer if simple enough.
        # For now, we'll route it to clarification_agent which is now enhanced to answer directly.
        return {"next": 'clarification_agent'} # Re-using clarification_agent for direct answers
    else:
        return {"next": 'clarification_agent'}

# Function to create and compile the LangGraph workflow.
def create_graph(retriever):
    # Initialize the StateGraph with the defined GraphState.
    workflow = StateGraph(GraphState)

    # Add nodes to the workflow, each representing an agent or router.
    workflow.add_node("router", lambda state: router_agent(state, retriever))
    workflow.add_node("rag_agent", lambda state: rag_agent(state, retriever))
    workflow.add_node("web_search_agent", web_search_agent)
    workflow.add_node("clarification_agent", clarification_agent)

    # Set the entry point of the graph to the router.
    workflow.set_entry_point("router")
    
    # Add conditional edges from the router based on its decision.
    workflow.add_conditional_edges(
        "router",
        lambda state: state["next"],
        {"rag_agent": "rag_agent", "web_search_agent": "web_search_agent", "clarification_agent": "clarification_agent"}
    )
    
    # Define the end points for each agent.
    workflow.add_edge("rag_agent", END)
    workflow.add_edge("web_search_agent", END)
    workflow.add_edge("clarification_agent", END)

    # Compile the workflow into an executable graph.
    return workflow.compile()
