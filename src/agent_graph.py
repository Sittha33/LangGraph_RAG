from typing import List, TypedDict, Any, Dict
import os
import numpy as np

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

class GraphState(TypedDict):
    question: str
    chat_history: List
    answer: str
    sources: List[Dict[str, Any]]
    next: str
    confidence_score: float
    confidence_explanation: str

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0, convert_system_message_to_human=True)
web_search_tool = TavilySearchResults(k=3)

def format_chat_history(chat_history: List[tuple]) -> str:
    if not chat_history:
        return "No conversation history."
    return "\n".join([f"{role}: {text}" for role, text in chat_history])

def rag_agent(state: GraphState, retriever):
    print("🤖 Calling RAG Agent")
    question = state["question"]
    chat_history = state["chat_history"]

    retrieved_docs_with_scores = retriever.vectorstore.similarity_search_with_score(question, k=4)
    context_docs = [doc for doc, score in retrieved_docs_with_scores]
    
    prompt = PromptTemplate(
        template="""You are an expert assistant for question-answering over academic papers.
Use the following conversation history and retrieved context to answer the question. If you don't know the answer from the context, say that you don't know.

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

    context_str = "\n\n---\n\n".join([f"Source Page: {doc.metadata.get('page', 'N/A')}\n\n{doc.page_content}" for doc in context_docs])
    chain = prompt | llm | StrOutputParser()
    
    formatted_history = format_chat_history(chat_history)
    answer = chain.invoke({"question": question, "context": context_str, "chat_history": formatted_history})
    
    sources = []
    seen_sources = set()
    for doc, score in retrieved_docs_with_scores:
        source_key = (doc.metadata.get('source'), doc.metadata.get('page'))
        if source_key not in seen_sources:
            sources.append({"type": "pdf", "filename": os.path.basename(doc.metadata.get('source', 'Unknown')), "page": doc.metadata.get('page', -1) + 1})
            seen_sources.add(source_key)

    scores = [score for doc, score in retrieved_docs_with_scores]
    avg_score = np.mean(scores) if scores else 0.0
    confidence_score = max(0.0, 1.0 - avg_score)
    explanation = f"Confidence based on the average relevance of {len(scores)} retrieved documents. Average L2 distance: {avg_score:.4f}."
    
    return {"answer": answer, "sources": sources, "confidence_score": float(confidence_score), "confidence_explanation": explanation}

def web_search_agent(state: GraphState):
    print("🤖 Calling Web Search Agent")
    question = state["question"]
    chat_history = state["chat_history"]
    
    search_results = web_search_tool.invoke({"query": question})
    context_str = "\n\n---\n\n".join([res["content"] for res in search_results])
    
    prompt = PromptTemplate(
        template="""You are a helpful research assistant. Use the conversation history and web search results to give a concise answer.

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
    chain = prompt | llm | StrOutputParser()
    formatted_history = format_chat_history(chat_history)
    answer = chain.invoke({"question": question, "context": context_str, "chat_history": formatted_history})
    sources = [{"type": "web", "url": res["url"]} for res in search_results]
    
    return {"answer": answer, "sources": sources, "confidence_score": 1.0, "confidence_explanation": "Answer derived from web search results."}

def clarification_agent(state: GraphState):
    print("🤖 Calling Clarification Agent")
    question = state["question"]
    chat_history = state["chat_history"]

    prompt = PromptTemplate(
        template="""The user's question is ambiguous, even given the conversation history.
Ask a targeted question to help them clarify their intent.

<CONVERSATION_HISTORY>
{chat_history}
</CONVERSATION_HISTORY>

User's ambiguous question: {question}
Your clarifying question:""",
        input_variables=["chat_history", "question"]
    )
    chain = prompt | llm | StrOutputParser()
    formatted_history = format_chat_history(chat_history)
    clarification_q = chain.invoke({"question": question, "chat_history": formatted_history})
    
    return {"answer": clarification_q, "sources": [], "confidence_score": 0.0, "confidence_explanation": "Query was ambiguous. Asking for clarification."}

def router_agent(state: GraphState, retriever):
    print("🧠 Routing question...")
    if not retriever:
        print("↳ No RAG retriever. Defaulting to web search.")
        return {"next": "web_search_agent"}

    prompt = PromptTemplate(
        template="""You are an expert at routing a user question to 'vectorstore', 'web_search', or 'clarification'.
Based on the conversation history and current question, choose the most appropriate tool.

- `vectorstore`: Use for specific, technical questions about generative AI or text-to-SQL found in academic papers.
- `web_search`: Use for general knowledge, recent events, or topics outside the scope of the provided papers.
- `clarification`: Use if the question is ambiguous, using vague terms like 'good' or 'best' without specific metrics.

<CONVERSATION_HISTORY>
{chat_history}
</CONVERSATION_HISTORY>

Current Question: {question}

Give a JSON output with a key 'datasource' and one of the three string values.
""",
        input_variables=["chat_history", "question"]
    )
    
    chain = prompt | llm | JsonOutputParser()
    formatted_history = format_chat_history(state["chat_history"])
    result = chain.invoke({"question": state["question"], "chat_history": formatted_history})
    
    print(f"↳ Routing decision: '{result['datasource']}'")
    
    if result['datasource'] == 'vectorstore':
        return {"next": 'rag_agent'}
    elif result['datasource'] == 'web_search':
        return {"next": 'web_search_agent'}
    else:
        return {"next": 'clarification_agent'}

def create_graph(retriever):
    workflow = StateGraph(GraphState)

    workflow.add_node("router", lambda state: router_agent(state, retriever))
    workflow.add_node("rag_agent", lambda state: rag_agent(state, retriever))
    workflow.add_node("web_search_agent", web_search_agent)
    workflow.add_node("clarification_agent", clarification_agent)

    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        lambda state: state["next"],
        {"rag_agent": "rag_agent", "web_search_agent": "web_search_agent", "clarification_agent": "clarification_agent"}
    )
    workflow.add_edge("rag_agent", END)
    workflow.add_edge("web_search_agent", END)
    workflow.add_edge("clarification_agent", END)

    return workflow.compile()