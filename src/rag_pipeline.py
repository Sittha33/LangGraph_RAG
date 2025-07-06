import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

def initialize_retriever(docs_path: str = "rag_docs"):
    """
    Initializes the RAG retriever by loading PDFs, splitting them into chunks,
    creating embeddings in batches, and setting up a FAISS vector store.
    """
    print("🔄 Initializing RAG pipeline...")
    vectorstore = None

    if not os.path.exists(docs_path):
        print(f"🟡 Warning: Document directory '{docs_path}' not found. RAG will be disabled.")
        return None

    pdf_files = [os.path.join(docs_path, f) for f in os.listdir(docs_path) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"🟡 Warning: No PDF files found in '{docs_path}'. RAG will be disabled.")
        return None

    print(f"📄 Found {len(pdf_files)} PDF(s). Processing for RAG...")
    try:
        loaders = [PyPDFLoader(file_path) for file_path in pdf_files]
        docs = [doc for loader in loaders for doc in loader.load()]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Process embeddings in batches of 50 to stay under the 100-request limit
        batch_size = 50
        vectorstore = None
        for i in range(0, len(all_splits), batch_size):
            batch = all_splits[i:i + batch_size]
            if vectorstore is None:
                vectorstore = FAISS.from_documents(documents=batch, embedding=embeddings)
            else:
                vectorstore.add_documents(batch)

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        print("✅ RAG pipeline ready with local documents.")
        return retriever
    except Exception as e:
        print(f"❌ Error initializing RAG pipeline: {e}")
        return None