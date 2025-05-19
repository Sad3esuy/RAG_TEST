# -*- coding: utf-8 -*-
"""RAG_with_LangChain_API_Upload.py

Adapted from RAG_with_LangChain_API.py to allow PDF uploads
and dynamic RAG chain creation.
"""

import os
import requests # Still needed if you want to keep the fallback download logic (optional)
import shutil # For directory operations
import tempfile # For handling temporary uploaded files
from typing import List, Dict, Optional
import traceback # Import traceback để in lỗi chi tiết hơn

# Tải biến môi trường từ tệp .env
from dotenv import load_dotenv
load_dotenv()
print(".env file loading attempted.")

# Import các thư viện cần thiết
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel # Để định nghĩa cấu trúc dữ liệu yêu cầu từ frontend

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, Runnable
from langchain_huggingface import HuggingFaceEmbeddings

# --- Định nghĩa TẤT CẢ các hàm Helper ở đây --- (Giữ nguyên các hàm helper của bạn)

def pdf_extract(pdf_path: str) -> List:
    """
    Extracts text from a PDF file using PyPDFLoader.
    """
    print(f"Extracting text from PDF: {pdf_path}")
    try:
        loader = PyPDFLoader(pdf_path)
        pdf_text = loader.load()
        print(f"Successfully extracted {len(pdf_text)} pages from {pdf_path}.")
        return pdf_text
    except Exception as e:
        print(f"Error during PDF extraction from {pdf_path}: {e}")
        traceback.print_exc()
        return []

def pdf_chunk(pdf_text: List) -> List:
    """
    Splits extracted PDF text into smaller chunks.
    """
    print("PDF file text is chunked....")
    if not pdf_text:
        print("No text to chunk.")
        return []
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(pdf_text)
        print(f"Successfully created {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        print(f"Error during chunking: {e}")
        traceback.print_exc()
        return []

def create_vector_store(chunks: List, db_path: str) -> Optional[Chroma]:
    global db  # Thêm dòng này để truy cập biến toàn cục db

    print(f"Chrome vector store is being created at {db_path}...\n")
    if not chunks:
        print("No chunks provided to create vector store.")
        return None
    try:
        # Đóng kết nối cũ nếu tồn tại
        if db is not None:
            try:
                db.delete_collection()  # Xóa collection hiện tại
                db._client.close()  # Đóng kết nối ChromaDB
                db = None
                print("Closed previous ChromaDB connection")
            except Exception as e:
                print(f"Warning: Error while closing previous ChromaDB connection: {e}")
        
        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Thêm thời gian chờ để đảm bảo handles được giải phóng
        import time
        time.sleep(1)
        
        # Xóa thư mục cũ nếu tồn tại
        if os.path.exists(db_path):
            print(f"Removing existing vector store at: {db_path}")
            try:
                shutil.rmtree(db_path)
                print(f"Successfully removed directory: {db_path}")
                # Thêm thời gian chờ sau khi xóa
                time.sleep(1)
            except Exception as e:
                print(f"Error removing directory {db_path}: {e}")
                # Nếu không thể xóa, tạo thư mục mới
                new_db_path = f"{db_path}_{int(time.time())}"
                print(f"Will use alternate path: {new_db_path}")
                db_path = new_db_path
        
        os.makedirs(db_path, exist_ok=True)

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        db = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=db_path)
        print(f"Chrome vector store created successfully at {db_path}.\n")
        return db
    except Exception as e:
        print(f"Error creating vector store at {db_path}: {e}")
        traceback.print_exc()
        return None

def retrieve_context(db: Chroma, query: str) -> List:
    """
    Retrieves relevant document chunks from the Chroma vector store.
    """
    print("Relevant chunks are being retrieved...\n")
    if db is None:
        print("Cannot retrieve context: Vector store is not initialized.")
        return []
    try:
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2}) # You can adjust k
        relevant_chunks = retriever.invoke(query)
        print(f"Retrieved {len(relevant_chunks)} relevant chunks.")
        return relevant_chunks
    except Exception as e:
        print(f"Error retrieving context: {e}")
        traceback.print_exc()
        return []

def build_context(relevant_chunks: List) -> str:
    """
    Builds a context string from retrieved relevant document chunks.
    """
    print("Context is built from relevant chunks")
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    return context

# Use the global 'db' which will be updated upon PDF upload
def get_context_for_chain(query: str) -> Dict[str, str]:
    global db # Access the global db variable
    if db is None:
        # This case should ideally be handled before invoking the chain,
        # but as a safeguard:
        print("Error in get_context_for_chain: Vector store (db) is not initialized.")
        return {'context': "Error: No document loaded or processed.", 'query': query}

    relevant_chunks = retrieve_context(db, query)
    context = build_context(relevant_chunks)
    return {'context': context, 'query': query}

# --- FastAPI App Initialization ---
app = FastAPI()

# Optional: Cấu hình CORS nếu frontend chạy ở domain/port khác
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Hoặc chỉ định chính xác domain, e.g., "http://localhost:3000"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],  # Thêm dòng này
)


# --- Pydantic model for request body ---
class QueryRequest(BaseModel):
    query: str

# --- Global variables ---
# These will be initialized either at startup (for LLM, prompt) or after PDF upload (for db, chain)
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_db_base_path = os.path.join(current_dir, "db_uploaded") # Base for ChromaDB for uploaded PDF
os.makedirs(persistent_db_base_path, exist_ok=True)
# Path for the specific Chroma DB instance for the currently active PDF
active_persistent_directory = os.path.join(persistent_db_base_path, "chroma_db_active_pdf")


llm: Optional[ChatDeepSeek] = None
rag_prompt: Optional[ChatPromptTemplate] = None
str_parser: Optional[StrOutputParser] = None
db: Optional[Chroma] = None # Vector store, initialized after PDF upload
rag_chain_runnable: Optional[Runnable] = None # RAG chain, initialized after PDF upload

# --- Application Setup (runs once at app startup for non-PDF specific parts) ---
def configure_llm_and_prompts():
    global llm, rag_prompt, str_parser, DEEPSEEK_API_KEY
    print("Configuring LLM and RAG prompt template...")

    # Configure the LLM
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    if not DEEPSEEK_API_KEY:
        print("CRITICAL ERROR: DEEPSEEK_API_KEY environment variable not set.")
        print("Please make sure you have a .env file in the script directory with DEEPSEEK_API_KEY=YOUR_KEY")
        # Application might not be fully functional
    else:
        try:
            llm = ChatDeepSeek(
                model="deepseek-chat",
                api_key=DEEPSEEK_API_KEY,
                temperature=0,
                # ... other params
            )
            print("LLM configured.")
        except Exception as e:
            print(f"Error configuring LLM: {e}")
            traceback.print_exc()
            llm = None # Ensure llm is None if configuration fails

    # Define the RAG prompt template
    template = """ You are an AI model trained for question answering. You should answer the
    given question based on the given context only.
    Question : {query}
    \n
    Context : {context}
    \n
    If the answer is not present in the given context, respond as: The answer to this question is not available
    in the provided content.
    """
    rag_prompt = ChatPromptTemplate.from_template(template)
    str_parser = StrOutputParser()
    print("RAG prompt template defined.")

# Call configuration at startup
configure_llm_and_prompts()


# --- FastAPI Endpoints ---

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global db, rag_chain_runnable, llm, rag_prompt, str_parser, active_persistent_directory

    if llm is None or rag_prompt is None or str_parser is None:
        raise HTTPException(status_code=503, detail="LLM or RAG prompt not configured. Backend setup incomplete.")

    # Create a temporary directory to store uploaded files if it doesn't exist
    temp_upload_dir = os.path.join(current_dir, "temp_uploads")
    os.makedirs(temp_upload_dir, exist_ok=True)
    
    temp_pdf_path = os.path.join(temp_upload_dir, file.filename)

    try:
        # Đóng kết nối ChromaDB cũ nếu tồn tại
        if db is not None:
            try:
                print("Closing existing ChromaDB connection before processing new PDF...")
                db.delete_collection()  # Thử xóa collection trước
                db._client.reset()      # Reset client
                db._client.close()      # Đóng kết nối
                db = None
                import time
                time.sleep(1)  # Cho phép hệ thống giải phóng tài nguyên
            except Exception as close_err:
                print(f"Warning: Could not close previous ChromaDB connection: {close_err}")
                traceback.print_exc()
                # Tiếp tục xử lý ngay cả khi không thể đóng kết nối cũ
        
        # Save uploaded file temporarily
        with open(temp_pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"PDF '{file.filename}' uploaded and saved to '{temp_pdf_path}'")

        # 1. Extract text from the uploaded PDF
        pdf_text_docs = pdf_extract(temp_pdf_path)
        if not pdf_text_docs:
            raise HTTPException(status_code=400, detail=f"Could not extract text from PDF: {file.filename}")

        # 2. Chunk the extracted text
        chunks = pdf_chunk(pdf_text_docs)
        if not chunks:
            raise HTTPException(status_code=400, detail=f"Could not chunk PDF text: {file.filename}")

        # 3. Tạo thư mục mới cho mỗi lần tải lên để tránh xung đột
        import time
        timestamp = int(time.time())
        new_db_path = os.path.join(persistent_db_base_path, f"chroma_db_{timestamp}")
        print(f"Creating new vector store at: {new_db_path}")
        
        # Đảm bảo thư mục tồn tại nhưng trống
        if os.path.exists(new_db_path):
            shutil.rmtree(new_db_path)
        os.makedirs(new_db_path, exist_ok=True)
        
        # 4. Create and persist vector store với thư mục mới
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        db = Chroma.from_documents(
            documents=chunks, 
            embedding=embedding_model, 
            persist_directory=new_db_path
        )
        
        if db is None:
            raise HTTPException(status_code=500, detail=f"Could not create vector store for PDF: {file.filename}")
        
        # Cập nhật đường dẫn active persistant directory
        active_persistent_directory = new_db_path
        print(f"Vector store successfully created at: {active_persistent_directory}")

        # 5. Rebuild the RAG chain with the new vector store
        print("Building the RAG chain runnable with new PDF context...")
        rag_chain_runnable = (
            RunnableLambda(lambda x: get_context_for_chain(x['query']))
            | rag_prompt
            | llm
            | str_parser
        )
        print("RAG chain runnable updated for the new PDF.")
        
        # Xóa thư mục cũ nếu tồn tại và không phải là thư mục hiện tại
        try:
            old_dirs = [d for d in os.listdir(persistent_db_base_path) 
                     if d.startswith("chroma_db_") and os.path.join(persistent_db_base_path, d) != active_persistent_directory]
            
            for old_dir in old_dirs:
                old_path = os.path.join(persistent_db_base_path, old_dir)
                print(f"Cleaning up old vector store: {old_path}")
                shutil.rmtree(old_path, ignore_errors=True)
        except Exception as cleanup_err:
            print(f"Warning: Error during cleanup of old vector stores: {cleanup_err}")
            # Không dừng xử lý nếu việc dọn dẹp thất bại
        
        return {"message": f"PDF '{file.filename}' processed successfully. You can now ask questions about it."}

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly
        raise http_exc
    except Exception as e:
        print(f"\nAn error occurred during PDF processing: {e}")
        traceback.print_exc()
        # Clean up global state if processing failed to prevent using a partially built chain
        db = None
        rag_chain_runnable = None
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the PDF: {str(e)}")
    finally:
        # Clean up the temporarily saved PDF file
        if os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
                print(f"Temporary file {temp_pdf_path} removed.")
            except Exception as e_remove:
                print(f"Error removing temporary file {temp_pdf_path}: {e_remove}")
        # Close the uploaded file explicitly
        await file.close()


@app.post("/ask")
async def ask_question(request: QueryRequest):
    print(f"\nReceived query: {request.query}")

    if rag_chain_runnable is None or db is None:
        print("Error: RAG chain or vector store not initialized. Please upload a PDF first.")
        raise HTTPException(status_code=400, detail="No PDF processed yet or RAG chain not ready. Please upload a PDF via /upload_pdf endpoint first.")
    
    if llm is None: # Check if LLM was successfully configured at startup
        print("Error: LLM is not configured. Cannot process query.")
        raise HTTPException(status_code=503, detail="Backend LLM service is not configured or unavailable.")

    try:
        answer = rag_chain_runnable.invoke({"query": request.query})
        print(f"\nGenerated answer:\n{answer}")
        return {"answer": answer}
    except Exception as e:
        print(f"\nAn error occurred during RAG chain execution: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the query: {e}")


@app.get("/")
async def read_root():
    return {"message": "RAG backend with PDF upload is running. POST to /upload_pdf to load a document, then POST to /ask to query it."}

# To run this app: uvicorn your_filename:app --reload
# (replace your_filename with the actual name of your Python file)