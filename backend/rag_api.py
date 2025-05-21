# -*- coding: utf-8 -*-
"""RAG_with_LangChain_API_Upload.py

Adapted from RAG_with_LangChain_API.py to allow PDF uploads
and dynamic RAG chain creation.
"""

import os
import requests # Still needed if you want to keep the fallback download logic (optional)
import shutil # For directory operations
import tempfile # For handling temporary uploaded files
from typing import List, Dict, Optional, Any
import traceback # Import traceback để in lỗi chi tiết hơn
from datetime import datetime, timedelta
import json

# Tải biến môi trường từ tệp .env
from dotenv import load_dotenv
load_dotenv()
print(".env file loading attempted.")

# Import các thư viện cần thiết
from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel # Để định nghĩa cấu trúc dữ liệu yêu cầu từ frontend

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, Runnable
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import PyPDF2
from langchain_core.documents import Document

# Add these imports at the top of the file
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session, relationship
from werkzeug.security import generate_password_hash, check_password_hash

# Create base class for declarative models
Base = declarative_base()

# Create database engine and session
engine = create_engine('sqlite:///app.db')
Session = scoped_session(sessionmaker(bind=engine))

# Database Models
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    role_id = Column(Integer, nullable=False, default=2)  # 1=admin, 2=user
    
    profile = relationship("Profile", back_populates="user", uselist=False)

class Profile(Base):
    __tablename__ = 'profiles'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    full_name = Column(String(120), nullable=False)
    
    user = relationship("User", back_populates="profile")

# Authentication Service
class AuthService:
    @staticmethod
    def hash_password(password):
        return generate_password_hash(password)
    
    @staticmethod
    def verify_password(password_hash, password):
        return check_password_hash(password_hash, password)

# Initialize database
Base.metadata.create_all(engine)

# Create admin user if not exists
session = Session()
try:
    admin = session.query(User).filter_by(email='admin@example.com').first()
    if not admin:
        admin = User(
            email='admin@example.com',
            password_hash=AuthService.hash_password('Admin@123'),
            role_id=1
        )
        session.add(admin)
        session.commit()
        
        # Create admin profile
        profile = Profile(
            user_id=admin.id,
            full_name='Admin User'
        )
        session.add(profile)
        session.commit()
        print("Admin user created successfully")
finally:
    session.close()

# --- Định nghĩa TẤT CẢ các hàm Helper ở đây --- (Giữ nguyên các hàm helper của bạn)

def pdf_extract(pdf_path: str) -> List:
    """
    Extracts text from a PDF file using PyPDF2.
    """
    print(f"Extracting text from PDF: {pdf_path}")
    try:
        # Check if file exists
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found at {pdf_path}")
            return []

        # Check file size
        file_size = os.path.getsize(pdf_path)
        print(f"PDF file size: {file_size} bytes")
        if file_size == 0:
            print("Error: PDF file is empty")
            return []

        # Extract text using PyPDF2
        documents = []
        
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            print(f"Found {num_pages} pages in PDF")
            
            for i, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        doc = Document(
                            page_content=text,
                            metadata={"source": pdf_path, "page": i + 1}
                        )
                        documents.append(doc)
                        print(f"Extracted text from page {i + 1}")
                    else:
                        print(f"No text found on page {i + 1}")
                except Exception as page_err:
                    print(f"Error extracting text from page {i + 1}: {str(page_err)}")
                    continue
            
            if documents:
                print(f"Successfully extracted text from {len(documents)} pages")
                return documents
            else:
                print("No text could be extracted from any page")
                return []

    except Exception as e:
        print(f"Error during PDF extraction from {pdf_path}: {str(e)}")
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

# Configure CORS
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add error handling middleware
@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )

# --- Pydantic model for request body ---
class QueryRequest(BaseModel):
    query: str

# --- Message and Conversation models ---
class Message(BaseModel):
    id: str
    role: str
    content: str
    timestamp: str

class Conversation(BaseModel):
    id: str
    title: str
    messages: List[Message]
    timestamp: str

# --- Global variables ---
# These will be initialized either at startup (for LLM, prompt) or after PDF upload (for db, chain)
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_db_base_path = os.path.join(current_dir, "db_uploaded") # Base for ChromaDB for uploaded PDF
os.makedirs(persistent_db_base_path, exist_ok=True)
# Path for the specific Chroma DB instance for the currently active PDF
active_persistent_directory = os.path.join(persistent_db_base_path, "chroma_db_active_pdf")


llm: Optional[HuggingFacePipeline] = None
rag_prompt: Optional[ChatPromptTemplate] = None
str_parser: Optional[StrOutputParser] = None
db: Optional[Chroma] = None # Vector store, initialized after PDF upload
rag_chain_runnable: Optional[Runnable] = None # RAG chain, initialized after PDF upload

# Add these global variables with other globals
conversations: Dict[str, Conversation] = {}

# --- Application Setup (runs once at app startup for non-PDF specific parts) ---
def configure_llm_and_prompts():
    global llm, rag_prompt, str_parser
    print("Configuring LLM and RAG prompt template...")

    try:
        # Use a simpler model for testing
        model_name = "distilgpt2"  # Smaller model that's easier to load
        print(f"Loading model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        print("Creating pipeline...")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=100,  # Shorter responses for testing
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        print("Creating LangChain LLM...")
        llm = HuggingFacePipeline(pipeline=pipe)
        print("LLM configured successfully")
    except Exception as e:
        print(f"Error configuring LLM: {e}")
        traceback.print_exc()
        llm = None

    # Define the RAG prompt template
    template = """Answer the question based on the context provided.
    Question: {query}
    Context: {context}
    
    If the answer is not in the context, say "I cannot find the answer in the provided content."
    """
    rag_prompt = ChatPromptTemplate.from_template(template)
    str_parser = StrOutputParser()
    print("RAG prompt template defined")

# Call configuration at startup
print("Starting application...")
configure_llm_and_prompts()
print("Application started")

# --- Pydantic models for authentication ---
class UserBase(BaseModel):
    email: str
    full_name: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    role_id: int

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: str | None = None

class LoginRequest(BaseModel):
    email: str
    password: str

# --- Authentication endpoints ---
@app.post("/api/auth/login", response_model=Token)
async def login(request: LoginRequest):
    print(f"\nLogin attempt for email: {request.email}")
    try:
        # Create a new session
        session = Session()
        try:
            # Query the database for the user
            user = session.query(User).filter_by(email=request.email).first()
            
            if user and AuthService.verify_password(user.password_hash, request.password):
                # Generate JWT token
                from datetime import datetime, timedelta
                import jwt
                
                # Create token payload
                payload = {
                    "sub": user.id,
                    "email": user.email,
                    "role_id": user.role_id,
                    "exp": datetime.utcnow() + timedelta(days=1)
                }
                
                # Generate token
                token = jwt.encode(payload, "your-secret-key", algorithm="HS256")
                
                print(f"Login successful for {request.email}")
                return {
                    "access_token": token,
                    "token_type": "bearer"
                }
            else:
                print(f"Login failed: Invalid credentials for {request.email}")
                raise HTTPException(
                    status_code=401,
                    detail="Incorrect email or password"
                )
        finally:
            session.close()
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions directly
        raise http_exc
    except Exception as e:
        print(f"Login error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during login: {str(e)}"
        )

@app.post("/api/auth/register", response_model=Token)
async def register(request: UserCreate):
    print(f"\nRegistration attempt for email: {request.email}")
    try:
        # Create a new session
        session = Session()
        try:
            # Check if user exists
            existing_user = session.query(User).filter_by(email=request.email).first()
            if existing_user:
                raise HTTPException(
                    status_code=400,
                    detail="Email already registered"
                )
            
            # Create user
            new_user = User(
                email=request.email,
                password_hash=AuthService.hash_password(request.password),
                role_id=2  # Default to user role
            )
            session.add(new_user)
            session.commit()
            
            # Create profile
            profile = Profile(
                user_id=new_user.id,
                full_name=request.full_name
            )
            session.add(profile)
            session.commit()
            
            # Generate token
            from datetime import datetime, timedelta
            import jwt
            
            payload = {
                "sub": new_user.id,
                "email": new_user.email,
                "role_id": new_user.role_id,
                "exp": datetime.utcnow() + timedelta(days=1)
            }
            
            token = jwt.encode(payload, "your-secret-key", algorithm="HS256")
            
            print(f"Registration successful for {request.email}")
            return {
                "access_token": token,
                "token_type": "bearer"
            }
        finally:
            session.close()
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Registration error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during registration: {str(e)}"
        )

@app.post("/api/auth/forgot-password")
async def forgot_password(email: str):
    # For demo purposes, just return success
    return {"message": "Password reset email sent"}

@app.post("/api/auth/reset-password")
async def reset_password(token: str, password: str):
    # For demo purposes, just return success
    return {"message": "Password reset successful"}

# --- FastAPI Endpoints ---

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global db, rag_chain_runnable, llm, rag_prompt, str_parser, active_persistent_directory

    print(f"\nReceived file upload request for: {file.filename}")
    print(f"Content type: {file.content_type}")
    print(f"File size: {file.size if hasattr(file, 'size') else 'unknown'}")

    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    if llm is None or rag_prompt is None or str_parser is None:
        print("Error: LLM or RAG components not initialized")
        raise HTTPException(
            status_code=503,
            detail="LLM or RAG prompt not configured. Backend setup incomplete."
        )

    # Create a temporary directory to store uploaded files if it doesn't exist
    temp_upload_dir = os.path.join(current_dir, "temp_uploads")
    os.makedirs(temp_upload_dir, exist_ok=True)
    
    temp_pdf_path = os.path.join(temp_upload_dir, file.filename)

    try:
        # Save uploaded file temporarily
        print(f"Saving file to: {temp_pdf_path}")
        content = await file.read()
        print(f"Read {len(content)} bytes from uploaded file")
        
        with open(temp_pdf_path, "wb") as buffer:
            buffer.write(content)
        print(f"File saved successfully, size: {os.path.getsize(temp_pdf_path)} bytes")

        # Verify the saved file
        if not os.path.exists(temp_pdf_path):
            raise HTTPException(
                status_code=500,
                detail="Failed to save uploaded file"
            )

        # Try to open the PDF with PyPDF2 first to check if it's valid
        try:
            with open(temp_pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                num_pages = len(pdf_reader.pages)
                print(f"PDF validation: {num_pages} pages found")
                
                # Try to extract text from first page
                first_page = pdf_reader.pages[0]
                first_page_text = first_page.extract_text()
                print(f"First page text preview: {first_page_text[:200] if first_page_text else 'No text found'}")
        except Exception as pdf_err:
            print(f"PDF validation error: {str(pdf_err)}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid or corrupted PDF file: {str(pdf_err)}"
            )

        # 1. Extract text from the uploaded PDF
        print("Extracting text from PDF...")
        pdf_text_docs = pdf_extract(temp_pdf_path)
        if not pdf_text_docs:
            raise HTTPException(
                status_code=400,
                detail=f"Could not extract text from PDF: {file.filename}. Please ensure the PDF is not corrupted and contains extractable text."
            )
        print(f"Successfully extracted {len(pdf_text_docs)} pages")

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
    
    if llm is None:
        print("Error: LLM is not configured. Cannot process query.")
        raise HTTPException(status_code=503, detail="Backend LLM service is not configured or unavailable.")

    try:
        # Generate a conversation ID if this is a new conversation
        conversation_id = str(int(datetime.now().timestamp()))
        
        # Create user message
        user_message = Message(
            id=str(int(datetime.now().timestamp())),
            role="user",
            content=request.query,
            timestamp=datetime.now().isoformat()
        )
        
        # Get answer from RAG chain
        answer = rag_chain_runnable.invoke({"query": request.query})
        
        # Create assistant message
        assistant_message = Message(
            id=str(int(datetime.now().timestamp()) + 1),
            role="assistant",
            content=answer,
            timestamp=datetime.now().isoformat()
        )
        
        # Store or update conversation
        if conversation_id not in conversations:
            conversations[conversation_id] = Conversation(
                id=conversation_id,
                title=request.query[:50] + "..." if len(request.query) > 50 else request.query,
                messages=[user_message, assistant_message],
                timestamp=datetime.now().isoformat()
            )
        else:
            conversations[conversation_id].messages.extend([user_message, assistant_message])
        
        return {
            "response": answer,
            "conversation_id": conversation_id
        }
    except Exception as e:
        print(f"\nAn error occurred during RAG chain execution: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the query: {e}")


@app.get("/chat/history")
async def get_chat_history():
    """Get all chat conversations."""
    try:
        return {
            "conversations": list(conversations.values())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a specific conversation."""
    try:
        if conversation_id not in conversations:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        del conversations[conversation_id]
        return {"message": "Conversation deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard/stats")
async def get_dashboard_stats():
    """Get dashboard statistics."""
    try:
        # Calculate stats from conversations
        total_queries = sum(len(conv.messages) // 2 for conv in conversations.values())
        avg_response_time = 0.8  # This would be calculated from actual response times
        
        # Generate activity data for the last 7 days
        today = datetime.now()
        activity_data = []
        for i in range(7):
            date = today - timedelta(days=i)
            queries = sum(1 for conv in conversations.values() 
                        if datetime.fromisoformat(conv.timestamp).date() == date.date())
            activity_data.append({
                "name": date.strftime("%a"),
                "queries": queries
            })
        activity_data.reverse()
        
        return {
            "documentCount": len(os.listdir(persistent_db_base_path)) if os.path.exists(persistent_db_base_path) else 0,
            "queryCount": total_queries,
            "avgResponseTime": avg_response_time,
            "activityData": activity_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return {"message": "RAG backend with PDF upload is running. POST to /upload_pdf to load a document, then POST to /ask to query it."}

# To run this app: uvicorn your_filename:app --reload
# (replace your_filename with the actual name of your Python file)