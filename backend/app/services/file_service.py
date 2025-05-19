from typing import List, Optional
from werkzeug.datastructures import FileStorage
from ..db.models import db, Document, FileChunk
from datetime import datetime
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import logging
import PyPDF2
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

class FileService:
    def __init__(self, upload_folder: str):
        self.upload_folder = upload_folder
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.allowed_extensions = {'pdf'}
        
        # Ensure upload directory exists and is writable
        try:
            os.makedirs(upload_folder, exist_ok=True)
            # Test write permission
            test_file = os.path.join(upload_folder, '.test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            logger.error(f"Upload directory error: {str(e)}")
            raise RuntimeError(f"Upload directory is not writable: {str(e)}")
            
        try:
            # Initialize embeddings with a publicly available model
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Successfully initialized embedding model")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def _is_allowed_file(self, filename: str) -> bool:
        """Check if file type is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")
    
    def save_file(self, file: FileStorage, user_id: int) -> Document:
        """Save uploaded file and create document record"""
        try:
            logger.info(f"=== Starting file save process ===")
            logger.info(f"User ID: {user_id}")
            logger.info(f"File details - name: {file.filename}, type: {file.content_type}")
            logger.info(f"File object dict: {file.__dict__}")
            
            # Validate file
            if not file or not file.filename:
                logger.error("No file or filename provided")
                raise ValueError("No file provided")
                
            filename = secure_filename(file.filename)
            logger.info(f"Secure filename: {filename}")
            
            if not self._is_allowed_file(filename):
                logger.error(f"File type not allowed: {filename}")
                raise ValueError(f"File type not allowed. Only {', '.join(self.allowed_extensions)} files are supported")
            
            # Create user directory
            user_dir = os.path.join(self.upload_folder, str(user_id))
            logger.info(f"Creating user directory: {user_dir}")
            try:
                os.makedirs(user_dir, exist_ok=True)
                logger.info(f"User directory created/verified: {user_dir}")
            except Exception as e:
                logger.error(f"Failed to create user directory: {str(e)}")
                raise RuntimeError(f"Failed to create upload directory: {str(e)}")
            
            # Save file
            file_path = os.path.join(user_dir, filename)
            logger.info(f"Saving file to: {file_path}")
            try:
                file.save(file_path)
                logger.info("File saved successfully")
            except Exception as e:
                logger.error(f"Failed to save file: {str(e)}")
                raise RuntimeError(f"Failed to save file: {str(e)}")
            
            # Verify file was saved and check size
            if not os.path.exists(file_path):
                logger.error("File was not saved successfully")
                raise RuntimeError("Failed to save file")
                
            file_size = os.path.getsize(file_path)
            logger.info(f"Saved file size: {file_size} bytes")
            
            if file_size > self.max_file_size:
                logger.error(f"File too large: {file_size} bytes")
                os.remove(file_path)  # Clean up
                raise ValueError(f"File too large. Maximum size is {self.max_file_size/1024/1024}MB")
            
            logger.info("File saved successfully, creating document record")
            
            # Create document record
            try:
                document = Document(
                    user_id=user_id,
                    filename=filename,
                    filetype=file.content_type,
                    size_bytes=file_size,
                    status='processing'
                )
                
                db.session.add(document)
                db.session.commit()
                logger.info(f"Document record created with ID: {document.document_id}")
                
                return document
            except Exception as e:
                logger.error(f"Failed to create document record: {str(e)}")
                # Clean up the saved file
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise
            
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}", exc_info=True)
            # Clean up if file was saved but database operation failed
            if 'file_path' in locals() and os.path.exists(file_path):
                logger.info(f"Cleaning up file: {file_path}")
                os.remove(file_path)
            raise
    
    async def process_document(self, document: Document) -> None:
        """Process document and create chunks with embeddings"""
        try:
            logger.info(f"Starting document processing for document {document.document_id}")
            file_path = os.path.join(self.upload_folder, str(document.user_id), document.filename)
            
            if not os.path.exists(file_path):
                logger.error(f"Document file not found: {file_path}")
                raise FileNotFoundError(f"Document file not found: {file_path}")
            
            # Extract text from PDF
            logger.info("Extracting text from PDF...")
            text = self._extract_text_from_pdf(file_path)
            if not text.strip():
                logger.error("No text content found in PDF")
                raise ValueError("No text content found in PDF")
            
            # Split into chunks
            logger.info("Splitting text into chunks...")
            chunks = self.text_splitter.split_text(text)
            if not chunks:
                logger.error("No text chunks generated from document")
                raise ValueError("No text chunks generated from document")
            
            logger.info(f"Generated {len(chunks)} chunks, getting embeddings...")
            
            try:
                # Get embeddings for all chunks
                embeddings = self.embeddings.embed_documents(chunks)
                logger.info("Successfully generated embeddings")
            except Exception as e:
                logger.error(f"Error generating embeddings: {str(e)}")
                raise RuntimeError(f"Failed to generate embeddings: {str(e)}")
            
            # Create chunk records
            logger.info("Creating chunk records...")
            for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                chunk = FileChunk(
                    document_id=document.document_id,
                    chunk_text=chunk_text,
                    embedding=embedding
                )
                db.session.add(chunk)
                if (i + 1) % 10 == 0:  # Log progress every 10 chunks
                    logger.info(f"Processed {i + 1}/{len(chunks)} chunks")
            
            # Update document status
            logger.info("Updating document status...")
            document.status = 'processed'
            db.session.commit()
            logger.info(f"Document {document.document_id} processed successfully")
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}", exc_info=True)
            document.status = 'failed'
            db.session.commit()
            raise
    
    def get_user_documents(self, user_id: int) -> List[Document]:
        """Get all documents for a user"""
        return Document.query.filter_by(user_id=user_id).all()
    
    def get_document(self, document_id: int, user_id: int) -> Optional[Document]:
        """Get specific document if it belongs to user"""
        return Document.query.filter_by(
            document_id=document_id,
            user_id=user_id
        ).first()
    
    def delete_document(self, document_id: int, user_id: int) -> bool:
        """Delete document and associated file"""
        document = self.get_document(document_id, user_id)
        if not document:
            return False
            
        # Delete file
        file_path = os.path.join(self.upload_folder, str(user_id), document.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            
        # Delete from database
        db.session.delete(document)
        db.session.commit()
        
        return True 