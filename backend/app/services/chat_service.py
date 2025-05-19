from typing import List, Optional, Dict
from ..db.models import db, Chat, Message, FileChunk
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import numpy as np
from sqlalchemy import text
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        try:
            # Initialize embeddings with a publicly available model
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",  # Public model
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Successfully initialized embedding model")
            
            # Initialize DeepSeek chat model with proper error handling
            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
            if not deepseek_api_key:
                raise ValueError("DEEPSEEK_API_KEY environment variable is not set")
                
            self.llm = ChatDeepSeek(
                temperature=0.7,
                model_name="deepseek-chat",
                api_key=deepseek_api_key
            )
            logger.info("Successfully initialized DeepSeek chat model")
            
        except Exception as e:
            logger.error(f"Error initializing ChatService: {str(e)}")
            raise
        
    def create_chat(self, user_id: int, title: Optional[str] = None) -> Chat:
        """Create a new chat session"""
        chat = Chat(
            user_id=user_id,
            title=title or "New Chat"
        )
        db.session.add(chat)
        db.session.commit()
        return chat
    
    def get_chat(self, chat_id: int, user_id: int) -> Optional[Chat]:
        """Get chat if it belongs to user"""
        return Chat.query.filter_by(
            chat_id=chat_id,
            user_id=user_id
        ).first()
    
    def get_user_chats(self, user_id: int) -> List[Chat]:
        """Get all chats for a user"""
        return Chat.query.filter_by(user_id=user_id).all()
    
    def delete_chat(self, chat_id: int, user_id: int) -> bool:
        """Delete a chat session"""
        chat = self.get_chat(chat_id, user_id)
        if not chat:
            return False
            
        db.session.delete(chat)
        db.session.commit()
        return True
    
    def add_message(self, chat_id: int, role: str, content: str) -> Message:
        """Add a new message to the chat"""
        message = Message(
            chat_id=chat_id,
            role=role,
            content=content
        )
        db.session.add(message)
        db.session.commit()
        return message
    
    def get_chat_messages(self, chat_id: int) -> List[Message]:
        """Get all messages in a chat"""
        return Message.query.filter_by(chat_id=chat_id).order_by(Message.created_at).all()
    
    async def process_message(self, chat_id: int, user_id: int, content: str) -> Dict:
        """Process a user message and generate response"""
        try:
            # Get relevant document chunks
            query_embedding = self.embeddings.embed_query(content)
            
            # Find similar chunks using cosine similarity
            similar_chunks = FileChunk.query.filter(
                FileChunk.document.has(user_id=user_id)
            ).from_self(
                FileChunk,
                (1 - text('(embedding <=> :query_embedding)')).label('similarity')
            ).params(
                query_embedding=query_embedding
            ).order_by(
                text('similarity DESC')
            ).limit(5).all()
            
            if not similar_chunks:
                logger.warning(f"No similar chunks found for query: {content}")
                return {
                    "response": "I couldn't find any relevant information to answer your question.",
                    "sources": []
                }
            
            # Build context from similar chunks
            context = "\n".join([chunk.chunk_text for chunk in similar_chunks])
            
            # Get chat history
            messages = self.get_chat_messages(chat_id)
            history = []
            for msg in messages[-6:]:  # Get last 3 exchanges (6 messages)
                if msg.role in ['user', 'assistant']:
                    history.append((msg.role, msg.content))
            
            # Generate response using LLM
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            for role, content in history:
                memory.chat_memory.add_message(role, content)
            
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                memory=memory,
                condense_question=True
            )
            
            response = await qa_chain.arun(
                question=content,
                context=context
            )
            
            # Save user message and response
            self.add_message(chat_id, "user", content)
            self.add_message(chat_id, "assistant", response)
            
            return {
                "response": response,
                "sources": [
                    {
                        "text": chunk.chunk_text,
                        "similarity": chunk.similarity
                    }
                    for chunk in similar_chunks
                ]
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            raise 