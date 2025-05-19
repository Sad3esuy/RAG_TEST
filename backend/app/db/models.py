# backend/app/db/models.py
from flask_sqlalchemy import SQLAlchemy
from pgvector.sqlalchemy import Vector
from datetime import datetime

db = SQLAlchemy()

class Role(db.Model):
    __tablename__ = 'roles'
    __table_args__ = {'schema': 'rag_app'}
    
    role_id = db.Column(db.Integer, primary_key=True)
    role_name = db.Column(db.String(50), unique=True, nullable=False)
    
    # Relationships
    users = db.relationship('User', backref='role')
    permissions = db.relationship('Permission', secondary='rag_app.role_permissions')

class Permission(db.Model):
    __tablename__ = 'permissions'
    __table_args__ = {'schema': 'rag_app'}
    
    perm_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)

class RolePermission(db.Model):
    __tablename__ = 'role_permissions'
    __table_args__ = {'schema': 'rag_app'}
    
    role_id = db.Column(db.Integer, db.ForeignKey('rag_app.roles.role_id', ondelete='CASCADE'), primary_key=True)
    perm_id = db.Column(db.Integer, db.ForeignKey('rag_app.permissions.perm_id', ondelete='CASCADE'), primary_key=True)

class User(db.Model):
    __tablename__ = 'users'
    __table_args__ = {'schema': 'rag_app'}
    
    user_id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role_id = db.Column(db.Integer, db.ForeignKey('rag_app.roles.role_id', ondelete='RESTRICT'), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)
    updated_at = db.Column(db.DateTime(timezone=True), onupdate=datetime.utcnow)
    
    # Relationships
    profile = db.relationship('Profile', backref='user', uselist=False, cascade='all, delete-orphan')
    documents = db.relationship('Document', backref='user', cascade='all, delete-orphan')
    chats = db.relationship('Chat', backref='user', cascade='all, delete-orphan')
    reset_tokens = db.relationship('ResetToken', backref='user', cascade='all, delete-orphan')

class Profile(db.Model):
    __tablename__ = 'profiles'
    __table_args__ = {'schema': 'rag_app'}
    
    user_id = db.Column(db.Integer, db.ForeignKey('rag_app.users.user_id', ondelete='CASCADE'), primary_key=True)
    full_name = db.Column(db.String(255))
    avatar_url = db.Column(db.Text)
    bio = db.Column(db.Text)

class Document(db.Model):
    __tablename__ = 'documents'
    __table_args__ = {'schema': 'rag_app'}
    
    document_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('rag_app.users.user_id', ondelete='CASCADE'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    filetype = db.Column(db.String(50))
    size_bytes = db.Column(db.BigInteger)
    status = db.Column(db.String(50), default='processing')
    uploaded_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    chunks = db.relationship('FileChunk', backref='document', cascade='all, delete-orphan')

class FileChunk(db.Model):
    __tablename__ = 'file_chunks'
    __table_args__ = {'schema': 'rag_app'}
    
    chunk_id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('rag_app.documents.document_id', ondelete='CASCADE'), nullable=False)
    chunk_text = db.Column(db.Text, nullable=False)
    embedding = db.Column(Vector(1536))
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)

class Chat(db.Model):
    __tablename__ = 'chats'
    __table_args__ = {'schema': 'rag_app'}
    
    chat_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('rag_app.users.user_id', ondelete='CASCADE'), nullable=False)
    title = db.Column(db.String(255))
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    messages = db.relationship('Message', backref='chat', cascade='all, delete-orphan')

class Message(db.Model):
    __tablename__ = 'messages'
    __table_args__ = {'schema': 'rag_app'}
    
    message_id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.Integer, db.ForeignKey('rag_app.chats.chat_id', ondelete='CASCADE'), nullable=False)
    role = db.Column(db.String(10), nullable=False)
    content = db.Column(db.Text, nullable=False)
    token_count = db.Column(db.Integer)
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)

# Additional table for password reset tokens
class ResetToken(db.Model):
    __tablename__ = 'reset_tokens'
    __table_args__ = {'schema': 'rag_app'}
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('rag_app.users.user_id', ondelete='CASCADE'), nullable=False)
    token = db.Column(db.String(255), unique=True, nullable=False)
    expires_at = db.Column(db.DateTime(timezone=True), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)