from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
from ..services.file_service import FileService
import os
import logging
from werkzeug.datastructures import FileStorage
import traceback

logger = logging.getLogger(__name__)

bp = Blueprint('files', __name__)
file_service = None
_service_initialized = False

@bp.before_app_request
def setup_file_service():
    global file_service, _service_initialized
    if not _service_initialized:
        upload_folder = current_app.config['UPLOAD_FOLDER']
        file_service = FileService(upload_folder)
        _service_initialized = True

@bp.route('/upload', methods=['POST'])
@jwt_required()
async def upload_file():
    """Upload a new document"""
    try:
        logger.info("=== Starting file upload process ===")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request content type: {request.content_type}")
        logger.info(f"Request files: {list(request.files.keys())}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"JWT Identity: {get_jwt_identity()}")
        logger.info(f"Request form data: {request.form}")
        logger.info(f"Request data: {request.get_data()}")
        
        # Check if file is present
        if 'file' not in request.files:
            logger.error("No 'file' key in request.files")
            logger.error(f"Available keys: {list(request.files.keys())}")
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        logger.info(f"File object type: {type(file)}")
        logger.info(f"File object attributes: {dir(file)}")
        logger.info(f"File object dict: {file.__dict__}")
        
        if not isinstance(file, FileStorage):
            logger.error(f"Invalid file object type: {type(file)}")
            return jsonify({'error': 'Invalid file object'}), 422
            
        # Log detailed file information
        logger.info("=== File Information ===")
        logger.info(f"Filename: {file.filename}")
        logger.info(f"Content type: {file.content_type}")
        logger.info(f"Content length: {file.content_length}")
        logger.info(f"Headers: {file.headers}")
        logger.info(f"File size: {len(file.read())}")
        file.seek(0)  # Reset file pointer after reading
        
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
            
        # Check content type
        if not file.content_type:
            logger.error("No content type provided")
            return jsonify({'error': 'File content type is required'}), 422
            
        if file.content_type != 'application/pdf':
            logger.error(f"Invalid content type: {file.content_type}")
            return jsonify({'error': f'Only PDF files are supported. Received: {file.content_type}'}), 422
            
        # Check file extension
        if not allowed_file(file.filename):
            logger.error(f"File type not allowed: {file.filename}")
            return jsonify({'error': f'File type not allowed. Only PDF files are supported. Received: {file.filename}'}), 422
        
        user_id = get_jwt_identity()
        logger.info(f"Processing upload for user_id: {user_id}")
        
        try:
            # Save file
            logger.info("Attempting to save file...")
            document = file_service.save_file(file, user_id)
            logger.info(f"File saved successfully, document_id: {document.document_id}")
            
            # Process document
            logger.info("Starting document processing...")
            await file_service.process_document(document)
            logger.info(f"Document processed successfully: {document.document_id}")
            
            return jsonify({
                'message': 'File uploaded successfully',
                'document': {
                    'id': document.document_id,
                    'filename': document.filename,
                    'status': document.status
                }
            }), 201
            
        except ValueError as ve:
            logger.error(f"Validation error: {str(ve)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': str(ve)}), 422
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 422

@bp.route('/documents', methods=['GET'])
@jwt_required()
def get_documents():
    """Get all documents for current user"""
    user_id = get_jwt_identity()
    documents = file_service.get_user_documents(user_id)
    
    return jsonify({
        'documents': [{
            'id': doc.document_id,
            'filename': doc.filename,
            'status': doc.status,
            'uploaded_at': doc.uploaded_at.isoformat()
        } for doc in documents]
    })

@bp.route('/documents/<int:document_id>', methods=['GET'])
@jwt_required()
def get_document(document_id):
    """Get specific document details"""
    user_id = get_jwt_identity()
    document = file_service.get_document(document_id, user_id)
    
    if not document:
        return jsonify({'error': 'Document not found'}), 404
    
    return jsonify({
        'document': {
            'id': document.document_id,
            'filename': document.filename,
            'status': document.status,
            'uploaded_at': document.uploaded_at.isoformat()
        }
    })

@bp.route('/documents/<int:document_id>', methods=['DELETE'])
@jwt_required()
def delete_document(document_id):
    """Delete a document"""
    user_id = get_jwt_identity()
    success = file_service.delete_document(document_id, user_id)
    
    if not success:
        return jsonify({'error': 'Document not found'}), 404
    
    return jsonify({'message': 'Document deleted successfully'})

def allowed_file(filename):
    """Check if file type is allowed"""
    ALLOWED_EXTENSIONS = {'pdf'}  # Only allow PDF files
    extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    logger.info(f"Checking file extension: {extension}")
    return extension in ALLOWED_EXTENSIONS 