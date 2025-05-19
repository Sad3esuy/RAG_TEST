from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from ..services.chat_service import ChatService

bp = Blueprint('chat', __name__)
chat_service = ChatService()

@bp.route('/chats', methods=['POST'])
@jwt_required()
def create_chat():
    """Create a new chat session"""
    user_id = get_jwt_identity()
    data = request.get_json()
    title = data.get('title') if data else None
    
    chat = chat_service.create_chat(user_id, title)
    
    return jsonify({
        'chat': {
            'id': chat.chat_id,
            'title': chat.title,
            'created_at': chat.created_at.isoformat()
        }
    }), 201

@bp.route('/chats', methods=['GET'])
@jwt_required()
def get_chats():
    """Get all chats for current user"""
    user_id = get_jwt_identity()
    chats = chat_service.get_user_chats(user_id)
    
    return jsonify({
        'chats': [{
            'id': chat.chat_id,
            'title': chat.title,
            'created_at': chat.created_at.isoformat()
        } for chat in chats]
    })

@bp.route('/chats/<int:chat_id>', methods=['GET'])
@jwt_required()
def get_chat(chat_id):
    """Get chat details and messages"""
    user_id = get_jwt_identity()
    chat = chat_service.get_chat(chat_id, user_id)
    
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404
    
    messages = chat_service.get_chat_messages(chat_id)
    
    return jsonify({
        'chat': {
            'id': chat.chat_id,
            'title': chat.title,
            'created_at': chat.created_at.isoformat(),
            'messages': [{
                'id': msg.message_id,
                'role': msg.role,
                'content': msg.content,
                'created_at': msg.created_at.isoformat()
            } for msg in messages]
        }
    })

@bp.route('/chats/<int:chat_id>', methods=['DELETE'])
@jwt_required()
def delete_chat(chat_id):
    """Delete a chat session"""
    user_id = get_jwt_identity()
    success = chat_service.delete_chat(chat_id, user_id)
    
    if not success:
        return jsonify({'error': 'Chat not found'}), 404
    
    return jsonify({'message': 'Chat deleted successfully'})

@bp.route('/chats/<int:chat_id>/messages', methods=['POST'])
@jwt_required()
async def send_message(chat_id):
    """Send a message in a chat"""
    user_id = get_jwt_identity()
    data = request.get_json()
    
    if not data or 'content' not in data:
        return jsonify({'error': 'Message content required'}), 400
    
    chat = chat_service.get_chat(chat_id, user_id)
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404
    
    try:
        result = await chat_service.process_message(
            chat_id=chat_id,
            user_id=user_id,
            content=data['content']
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500 