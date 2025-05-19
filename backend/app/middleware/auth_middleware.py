# backend/app/middleware/auth_middleware.py
from functools import wraps
from flask import request, jsonify, g
from app.services.auth_service import AuthService
from app.db.models import User, Role, Permission

def token_required(f):
    """Decorator to check if JWT token is valid"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Check if Authorization header exists
        auth_header = request.headers.get('Authorization')
        if auth_header:
            # Bearer token format: "Bearer <token>"
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == 'bearer':
                token = parts[1]
        
        # No token provided
        if not token:
            return jsonify({
                'success': False,
                'message': 'Authentication token is missing'
            }), 401
        
        try:
            # Decode token
            payload = AuthService.verify_token(token)
            if not payload:
                return jsonify({
                    'success': False,
                    'message': 'Invalid authentication token'
                }), 401
            
            user_id = payload['sub']
            
            # Get user from database
            current_user = User.query.get(user_id)
            if not current_user:
                return jsonify({
                    'success': False,
                    'message': 'User associated with token not found'
                }), 401
            
            # Store current user in context
            g.current_user = current_user
            
            # Pass current user to the decorated function
            return f(current_user, *args, **kwargs)
            
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Authentication failed: {str(e)}'
            }), 401
    
    return decorated

def role_required(role_id):
    """Decorator to check if user has required role"""
    def decorator(f):
        @wraps(f)
        def decorated_function(current_user, *args, **kwargs):
            # Check if user has the required role
            if current_user.role_id != role_id:
                return jsonify({
                    'success': False,
                    'message': 'Access denied: Insufficient privileges'
                }), 403
            
            return f(current_user, *args, **kwargs)
        return decorated_function
    return decorator

def permission_required(permission_name):
    """Decorator to check if user has required permission"""
    def decorator(f):
        @wraps(f)
        def decorated_function(current_user, *args, **kwargs):
            # Get user's role
            role = Role.query.get(current_user.role_id)
            if not role:
                return jsonify({
                    'success': False,
                    'message': 'User role not found'
                }), 403
            
            # Check if role has the required permission
            permission_exists = False
            for permission in role.permissions:
                if permission.name == permission_name:
                    permission_exists = True
                    break
            
            if not permission_exists:
                return jsonify({
                    'success': False,
                    'message': f'Access denied: Required permission {permission_name} not found'
                }), 403
            
            return f(current_user, *args, **kwargs)
        return decorated_function
    return decorator