# backend/app/routes/auth_routes.py
from flask import Blueprint, request, jsonify
from app.services.auth_service import AuthService
from app.middleware.auth_middleware import token_required, role_required

bp = Blueprint('auth', __name__, url_prefix='/api/auth')

@bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    
    # Validate request data
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'success': False, 'message': 'Missing required fields'}), 400
    
    email = data.get('email')
    password = data.get('password')
    full_name = data.get('fullName', '')
    
    # Validate email format
    # (Basic validation, you might want a more comprehensive one)
    if '@' not in email or '.' not in email:
        return jsonify({'success': False, 'message': 'Invalid email format'}), 400
    
    # Validate password strength
    if len(password) < 8:
        return jsonify({'success': False, 'message': 'Password must be at least 8 characters'}), 400
    
    # Register user
    success, message, user = AuthService.register_user(email, password, full_name)
    
    if success:
        # Generate token for auto-login
        token = AuthService.generate_token(user.user_id, user.email, user.role_id)
        return jsonify({
            'success': True,
            'message': message,
            'token': token,
            'user': {
                'id': user.user_id,
                'email': user.email,
                'fullName': full_name,
                'role_id': user.role_id
            }
        }), 201
    else:
        return jsonify({'success': False, 'message': message}), 400

@bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    
    # Validate request data
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'success': False, 'message': 'Missing email or password'}), 400
    
    email = data.get('email')
    password = data.get('password')
    
    # Login user
    success, message, result = AuthService.login_user(email, password)
    
    if success:
        return jsonify({
            'success': True,
            'message': message,
            'token': result['token'],
            'user': result['user']
        }), 200
    else:
        return jsonify({'success': False, 'message': message}), 401

@bp.route('/forgot-password', methods=['POST'])
def forgot_password():
    data = request.get_json()
    
    # Validate request data
    if not data or not data.get('email'):
        return jsonify({'success': False, 'message': 'Email is required'}), 400
    
    email = data.get('email')
    
    # Send password reset email
    success, message = AuthService.send_password_reset_email(email)
    
    # Always return 200 to prevent email enumeration
    return jsonify({'success': success, 'message': message}), 200

@bp.route('/reset-password', methods=['POST'])
def reset_password():
    data = request.get_json()
    
    # Validate request data
    if not data or not data.get('token') or not data.get('password'):
        return jsonify({'success': False, 'message': 'Token and new password are required'}), 400
    
    token = data.get('token')
    password = data.get('password')
    
    # Validate password strength
    if len(password) < 8:
        return jsonify({'success': False, 'message': 'Password must be at least 8 characters'}), 400
    
    # Reset password
    success, message = AuthService.reset_password(token, password)
    
    if success:
        return jsonify({'success': True, 'message': message}), 200
    else:
        return jsonify({'success': False, 'message': message}), 400

@bp.route('/me', methods=['GET'])
@token_required
def get_me(current_user):
    """Get current user info"""
    from app.db.models import Profile
    
    profile = Profile.query.filter_by(user_id=current_user.user_id).first()
    
    return jsonify({
        'success': True,
        'user': {
            'id': current_user.user_id,
            'email': current_user.email,
            'role_id': current_user.role_id,
            'full_name': profile.full_name if profile else "",
            'avatar_url': profile.avatar_url if profile else None,
            'bio': profile.bio if profile else None
        }
    }), 200

# Test protected routes for different roles
@bp.route('/admin-test', methods=['GET'])
@token_required
@role_required(1)  # Admin role
def admin_test(current_user):
    return jsonify({
        'success': True,
        'message': 'You have admin access',
        'user_id': current_user.user_id
    }), 200

@bp.route('/user-test', methods=['GET'])
@token_required
@role_required(2)  # User role
def user_test(current_user):
    return jsonify({
        'success': True,
        'message': 'You have user access',
        'user_id': current_user.user_id
    }), 200