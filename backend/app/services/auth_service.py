# backend/app/services/auth_service.py
import os
import jwt
import bcrypt
import datetime
from typing import Dict, Optional, Tuple
from datetime import timedelta
from passlib.hash import pbkdf2_sha256
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import secrets
import string

# Database models will be imported from db module
from app.db.models import db, User, Profile, ResetToken

# JWT Configuration
JWT_SECRET = os.environ.get('JWT_SECRET', 'JWT_SECRET_KEY')  # Should be in env
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION = 24  # hours

# Email Service Configuration (should be in env)
EMAIL_HOST = os.environ.get('EMAIL_HOST', 'smtp.gmail.com')
EMAIL_PORT = int(os.environ.get('EMAIL_PORT', 587))
EMAIL_USER = os.environ.get('EMAIL_USER', 'demotestmot@gmail.com')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', 'app-password')
EMAIL_FROM = os.environ.get('EMAIL_FROM', 'no-reply@ragvlangchain.com')

class AuthService:
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using pbkdf2_sha256"""
        return pbkdf2_sha256.hash(password)
    
    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return pbkdf2_sha256.verify(password, password_hash)
    
    @staticmethod
    def generate_token(user_id: int, email: str, role_id: int) -> str:
        """Generate JWT token with user information"""
        payload = {
            'sub': user_id,
            'email': email,
            'role_id': role_id,
            'exp': datetime.datetime.utcnow() + timedelta(hours=JWT_EXPIRATION)
        }
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    @staticmethod
    def verify_token(token: str) -> Optional[Dict]:
        """Verify and decode JWT token"""
        try:
            return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        except jwt.PyJWTError:
            return None
    
    @staticmethod
    def register_user(email: str, password: str, full_name: str = None) -> Tuple[bool, str, Optional[User]]:
        """Register a new user"""
        try:
            # Check if user exists
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                return False, "Email already registered", None
            
            # Hash password
            password_hash = AuthService.hash_password(password)
            
            # Create user with default role_id=2 (user role)
            new_user = User(
                email=email,
                password_hash=password_hash,
                role_id=2,  # Default user role
                created_at=datetime.datetime.utcnow()
            )
            
            db.session.add(new_user)
            db.session.commit()
            
            # Create user profile
            profile = Profile(
                user_id=new_user.user_id,
                full_name=full_name or ""
            )
            
            db.session.add(profile)
            db.session.commit()
            
            return True, "Registration successful", new_user
        except Exception as e:
            db.session.rollback()
            return False, f"Registration failed: {str(e)}", None
    
    @staticmethod
    def login_user(email: str, password: str) -> Tuple[bool, str, Optional[Dict]]:
        """Login a user"""
        try:
            # Find user
            user = User.query.filter_by(email=email).first()
            
            if not user:
                return False, "Invalid email or password", None
            
            # Verify password
            if not AuthService.verify_password(password, user.password_hash):
                return False, "Invalid email or password", None
            
            # Generate token
            token = AuthService.generate_token(user.user_id, user.email, user.role_id)
            
            # Get user profile
            profile = Profile.query.filter_by(user_id=user.user_id).first()
            
            return True, "Login successful", {
                "token": token,
                "user": {
                    "id": user.user_id,
                    "email": user.email,
                    "role_id": user.role_id,
                    "full_name": profile.full_name if profile else "",
                    "avatar_url": profile.avatar_url if profile else None
                }
            }
        except Exception as e:
            return False, f"Login failed: {str(e)}", None
    
    @staticmethod
    def send_password_reset_email(email: str) -> Tuple[bool, str]:
        """Send password reset email"""
        try:
            # Check if user exists
            user = User.query.filter_by(email=email).first()
            if not user:
                # Always return success to prevent email enumeration
                return True, "If your email exists in our system, you will receive a password reset link shortly"
            
            # Generate a random reset token
            token = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(64))
            
            # Store token in the database (replace old token if exists)
            reset_record = ResetToken.query.filter_by(user_id=user.user_id).first()
            if reset_record:
                reset_record.token = token
                reset_record.expires_at = datetime.datetime.utcnow() + timedelta(hours=1)
            else:
                reset_record = ResetToken(
                    user_id=user.user_id,
                    token=token,
                    expires_at=datetime.datetime.utcnow() + timedelta(hours=1)
                )
                db.session.add(reset_record)
            
            db.session.commit()
            
            # Send email
            subject = "Password Reset Request"
            body = f"""
            <html>
            <body>
                <h2>Password Reset</h2>
                <p>You requested to reset your password.</p>
                <p>Please click the link below to reset your password:</p>
                <p><a href="http://localhost:3000/reset-password?token={token}">Reset Password</a></p>
                <p>This link will expire in 1 hour.</p>
                <p>If you did not request a password reset, please ignore this email.</p>
            </body>
            </html>
            """
            
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = EMAIL_FROM
            message["To"] = email
            
            # Add HTML content
            message.attach(MIMEText(body, "html"))
            
            # Send email
            with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
                server.starttls()
                server.login(EMAIL_USER, EMAIL_PASSWORD)
                server.sendmail(EMAIL_FROM, email, message.as_string())
            
            return True, "If your email exists in our system, you will receive a password reset link shortly"
            
        except Exception as e:
            print(f"Error sending reset email: {str(e)}")
            return False, "Failed to send password reset email"
    
    @staticmethod
    def reset_password(token: str, new_password: str) -> Tuple[bool, str]:
        """Reset user password using token"""
        try:
            # Find token in database
            reset_record = ResetToken.query.filter_by(token=token).first()
            
            if not reset_record:
                return False, "Invalid or expired reset token"
            
            # Check if token is expired
            if reset_record.expires_at < datetime.datetime.utcnow():
                db.session.delete(reset_record)
                db.session.commit()
                return False, "Reset token has expired"
            
            # Update user password
            user = User.query.get(reset_record.user_id)
            if not user:
                return False, "User not found"
            
            user.password_hash = AuthService.hash_password(new_password)
            user.updated_at = datetime.datetime.utcnow()
            
            # Delete the token
            db.session.delete(reset_record)
            db.session.commit()
            
            return True, "Password has been reset successfully"
            
        except Exception as e:
            db.session.rollback()
            return False, f"Password reset failed: {str(e)}"