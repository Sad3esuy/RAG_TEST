# backend/app/__init__.py
import os
from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_migrate import Migrate
from app.db.models import db

def create_app(config=None):
    app = Flask(__name__)
    
    # Configure CORS with specific origin
    CORS(app, resources={
        r"/api/*": {
            "origins": [
                os.getenv('FRONTEND_URL', 'http://localhost:3000'),
                'http://127.0.0.1:3000'
            ],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True
        }
    })
    
    # Default configuration
    app.config.update(
        SECRET_KEY=os.getenv('SECRET_KEY', 'SECRET_KEY'),
        SQLALCHEMY_DATABASE_URI=os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/rag_app'),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        JWT_SECRET_KEY=os.getenv('JWT_SECRET_KEY', 'JWT_SECRET_KEY'),
        UPLOAD_FOLDER=os.getenv('UPLOAD_FOLDER', 'uploads'),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
        FRONTEND_URL=os.getenv('FRONTEND_URL', 'http://localhost:3000')
    )
    
    # Override with provided config
    if config:
        app.config.update(config)
    
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize extensions
    db.init_app(app)
    JWTManager(app)
    Migrate(app, db)
    
    # Register blueprints
    from .routes import auth_routes, file_routes, chat_routes
    
    app.register_blueprint(auth_routes.bp, url_prefix='/api/auth')
    app.register_blueprint(file_routes.bp, url_prefix='/api/files')
    app.register_blueprint(chat_routes.bp, url_prefix='/api/chat')
    
    # Health check route
    @app.route('/api/health')
    def health_check():
        return {'status': 'OK'}, 200
    
    return app

# For running with Flask CLI
app = create_app()

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)