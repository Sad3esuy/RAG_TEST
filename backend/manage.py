# backend/manage.py
import os
import click
from flask.cli import FlaskGroup
from app import create_app
from app.db.models import db, User, Role, Permission, Profile
from app.services.auth_service import AuthService
import datetime
from sqlalchemy import text, select

app = create_app()
cli = FlaskGroup(create_app=create_app)

@cli.command('create-tables')
def create_tables():
    """Create database tables"""
    with app.app_context():
        try:
            # Create schema if it doesn't exist
            with db.engine.connect() as conn:
                # Create schema
                conn.execute(text('CREATE SCHEMA IF NOT EXISTS rag_app'))
                
                # Create pgvector extension if it doesn't exist
                conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
                
                # Set search path to include our schema
                conn.execute(text('SET search_path TO rag_app, public'))
                
                conn.commit()
            
            # Set the schema for all tables
            for table in db.metadata.tables.values():
                table.schema = 'rag_app'
            
            # Create tables
            db.create_all()
            click.echo('Tables created successfully!')
        except Exception as e:
            click.echo(f'Error creating tables: {str(e)}', err=True)
            raise

@cli.command('init-db')
def init_db():
    """Initialize database with required data"""
    with app.app_context():
        try:
            # Create roles if they don't exist
            roles = {
                1: 'admin',
                2: 'user',
                3: 'guest'
            }
            
            for role_id, role_name in roles.items():
                # Use SQLAlchemy 2.0 style query
                stmt = select(Role).where(Role.role_id == role_id)
                role = db.session.execute(stmt).scalar_one_or_none()
                
                if not role:
                    role = Role(role_id=role_id, role_name=role_name)
                    db.session.add(role)
                    click.echo(f'Created role: {role_name}')
            
            # Create permissions
            permissions = [
                'document:read', 'document:write',
                'chat:read', 'chat:write',
                'user:manage'
            ]
            
            for perm_name in permissions:
                stmt = select(Permission).where(Permission.name == perm_name)
                perm = db.session.execute(stmt).scalar_one_or_none()
                
                if not perm:
                    perm = Permission(name=perm_name)
                    db.session.add(perm)
                    click.echo(f'Created permission: {perm_name}')
            
            db.session.commit()
            
            # Create admin user if it doesn't exist
            admin_email = os.environ.get('ADMIN_EMAIL', 'admin@example.com')
            admin_password = os.environ.get('ADMIN_PASSWORD', 'Admin@123')
            
            stmt = select(User).where(User.email == admin_email)
            admin = db.session.execute(stmt).scalar_one_or_none()
            
            if not admin:
                admin = User(
                    email=admin_email,
                    password_hash=AuthService.hash_password(admin_password),
                    role_id=1  # Admin role
                )
                db.session.add(admin)
                db.session.commit()
                
                # Create admin profile
                profile = Profile(
                    user_id=admin.user_id,
                    full_name='Admin User'
                )
                db.session.add(profile)
                db.session.commit()
                
                click.echo(f'Admin user created: {admin_email}')
            else:
                click.echo(f'Admin user already exists: {admin_email}')
                
            click.echo('Database initialization completed successfully!')
            
        except Exception as e:
            db.session.rollback()
            click.echo(f'Error initializing database: {str(e)}', err=True)
            raise

@cli.command('create-user')
@click.argument('email')
@click.argument('password')
@click.option('--name', prompt='Full name', help='User\'s full name')
@click.option('--role', type=int, default=2, help='User role ID (1=admin, 2=user, 3=guest)')
def create_user(email, password, name, role):
    """Create a new user"""
    with app.app_context():
        try:
            # Check if user exists using SQLAlchemy 2.0 style
            stmt = select(User).where(User.email == email)
            if db.session.execute(stmt).scalar_one_or_none():
                click.echo(f'Error: User with email {email} already exists', err=True)
                return
            
            # Create user
            user = User(
                email=email,
                password_hash=AuthService.hash_password(password),
                role_id=role
            )
            db.session.add(user)
            db.session.commit()
            
            # Create profile
            profile = Profile(
                user_id=user.user_id,
                full_name=name
            )
            db.session.add(profile)
            db.session.commit()
            
            click.echo(f'User created successfully: {email}')
            
        except Exception as e:
            db.session.rollback()
            click.echo(f'Error creating user: {str(e)}', err=True)
            raise

@cli.command('list-users')
def list_users():
    """List all users in the database"""
    with app.app_context():
        try:
            stmt = select(User)
            users = db.session.execute(stmt).scalars().all()
            
            if not users:
                click.echo('No users found in the database')
                return
                
            click.echo('\nUsers in database:')
            click.echo('-' * 80)
            click.echo(f'{"ID":<5} {"Email":<30} {"Role":<10} {"Created At":<20}')
            click.echo('-' * 80)
            
            for user in users:
                role_stmt = select(Role).where(Role.role_id == user.role_id)
                role = db.session.execute(role_stmt).scalar_one_or_none()
                
                profile_stmt = select(Profile).where(Profile.user_id == user.user_id)
                profile = db.session.execute(profile_stmt).scalar_one_or_none()
                
                click.echo(f'{user.user_id:<5} {user.email:<30} {role.role_name:<10} {user.created_at.strftime("%Y-%m-%d %H:%M:%S"):<20}')
                
        except Exception as e:
            click.echo(f'Error listing users: {str(e)}', err=True)
            raise

@cli.command('reset-password')
@click.argument('email')
@click.password_option(confirmation_prompt=True)
def reset_password(email, password):
    """Reset a user's password"""
    with app.app_context():
        try:
            stmt = select(User).where(User.email == email)
            user = db.session.execute(stmt).scalar_one_or_none()
            
            if not user:
                click.echo(f'Error: User with email {email} not found', err=True)
                return
            
            user.password_hash = AuthService.hash_password(password)
            user.updated_at = datetime.datetime.utcnow()
            db.session.commit()
            
            click.echo(f'Password reset successfully for user: {email}')
            
        except Exception as e:
            db.session.rollback()
            click.echo(f'Error resetting password: {str(e)}', err=True)
            raise

if __name__ == '__main__':
    cli()