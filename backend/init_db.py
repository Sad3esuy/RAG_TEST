import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

def init_database():
    # Load environment variables
    load_dotenv()
    
    # Get database connection parameters
    db_params = {
        'dbname': 'rag_app_db',  # Connect to default postgres database first
        'user': os.getenv('POSTGRES_USER'),
        'password': os.getenv('POSTGRES_PASSWORD'),
        'host': os.getenv('POSTGRES_HOST'),
        'port': os.getenv('POSTGRES_PORT')
    }
    
    try:
        # Connect to postgres database to create new database
        conn = psycopg2.connect(**db_params)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Check if database exists
        cur.execute("SELECT 1 FROM pg_database WHERE datname = 'rag_app'")
        exists = cur.fetchone()
        
        if not exists:
            print("Creating database 'rag_app'...")
            cur.execute('CREATE DATABASE rag_app')
            print("Database created successfully!")
        else:
            print("Database 'rag_app' already exists.")
        
        cur.close()
        conn.close()
        
        # Connect to rag_app database to create schema
        db_params['dbname'] = 'rag_app'
        conn = psycopg2.connect(**db_params)
        cur = conn.cursor()
        
        # Create schema if it doesn't exist
        cur.execute("SELECT 1 FROM pg_namespace WHERE nspname = 'rag_app'")
        exists = cur.fetchone()
        
        if not exists:
            print("Creating schema 'rag_app'...")
            cur.execute('CREATE SCHEMA rag_app')
            print("Schema created successfully!")
        else:
            print("Schema 'rag_app' already exists.")
        
        cur.close()
        conn.close()
        
        print("\nDatabase initialization completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python manage.py create-tables' to create the database tables")
        print("2. Run 'python manage.py init-db' to initialize the database with required data")
        
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        raise

if __name__ == '__main__':
    init_database() 