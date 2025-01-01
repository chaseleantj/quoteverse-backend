import os
from alembic import command
from alembic.config import Config
from sqlalchemy_utils import database_exists, create_database
from api.settings import settings


def run_migrations():
    try:
        database_url = settings.DATABASE_URL
        if not database_url:
            raise ValueError("DATABASE_URL environment variable is not set.")
        
        print(f"Running migrations on database: {database_url}")

        # Create database if it doesn't exist
        if not database_exists(database_url):
            create_database(database_url)
            print("Database created.")
        else:
            print("Database already exists.")

        # Configure Alembic with the correct database URL
        alembic_cfg = Config("alembic.ini")
        alembic_cfg.set_main_option("sqlalchemy.url", database_url)
        
        # Run migrations
        command.upgrade(alembic_cfg, "head")
        print("Migrations applied successfully.")
    except Exception as e:
        print(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    run_migrations()