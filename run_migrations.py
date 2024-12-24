import os
from alembic import command
from alembic.config import Config
from sqlalchemy_utils import database_exists, create_database

def run_migrations():
    database_url = os.getenv("DATABASE_URL")
    
    # Create database if it doesn't exist
    if not database_exists(database_url):
        create_database(database_url)
    
    # Run migrations
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")

if __name__ == "__main__":
    run_migrations()