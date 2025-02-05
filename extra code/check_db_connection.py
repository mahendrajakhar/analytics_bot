import os
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the DATABASE_URL from the environment
database_url = os.getenv('DATABASE_URL')

# Check if the DATABASE_URL is set
if not database_url:
    print("DATABASE_URL is not set in the .env file.")
else:
    try:
        # Create a SQLAlchemy engine
        engine = create_engine(database_url)

        # Try to connect to the database
        with engine.connect() as connection:
            print("Connection to the database was successful!")
    except OperationalError as e:
        print(f"Failed to connect to the database: {e}") 