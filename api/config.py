from pydantic_settings import BaseSettings
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    DATABASE_URL: str
    SLACK_BOT_TOKEN: str
    SLACK_SIGNING_SECRET: str
    SLACK_VERIFICATION_TOKEN: str
    MONGODB_URI: str
    BASE_URL: str = "http://localhost:8000"

    class Config:
        env_file = ".env"

settings = Settings()

# Configure logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("slack_sdk").setLevel(logging.WARNING)
logging.getLogger("python_multipart").setLevel(logging.WARNING)

# Set basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Database setup
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)

SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
