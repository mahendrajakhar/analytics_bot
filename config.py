# config.py

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    DATABASE_URL: str = "mysql+pymysql://root:mahensql@localhost:3306/datasets_db"  # MySQL Connection String

    class Config:
        env_file = ".env"  # Loads variables from the .env file

# Initialize settings
settings = Settings()