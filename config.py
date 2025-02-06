# config.py

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    DATABASE_URL: str
    SLACK_BOT_TOKEN: str
    SLACK_SIGNING_SECRET: str
    SLACK_VERIFICATION_TOKEN: str
    MONGODB_URI: str
    BASE_URL: str = "http://localhost:8000"  # Change this to your actual domain in production

    class Config:
        env_file = ".env"  # Loads variables from the .env file

# Initialize settings
settings = Settings()