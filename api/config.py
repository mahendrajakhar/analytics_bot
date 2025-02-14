from pydantic_settings import BaseSettings
import logging
from databricks import sql
from api.services.langchain_service import LangChainService
import os

CHAT_HISTORY_PATH = os.path.join("static", "chat_history")

# Ensure chat history directory exists
os.makedirs(CHAT_HISTORY_PATH, exist_ok=True)

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    SLACK_BOT_TOKEN: str
    SLACK_SIGNING_SECRET: str
    SLACK_VERIFICATION_TOKEN: str
    MONGODB_URI: str
    BASE_URL: str = "http://localhost:8000"
    
    # Databricks settings
    DATABRICKS_HOST: str
    DATABRICKS_HTTP_PATH: str
    DATABRICKS_TOKEN: str

    # LangChain settings
    USE_GPU_FAISS: bool = False
    AUTO_REFRESH_SCHEMA: bool = True
    INDEX_CHUNK_SIZE: int = 1000
    INDEX_CHUNK_OVERLAP: int = 200

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

# Remove SQLAlchemy engine and session setup
# Instead add Databricks connection validation
def validate_databricks_connection():
    try:
        connection = sql.connect(
            server_hostname=settings.DATABRICKS_HOST,
            http_path=settings.DATABRICKS_HTTP_PATH,
            access_token=settings.DATABRICKS_TOKEN
        )
        connection.close()
        logger.info("Databricks connection validated successfully")
    except Exception as e:
        logger.error(f"Failed to validate Databricks connection: {e}")
        raise

# Add function to get chat history file path
def get_chat_history_path(user_id: str) -> str:
    return os.path.join(CHAT_HISTORY_PATH, f"{user_id}_history.json")

# Initialize connection validation
validate_databricks_connection()

# Initialize LangChain service with config
langchain_service = LangChainService(
    openai_api_key=settings.OPENAI_API_KEY,
    use_gpu=settings.USE_GPU_FAISS,
    auto_refresh=settings.AUTO_REFRESH_SCHEMA,
    chunk_size=settings.INDEX_CHUNK_SIZE,
    chunk_overlap=settings.INDEX_CHUNK_OVERLAP
)
