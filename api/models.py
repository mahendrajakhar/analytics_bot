from pydantic import BaseModel
from typing import Dict, Optional
from datetime import datetime
import enum

# Enum for command types
class CommandType(enum.Enum):
    SQL = "/sql"
    ASK = "/ask"
    GRAPH = "/graph"
    HELP = "/help"

# Pydantic schemas
class ChatRequest(BaseModel):
    user_id: str
    question: str

class ChatResponse(BaseModel):
    response: str
    sql_query: Optional[str] = None
    graph_url: Optional[str] = None
    query_result: Optional[Dict] = None
    error: Optional[str] = None

# Remove SQLAlchemy model since we're using JSON files for history
