from sqlalchemy import Column, Integer, String, Text, DateTime
from pydantic import BaseModel
from typing import Dict, Optional
from datetime import datetime
import enum
from api.config import Base

# Enum for command types
class CommandType(enum.Enum):
    SQL = "/sql"
    ASK = "/ask"
    GRAPH = "/graph"
    HELP = "/help"

# Database models
class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), index=True)
    command = Column(String(10))
    question = Column(Text)
    sql_query = Column(Text, nullable=True)
    response = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

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
