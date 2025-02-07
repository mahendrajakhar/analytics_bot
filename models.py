from sqlalchemy import Column, Integer, String, DateTime, Text, Enum
from database import Base
from datetime import datetime
import enum
import uuid

class CommandType(enum.Enum):
    SQL = "/sql"
    ASK = "/ask"
    GRAPH = "/graph"
    HELP = "/help"

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(36), index=True)  # UUID for conversation tracking
    user_id = Column(String(50), index=True)
    command = Column(String(10))  # Store command type
    question = Column(Text)
    sql_query = Column(Text, nullable=True)
    sql_error = Column(Text, nullable=True)  # Store SQL errors
    response = Column(Text)
    ai_response = Column(Text)  # Store raw AI response
    graph_code = Column(Text, nullable=True)
    graph_url = Column(String(255), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow) 