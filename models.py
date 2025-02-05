from sqlalchemy import Column, Integer, String, DateTime, Text
from database import Base
from datetime import datetime

class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), index=True)
    question = Column(Text)
    sql_query = Column(Text, nullable=True)
    response = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow) 