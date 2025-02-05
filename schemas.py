from pydantic import BaseModel
from typing import Optional, List, Dict

class ChatRequest(BaseModel):
    user_id: str
    question: str

class ChatResponse(BaseModel):
    response: str
    sql_query: Optional[str] = None
    graph_url: Optional[str] = None
    query_result: Optional[Dict] = None
    error: Optional[str] = None 