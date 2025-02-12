from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, Optional, Tuple
import logging
import asyncio
from api.models import ChatRequest
from api.config import logger

logger = logging.getLogger(__name__)

class SQLService:
    def __init__(self, db: Session):
        self.db = db

    async def execute_query(self, sql_query: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Execute SQL query and return results"""
        try:
            # Execute query in a thread pool since SQLAlchemy operations are blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: self.db.execute(text(sql_query)))
            
            # Get column names
            columns = result.keys()
            
            # Fetch all rows
            rows = await loop.run_in_executor(None, result.fetchall)
            
            return {
                "columns": columns,
                "rows": [tuple(row) for row in rows]
            }, None

        except SQLAlchemyError as e:
            error_msg = str(e)
            logger.error(f"SQL Error: {error_msg}")
            return None, error_msg
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error executing query: {error_msg}")
            return None, error_msg
