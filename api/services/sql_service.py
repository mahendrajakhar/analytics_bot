from typing import Dict, Optional, Tuple
import logging
from databricks import sql
from api.config import settings, logger, langchain_service
import asyncio
import pandas as pd
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class SQLService:
    def __init__(self):
        self.connection_params = {
            "server_hostname": settings.DATABRICKS_HOST,
            "http_path": settings.DATABRICKS_HTTP_PATH,
            "access_token": settings.DATABRICKS_TOKEN
        }
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.max_retries = 2

    async def resolve_sql_error(self, error: str, sql_query: str, user_question: str) -> Optional[str]:
        """Use AI to resolve SQL query errors"""
        try:
            # Load schema information
            with open("static/db_structure.txt", "r") as f:
                db_structure = f.read()

            # Get relevant schema context
            schema_context = langchain_service.get_schema_context(user_question)

            system_prompt = """You are an expert SQL debugger. Given an SQL error, the original query, and database schema:
            1. Analyze the error message
            2. Check the query syntax
            3. Verify table and column names against the schema
            4. Suggest fixes for any issues found
            5. Provide a complete corrected SQL query
            
            Return ONLY the corrected SQL query without any explanation."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""
                Error: {error}
                Original Query: {sql_query}
                User Question: {user_question}
                
                Database Schema:
                {db_structure}
                
                """}
                # Relevant Schema Context:
                # {schema_context}
            ]

            # Get AI response
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7
            )

            corrected_query = response.choices[0].message.content.strip()
            logger.info(f"AI suggested corrected query: {corrected_query}")
            return corrected_query

        except Exception as e:
            logger.error(f"Error in resolve_sql_error: {e}")
            return None

    async def execute_query(self, sql_query: str, user_question: str = "", retry_count: int = 0) -> Tuple[Optional[Dict], Optional[str]]:
        """Execute SQL query with retry logic only on SQL errors"""
        try:
            # Clean up SQL query
            sql_query = sql_query.strip().replace("```sql", "").replace("```", "").strip()
            
            # Execute query in a thread pool since Databricks operations are blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._execute_databricks_query, sql_query)
            
            if isinstance(result, str):  # Error occurred
                error_msg = result
                
                # Only retry if it's a SQL error (not a connection/other error)
                sql_error_keywords = [
                    "syntax error",
                    "table not found",
                    "column not found",
                    "ambiguous column",
                    "invalid identifier",
                    "missing expression",
                    "invalid table name",
                    "cannot resolve"
                ]
                
                is_sql_error = any(keyword.lower() in error_msg.lower() 
                                 for keyword in sql_error_keywords)
                
                if is_sql_error and retry_count < self.max_retries:
                    logger.info(f"SQL error detected, attempting resolution (retry {retry_count + 1}/{self.max_retries})")
                    logger.info(f"Error message: {error_msg}")
                    
                    # Try to get corrected query
                    corrected_query = await self.resolve_sql_error(error_msg, sql_query, user_question)
                    
                    if corrected_query and corrected_query != sql_query:
                        return await self.execute_query(corrected_query, user_question, retry_count + 1)
                    else:
                        logger.info("Could not generate corrected query or same query returned")
                
                return None, error_msg
                
            return result, None

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error executing query: {error_msg}")
            return None, error_msg

    def _execute_databricks_query(self, sql_query: str) -> Dict:
        """Execute query on Databricks and return results"""
        with sql.connect(**self.connection_params) as connection:
            with connection.cursor() as cursor:
                try:
                    cursor.execute(sql_query)
                    
                    # Get column names
                    columns = [desc[0] for desc in cursor.description]
                    
                    # Fetch all rows
                    rows = cursor.fetchall()
                    
                    return {
                        "columns": columns,
                        "rows": [tuple(row) for row in rows]
                    }
                    
                except Exception as e:
                    return str(e)
