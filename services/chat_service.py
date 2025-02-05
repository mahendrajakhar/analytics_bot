from sqlalchemy.orm import Session
from openai import AsyncOpenAI
from config import settings
# from models import ChatHistory
from schemas import ChatRequest, ChatResponse
from services.sql_service import SQLService
from utils.graph_utils import generate_graph
import json
import os
import re
import logging
from typing import Dict, Any
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class ChatService:
    def __init__(self, db: Session):
        self.db = db
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.sql_service = SQLService()

    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        try:
            # Get database schema and history
            schema = await self.sql_service.get_table_schema()
            history = self._load_history(request.user_id)
            
            # Prepare conversation context with schema
            messages = self._prepare_conversation_context(history, request.question, schema)
            
            # Get AI response
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7
            )

            assistant_response = response.choices[0].message.content
            sql_query = self._extract_sql_query(assistant_response)
            graph_code = self._extract_graph_code(assistant_response)
            query_result = None
            query_error = None
            graph_url = None

            # Execute SQL query if present
            if sql_query:
                query_result, query_error = await self.sql_service.execute_query(sql_query)
                
                if query_error:
                    assistant_response += f"\nError executing SQL: {query_error}"
                elif query_result:
                    try:
                        formatted_result = self._format_sql_result(query_result)
                        assistant_response += f"\nSQL Result: {formatted_result}"
                    except Exception as e:
                        assistant_response += f"\nError formatting results: {str(e)}"

            # Generate graph if code is present
            if graph_code:
                try:
                    if query_result:
                        graph_url = generate_graph(graph_code, query_result)
                    else:
                        graph_url = generate_graph(graph_code)
                except Exception as e:
                    # logger.error(f"Error generating graph: {str(e)}", exc_info=True)
                    pass

            # Comment out database storage for now
            # try:
            #     chat_history = ChatHistory(
            #         user_id=request.user_id,
            #         question=request.question,
            #         sql_query=sql_query,
            #         response=assistant_response
            #     )
            #     self.db.add(chat_history)
            #     self.db.commit()
            # except Exception:
            #     self.db.rollback()

            # Save to JSON file
            try:
                self._save_history(request.user_id, request.question, sql_query, assistant_response, graph_code)
            except Exception as e:
                # logger.error(f"Error saving to JSON file: {str(e)}", exc_info=True)
                pass

            # Return response with proper dictionary structure
            return ChatResponse(
                response=assistant_response,
                sql_query=sql_query,
                graph_url=graph_url,
                query_result=query_result if query_result else None,
                error=query_error
            )

        except Exception as e:
            # logger.error(f"Unexpected error in process_chat: {str(e)}", exc_info=True)
            raise

    def _prepare_conversation_context(self, history, new_question, schema: Dict[str, Any]):
        system_message = """You are a SQL assistant for MySQL. Generate SQL queries and graph code as needed.
Available tables and their schemas:
"""
        # Add schema information
        for table, columns in schema.items():
            system_message += f"\n{table}:\n"
            for col in columns:
                system_message += f"  - {col['column_name']} ({col['data_type']})\n"

        # Add instructions for graph generation
        system_message += "\nWhen generating graphs:\n"
        system_message += "1. Use matplotlib for visualization\n"
        system_message += "2. Access SQL results through the 'sql_result' dictionary\n"
        system_message += "3. Format code blocks with ```sql and ```python markers\n"

        messages = [{"role": "system", "content": system_message}]
        
        # Add chat history
        for chat in history[-5:]:  # Last 5 messages for context
            messages.append({"role": "user", "content": chat['question']})
            messages.append({"role": "assistant", "content": chat['response']})
        
        messages.append({"role": "user", "content": new_question})
        return messages

    def _extract_sql_query(self, response: str) -> str:
        if "```sql" in response:
            return response.split("```sql")[1].split("```")[0].strip()
        return ""

    def _extract_graph_code(self, response: str) -> str:
        if "```python" in response:
            return response.split("```python")[1].split("```")[0].strip()
        return ""

    def _format_sql_result(self, result: Dict) -> str:
        try:
            if not result or not result.get('rows'):
                return "No results found"
            
            rows = result['rows']
            if not rows:
                return "No results found"

            columns = result['columns']
            
            # Format as table
            formatted = "\nResults:\n"
            
            # Calculate column widths
            col_widths = {col: len(str(col)) for col in columns}
            for row in rows:
                for col in columns:
                    col_widths[col] = max(col_widths[col], len(str(row.get(col, ''))))
            
            # Format table
            try:
                # Header
                for col in columns:
                    formatted += f"{str(col):<{col_widths[col]}} | "
                formatted += "\n" + "-" * (sum(col_widths.values()) + len(columns) * 3)
                
                # Rows
                for row in rows:
                    formatted += "\n"
                    for col in columns:
                        value = str(row.get(col, ''))
                        formatted += f"{value:<{col_widths[col]}} | "
                
                return formatted
            except Exception as e:
                # logger.error(f"Error during table formatting: {str(e)}", exc_info=True)
                raise

        except Exception as e:
            # logger.error(f"Error in _format_sql_result: {str(e)}", exc_info=True)
            raise

    def _load_history(self, user_id: str):
        """Load chat history from JSON file"""
        history_path = "static/chat_history.json"
        os.makedirs("static", exist_ok=True)  # Ensure static directory exists
        
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as file:
                    all_history = json.load(file)
                    return all_history.get(user_id, [])
            except json.JSONDecodeError:
                print("Error reading history file, starting fresh")
                return []
        return []

    def _save_history(self, user_id, question, sql_query, response, graph_code):
        """Save chat history to JSON file"""
        history_path = "static/chat_history.json"
        os.makedirs("static", exist_ok=True)  # Ensure static directory exists
        
        # Load existing history or create new
        all_history = {}
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as file:
                    all_history = json.load(file)
            except json.JSONDecodeError:
                print("Error reading history file, starting fresh")

        # Get or create user history
        user_history = all_history.get(user_id, [])
        
        # Add new entry
        user_history.append({
            "question": question,
            "sql_query": sql_query,
            "response": response,
            "graph_code": graph_code,
            "timestamp": str(datetime.now())  # Add timestamp for reference
        })

        # Keep only last 50 conversations
        user_history = user_history[-50:]
        
        # Update and save
        all_history[user_id] = user_history
        try:
            with open(history_path, 'w') as file:
                json.dump(all_history, file, indent=4)
        except Exception as e:
            print(f"Error saving history: {str(e)}") 