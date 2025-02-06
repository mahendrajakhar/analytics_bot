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
import matplotlib.pyplot as plt
from services.mongodb_service import MongoDBService
import io
import time

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class ChatService:
    def __init__(self, db: Session):
        self.db = db
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.sql_service = SQLService(db)
        self.mongodb_service = MongoDBService()

    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        try:
            # Load chat history
            history = self._load_history(request.user_id)
            
            # Prepare conversation context with schema from JSON
            messages = self._prepare_conversation_context(history, request.question)
            
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
                        assistant_response = f"\nSQL Result: {formatted_result}"
                    except Exception as e:
                        assistant_response += f"\nError formatting results: {str(e)}"

            # Generate graph if code is present
            if graph_code and query_result:
                try:
                    logger.info("\n\n\nExecuting graph code...")
                    
                    # Clear any existing plots
                    plt.clf()
                    plt.close('all')
                    
                    # Create new figure
                    plt.figure(figsize=(10, 6))
                    
                    # Create namespace with query results
                    local_namespace = {
                        'plt': plt,
                        'sql_result': query_result
                    }
                    
                    # Execute graph code
                    exec(graph_code, globals(), local_namespace)
                    
                    # Ensure proper rendering
                    plt.tight_layout()
                    
                    # Save to buffer
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                    img_buffer.seek(0)
                    
                    # Store in MongoDB
                    graph_url = self.mongodb_service.store_image(
                        img_buffer.getvalue(),
                        f"graph_{int(time.time())}.png",
                        request.user_id
                    )
                    
                    # Clean up
                    plt.close('all')
                    
                    logger.info(f"\n\n\nGraph stored with URL: {graph_url}")
                    
                except Exception as e:
                    logger.error(f"Graph generation error: {e}")
                    plt.close('all')  # Ensure cleanup on error

            # Save to JSON file
            self._save_to_json(
                request.user_id,
                request.question,
                sql_query,
                assistant_response,
                graph_url
            )

            return ChatResponse(
                response=assistant_response,
                sql_query=sql_query,
                graph_url=graph_url,
                query_result=query_result if query_result else None,
                error=query_error
            )

        except Exception as e:
            logger.error(f"Unexpected error in process_chat: {str(e)}", exc_info=True)
            raise

    def _prepare_conversation_context(self, history, new_question):
        """Prepare conversation context with schema from JSON file"""
        try:
            # Load schema from JSON file
            with open("static/db_structure.json", "r") as f:
                db_structure = json.load(f)

            system_message = """You are a SQL assistant for MySQL. Generate SQL queries and graph code as needed.
Available tables and their schemas:
"""
            # Add schema directly from JSON file
            system_message += f"\n{json.dumps(db_structure, indent=2)}\n"

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
        except Exception as e:
            logger.error(f"Error preparing conversation context: {e}")
            raise

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

    def _save_to_json(self, user_id: str, question: str, sql_query: str, response: str, graph_url: str = None):
        """Save chat history to JSON file alongside existing storage methods"""
        try:
            history_file = "static/chat_history.json"
            
            # Create default structure if file doesn't exist
            if not os.path.exists(history_file):
                with open(history_file, "w") as f:
                    json.dump({}, f)
            
            # Read existing history
            with open(history_file, "r") as f:
                all_history = json.load(f)
            
            # Initialize user history if not exists
            if user_id not in all_history:
                all_history[user_id] = []
            
            # Add new chat entry
            chat_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "question": question,
                "sql_query": sql_query,
                "response": response,
                "graph_url": graph_url
            }
            
            all_history[user_id].append(chat_entry)
            
            # Save updated history
            with open(history_file, "w") as f:
                json.dump(all_history, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving to JSON file: {e}")
            # Don't raise exception to avoid interrupting main flow 