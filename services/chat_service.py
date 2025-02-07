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
from api.slack.message_handlers import (update_final_message)
from api.slack.formatters import format_slack_message
import uuid

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

            # Save history with all details
            self._save_history(
                user_id=request.user_id,
                command="/graph" if graph_code else "/ask",
                question=request.question,
                sql_query=sql_query,
                response=assistant_response,
                ai_response=response.choices[0].message.content,
                graph_code=graph_code,
                graph_url=graph_url
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
        """Load chat history from MySQL"""
        try:
            from models import ChatHistory
            
            # Query last 5 chat entries
            history = (
                self.db.query(ChatHistory)
                .filter(ChatHistory.user_id == user_id)
                .order_by(ChatHistory.timestamp.desc())
                .limit(5)
                .all()
            )
            
            return [
                {
                    'command': chat.command,
                    'question': chat.question,
                    'ai_response': chat.ai_response,
                    'conversation_id': chat.conversation_id,
                    'timestamp': chat.timestamp.isoformat()
                }
                for chat in reversed(history)
            ]
            
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            return []

    def _save_history(self, user_id: str, command: str, question: str, sql_query: str, 
                     response: str, ai_response: str, conversation_id: str = None,
                     sql_error: str = None, graph_code: str = None, 
                     graph_url: str = None):
        """Save chat history to both MySQL and JSON"""
        try:
            # Generate conversation ID if not provided
            if not conversation_id:
                conversation_id = str(uuid.uuid4())

            # Save to MySQL
            from models import ChatHistory
            
            chat_entry = ChatHistory(
                conversation_id=conversation_id,
                user_id=user_id,
                command=command,
                question=question,
                sql_query=sql_query,
                sql_error=sql_error,
                response=response,
                ai_response=ai_response,
                graph_code=graph_code,
                graph_url=graph_url
            )
            
            self.db.add(chat_entry)
            self.db.commit()

            # Save to JSON
            history_file = "static/chat_history.json"
            os.makedirs("static", exist_ok=True)
            
            # Load existing or create new
            all_history = {}
            if os.path.exists(history_file):
                try:
                    with open(history_file, 'r') as f:
                        all_history = json.load(f)
                except json.JSONDecodeError:
                    pass

            if user_id not in all_history:
                all_history[user_id] = []
            
            # Add new entry
            json_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "conversation_id": conversation_id,
                "command": command,
                "question": question,
                "sql_query": sql_query,
                "sql_error": sql_error,
                "response": response,
                "ai_response": ai_response,
                "graph_code": graph_code,
                "graph_url": graph_url
            }
            
            all_history[user_id].append(json_entry)
            
            # Keep last 50 conversations
            all_history[user_id] = all_history[user_id][-50:]
            
            # Save to file
            with open(history_file, 'w') as f:
                json.dump(all_history, f, indent=2)
            
            return conversation_id
            
        except Exception as e:
            logger.error(f"Error saving history: {e}")
            self.db.rollback()
            return None

    async def get_sql_only(self, request: ChatRequest) -> ChatResponse:
        """Get SQL query without executing it"""
        try:
            history = self._load_history(request.user_id)
            messages = self._prepare_conversation_context(history, 
                "Generate only SQL query for this question: " + request.question)
            
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7
            )

            assistant_response = response.choices[0].message.content
            sql_query = self._extract_sql_query(assistant_response)
            
            self._save_history(
                user_id=request.user_id,
                command="/sql",
                question=request.question,
                sql_query=sql_query,
                response=assistant_response,
                ai_response=response.choices[0].message.content,
                graph_code=None,
                graph_url=None
            )

            return ChatResponse(
                response=assistant_response,
                sql_query=sql_query,
                graph_url=None,
                query_result=None,
                error=None
            )

        except Exception as e:
            logger.error(f"Error in get_sql_only: {str(e)}", exc_info=True)
            raise 

    async def process_ask_command(self, request: ChatRequest) -> ChatResponse:
        """Process /ask command - returns only results without SQL query"""
        try:
            # Load chat history
            history = self._load_history(request.user_id)
            
            # Prepare conversation context with specific instruction
            messages = self._prepare_conversation_context(history, 
                "Generate and execute SQL query for this question, return only the results: " + request.question)
            
            # Get AI response
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7
            )

            assistant_response = response.choices[0].message.content
            sql_query = self._extract_sql_query(assistant_response)
            query_result = None
            query_error = None

            # Execute SQL query if present
            if sql_query:
                query_result, query_error = await self.sql_service.execute_query(sql_query)
                
                if query_error:
                    return ChatResponse(
                        response="Error executing query",
                        sql_query=None,
                        graph_url=None,
                        query_result=None,
                        error=query_error
                    )

            # Save to MySQL database instead of JSON
            self._save_history(
                user_id=request.user_id,
                command="/ask",
                question=request.question,
                sql_query=sql_query,
                response=assistant_response,
                ai_response=assistant_response
            )

            return ChatResponse(
                response=assistant_response,
                sql_query=None,  # Don't send SQL query in response
                graph_url=None,
                query_result=query_result if query_result else None,
                error=query_error
            )

        except Exception as e:
            logger.error(f"Error in process_ask_command: {str(e)}", exc_info=True)
            raise 

    async def process_graph_command(self, request: ChatRequest, slack_client, mongodb_service, text, channel_id, user_id, initial_response) -> ChatResponse:
        """Process /graph command - returns only results for graph"""
        try:
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
                    
            if graph_code and query_result:
                # Generate and save graph
                img_buffer, local_path = self.generate_and_save_graph(
                    graph_code, 
                    query_result,
                    user_id,
                    text
                )
                
                if img_buffer:
                    # Upload graph
                    send_graph(channel_id, img_buffer, text, slack_client)
                    
                    # Store in MongoDB
                    image_url = mongodb_service.store_image(
                        img_buffer.getvalue(),
                        f"graph_{int(time.time())}.png",
                        f"slack_{user_id}"
                    )
                    
                    # Save history
                    self._save_history(
                        user_id=user_id,
                        command="/graph",
                        question=text,
                        sql_query=sql_query,
                        response=assistant_response,
                        ai_response=assistant_response,
                        graph_code=graph_code,
                        graph_url=image_url
                    )
                    
                    # Final update
                    return slack_client.chat_update(
                        channel=channel_id,
                        ts=initial_response['ts'],
                        text=f"{text}\n\nVisualization complete! ✨",
                        blocks=[
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": text
                                }
                            },
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": "Visualization complete! ✨"
                                }
                            }
                        ]
                    )
            
            return ChatResponse(
                response="Failed to generate graph",
                sql_query=sql_query,
                graph_url=None,
                query_result=query_result,
                error=query_error or "No graph code or query result available"
            )

        except Exception as e:
            logger.error(f"Error in process_graph_command: {str(e)}")
            raise

    def generate_and_save_graph(self, graph_code: str, sql_result: dict, user_id: str, 
                              question: str, conversation_id: str = None) -> tuple:
        """Generate and save graph, return buffer and local path"""
        try:
            # Clear any existing plots
            plt.clf()
            plt.close('all')
            
            # Create new figure with specific size
            plt.figure(figsize=(10, 6))
            
            # Create namespace with query results
            local_namespace = {
                'plt': plt,
                'sql_result': sql_result
            }
            
            # Execute graph code in the namespace
            exec(graph_code, globals(), local_namespace)
            
            # Ensure proper rendering
            plt.tight_layout()
            
            # Save to buffer
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            
            # Save locally
            timestamp = int(time.time())
            local_path = f"static/images/graph_{timestamp}.png"
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            plt.savefig(local_path, dpi=300, bbox_inches='tight')
            
            # Store in MongoDB
            graph_url = self.mongodb_service.store_image(
                img_buffer.getvalue(),
                f"graph_{timestamp}.png",
                user_id
            )
            
            # Save history
            self._save_history(
                user_id=user_id,
                command="/graph",
                question=question,
                sql_query="",  # No SQL query for graph generation
                response="Graph generated successfully",
                ai_response="",  # No AI response for graph generation
                conversation_id=conversation_id,
                graph_code=graph_code,
                graph_url=graph_url
            )
            
            # Clean up
            plt.close('all')
            
            return img_buffer, local_path
            
        except Exception as e:
            logger.error(f"Error generating graph: {e}")
            plt.close('all')
            return None, None

def send_graph(channel_id: str, img_buffer: bytes, text: str, slack_client):
    """Upload graph to Slack"""
    return slack_client.files_upload_v2(
        channel=channel_id,
        file=img_buffer,
        filename="graph.png",
        title=f"Visualization for: {text}",
        initial_comment="Here's your visualization 📈"
    )