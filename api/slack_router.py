##################################################################################
##################################################################################
############################# Imports & Settings ##################################
##################################################################################
##################################################################################

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from openai import AsyncOpenAI
from pymongo import MongoClient
from bson.binary import Binary
from slack_sdk import WebClient
from slack_sdk.signature import SignatureVerifier
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
import json
import os
import re
import logging
import matplotlib.pyplot as plt
import io
import time
import uuid
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import ast
import enum
from sqlalchemy import Column, Integer, String, Text, DateTime

##################################################################################
##################################################################################
############################# Configuration ######################################
##################################################################################
##################################################################################

# Settings configuration
class Settings(BaseSettings):
    OPENAI_API_KEY: str
    DATABASE_URL: str
    SLACK_BOT_TOKEN: str
    SLACK_SIGNING_SECRET: str
    SLACK_VERIFICATION_TOKEN: str
    MONGODB_URI: str
    BASE_URL: str = "http://localhost:8000"

    class Config:
        env_file = ".env"

settings = Settings()

# Configure logging - more restrictive settings
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("slack_sdk").setLevel(logging.WARNING)
logging.getLogger("python_multipart").setLevel(logging.WARNING)

# Set basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

# Initialize logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize router
router = APIRouter()

##################################################################################
##################################################################################
############################# Database Setup #####################################
##################################################################################
##################################################################################

# Database setup
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)

SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()

##################################################################################
##################################################################################
############################# Models & Schemas ###################################
##################################################################################
##################################################################################

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

##################################################################################
##################################################################################
############################# Services ##########################################
##################################################################################
##################################################################################

# MongoDB service for image storage
class MongoDBService:
    def __init__(self):
        self.uri = settings.MONGODB_URI
        self.client = MongoClient(self.uri)
        self.db = self.client.analytics_bot
        self.images = self.db.images
        self.history = self.db.chat_history

    def store_image(self, image_data: bytes, filename: str, user_id: str) -> str:
        """Store image in MongoDB and return its URL"""
        try:
            image_binary = Binary(image_data)
            image_doc = {
                "filename": filename,
                "data": image_binary,
                "user_id": user_id,
                "created_at": datetime.utcnow(),
                "content_type": "image/png"
            }
            result = self.images.insert_one(image_doc)
            return f"/api/images/{result.inserted_id}"
        except Exception as e:
            logger.error(f"Error storing image in MongoDB: {e}")
            raise

    def get_image(self, image_id: str):
        """Retrieve image from MongoDB"""
        try:
            image_doc = self.images.find_one({"_id": image_id})
            if image_doc:
                return image_doc["data"], image_doc["content_type"]
            return None, None
        except Exception as e:
            logger.error(f"Error retrieving image from MongoDB: {e}")
            raise

# Move this initialization after the class definition
mongodb_service = MongoDBService()

# Initialize services after class definitions
slack_client = WebClient(token=settings.SLACK_BOT_TOKEN)
signature_verifier = SignatureVerifier(settings.SLACK_SIGNING_SECRET)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def format_query_result(query_result):
    """Format query result as a readable table for Slack"""
    if not query_result:
        return ""
    
    try:
        # Extract columns and rows
        columns = [str(col) for col in query_result['columns']]
        rows = query_result['rows']
        
        if not rows:
            return "No results found"
        
        # Calculate max width for each column
        col_widths = {col: len(col) for col in columns}
        for row in rows:
            for i, value in enumerate(row):
                col_widths[columns[i]] = max(col_widths[columns[i]], len(str(value)))
        
        # Build table
        table = []
        
        # Header
        header = "| " + " | ".join(f"{col:<{col_widths[col]}}" for col in columns) + " |"
        separator = "|" + "|".join("-" * (width + 2) for width in col_widths.values()) + "|"
        
        table.append(header)
        table.append(separator)
        
        # Rows
        for row in rows:
            row_str = "| " + " | ".join(f"{str(value):<{col_widths[columns[i]]}} " 
                          for i, value in enumerate(row)) + "|"
            table.append(row_str)
        
        return "```\n" + "\n".join(table) + "\n```"
        
    except Exception as e:
        logger.error(f"Error formatting table: {e}")
        return str(query_result)

def format_slack_message(response):
    """Format the response for Slack with support for large tables"""
    if not response.query_result:
        return {"type": "text", "text": "No results to display"}
    
    table_handler = TableHandler()
    
    # Analyze data dimensions
    num_rows, num_cols = table_handler.analyze_data(response.query_result)
    
    # Determine format based on data size
    if table_handler.should_use_excel(num_rows, num_cols):
        excel_result = table_handler.create_excel_file(response.query_result)
        if excel_result:
            excel_buffer, summary = excel_result
            return {
                "type": "excel",
                "excel_file": excel_buffer,
                "text": summary
            }
    
    # For smaller tables, use regular formatting
    formatted_table = format_query_result(response.query_result)
    return {
        "type": "table",
        "text": "*Results:*\n" + formatted_table
    }

class TableHandler:
    """Handles large table data formatting and conversion"""
    
    MAX_COLUMNS = 3
    MAX_ROWS = 15
    
    @staticmethod
    def analyze_data(query_result: Dict[str, Any]) -> Tuple[int, int]:
        """Analyze the data dimensions"""
        if not query_result or 'rows' not in query_result:
            return 0, 0
            
        columns = query_result.get('columns', [])
        rows = query_result.get('rows', [])
        
        return len(rows), len(columns)

    @staticmethod
    def should_use_excel(rows: int, columns: int) -> bool:
        """Determine if data should be sent as Excel file"""
        return rows > TableHandler.MAX_ROWS or columns > TableHandler.MAX_COLUMNS

    @staticmethod
    def create_excel_file(query_result: Dict[str, Any]) -> Optional[Tuple[io.BytesIO, str]]:
        """Create Excel file from query results"""
        try:
            columns = [str(col) for col in query_result['columns']]
            rows = query_result['rows']
            
            # Create DataFrame with all data
            df = pd.DataFrame(rows, columns=columns)
            
            # Create Excel file in memory
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Query Results', index=False)
                
                # Auto-adjust columns width
                worksheet = writer.sheets['Query Results']
                for idx, col in enumerate(df.columns):
                    max_length = max(
                        df[col].astype(str).apply(len).max(),
                        len(str(col))
                    ) + 2
                    worksheet.set_column(idx, idx, max_length)

            excel_buffer.seek(0)
            
            # Create summary text
            summary = (
                "*Query Results Summary:*\n"
                f"• Total Rows: {len(df)}\n"
                f"• Total Columns: {len(df.columns)}\n\n"
                "📊 Complete results available in the Excel file above."
            )
            
            return excel_buffer, summary
            
        except Exception as e:
            logger.error(f"Error creating Excel file: {e}")
            return None

    @staticmethod
    def format_preview(query_result: Dict[str, Any], max_rows: int = 5) -> str:
        """Format a preview of the data"""
        columns = [str(col) for col in query_result['columns']]
        rows = query_result['rows'][:max_rows]
        
        preview = ["```"]
        
        # Headers
        header = " | ".join(f"{col[:15]:<15}" for col in columns)
        separator = "-" * (17 * len(columns))
        
        preview.append(header)
        preview.append(separator)
        
        # Rows
        for row in rows:
            row_str = " | ".join(f"{str(val)[:15]:<15}" for val in row)
            preview.append(row_str)
        
        preview.append("```")
        return "\n".join(preview)

class SQLService:
    def __init__(self, db: Session):
        self.db = db

    async def get_sql_only(self, request: ChatRequest) -> Dict[str, Any]:
        """Get SQL query without executing it"""
        try:
            # Load schema information
            with open("static/db_structure.json", "r") as f:
                db_structure = json.load(f)

            # 1. System prompt
            system_message = """You are a SQL expert. Generate only the SQL query without executing it. 
Format your response with the SQL query between sql ``` sql query ```.
Provide clear and efficient queries that follow best practices."""

            # 2. Assistant message with schema and history context
            context_message = "Database schema and tables:\n"
            context_message += f"{json.dumps(db_structure, indent=2)}\n\n"
            
            # Add history context
            history = load_chat_history(request.user_id)
            if history:
                context_message += "Previous conversations:\n"
                for chat in history[-5:]:  # Last 5 conversations
                    context_message += f"\nUser: {chat['question']}"
                    if chat.get('sql_query'):
                        context_message += f"\nGenerated SQL: ```sql\n{chat['sql_query']}\n```"

            # Prepare messages array
            messages = [
                {"role": "system", "content": system_message},
                {"role": "assistant", "content": context_message},
                {"role": "user", "content": request.question}
            ]

            # Get response from OpenAI
            client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7
            )

            sql_query = self._extract_sql_query(response.choices[0].message.content)
            if not sql_query:
                return {
                    "error": "Could not generate SQL query",
                    "sql_query": None
                }

            # Save to history
            save_chat_history(
                user_id=request.user_id,
                command="/sql",
                question=request.question,
                sql_query=sql_query,
                ai_response=response.choices[0].message.content
            )

            return {
                "sql_query": sql_query,
                "error": None
            }

        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            return {
                "error": str(e),
                "sql_query": None
            }

    async def execute_query(self, sql_query: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Execute SQL query and return results"""
        try:
            # Execute query
            result = self.db.execute(text(sql_query))
            
            # Get column names
            columns = result.keys()
            
            # Fetch all rows
            rows = result.fetchall()
            
            return {
                "columns": columns,
                "rows": rows
            }, None

        except SQLAlchemyError as e:
            error_msg = str(e)
            logger.error(f"SQL Error: {error_msg}")
            return None, error_msg
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error executing query: {error_msg}")
            return None, error_msg

    def _extract_sql_query(self, response: str) -> Optional[str]:
        """Extract SQL query from response"""
        # Look for SQL between triple backticks
        sql_match = re.search(r"```sql\n(.*?)\n```", response, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
            
        # Look for SQL between single backticks
        sql_match = re.search(r"`(.*?)`", response, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
            
        # If no markers, try to extract just SQL-like text
        sql_match = re.search(r"SELECT.*?(?:;|$)", response, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(0).strip()
            
        return None

class SlackChatService:
    def __init__(self, db: Session):
        self.db = db
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.sql_service = SQLService(db)
        self.mongodb_service = MongoDBService()

    def _prepare_conversation_context(self, user_id: str, new_question: str):
        """Prepare conversation context with schema and history"""
        try:
            # Load schema information
            with open("static/db_structure.json", "r") as f:
                db_structure = json.load(f)

            # Create system message with schema info
            system_message = """You are a SQL assistant for MySQL. Generate SQL queries and graph code as needed.
Available tables and their schemas:
"""
            system_message += f"\n{json.dumps(db_structure, indent=2)}\n"
            system_message += "\nWhen generating graphs:\n"
            system_message += "1. Use matplotlib for visualization\n"
            system_message += "2. Access SQL results through the 'sql_result' dictionary\n"
            system_message += "3. Format code blocks with ```sql and ```python markers\n"

            # Load user's chat history
            history = load_chat_history(user_id)

            # Prepare conversation messages
            messages = [{"role": "system", "content": system_message}]
            
            # Add history to context
            for chat in history[-5:]:  # Last 5 conversations
                messages.append({"role": "user", "content": chat['question']})
                if chat.get('sql_query'):
                    assistant_response = f"Here's the SQL query:\n```sql\n{chat['sql_query']}\n```"
                    if chat.get('graph_url'):
                        assistant_response += f"\nGraph generated: {chat['graph_url']}"
                    messages.append({"role": "assistant", "content": assistant_response})
                elif chat.get('ai_response'):
                    messages.append({"role": "assistant", "content": chat['ai_response']})

            # Add current question
            messages.append({"role": "user", "content": new_question})
            return messages

        except Exception as e:
            logger.error(f"Error preparing conversation context: {e}")
            # Return basic context if there's an error
            return [
                {"role": "system", "content": "You are a SQL assistant for MySQL."},
                {"role": "user", "content": new_question}
            ]

    def _extract_sql_query(self, response: str) -> str:
        if "```sql" in response:
            return response.split("```sql")[1].split("```")[0].strip()
        return ""

    def _extract_graph_code(self, response: str) -> str:
        if "```python" in response:
            return response.split("```python")[1].split("```")[0].strip()
        return ""

    async def process_graph_command(self, request: ChatRequest, slack_client, 
                                  mongodb_service, text: str, channel_id: str, 
                                  user_id: str, initial_response: dict) -> dict:
        try:
            # Send processing message
            update_with_processing(channel_id, initial_response['ts'], "Generating visualization... 📊")

            # Load schema information
            with open("static/db_structure.json", "r") as f:
                db_structure = json.load(f)

            # 1. System prompt
            system_message = """You are a data visualization expert. Generate both SQL query and matplotlib code.
Format SQL with ```sql markers and Python code with ```python markers.
Focus on creating clear and informative visualizations."""

            # 2. Assistant message with schema and history context
            context_message = "Database schema and tables:\n"
            context_message += f"{json.dumps(db_structure, indent=2)}\n\n"
            context_message += "Visualization guidelines:\n"
            context_message += "1. Use matplotlib for visualization\n"
            context_message += "2. Access SQL results through the 'sql_result' dictionary\n"
            context_message += "3. Set appropriate labels and titles\n"
            
            # Add history context
            history = load_chat_history(request.user_id)
            if history:
                context_message += "\nPrevious conversations:\n"
                for chat in history[-5:]:
                    context_message += f"\nUser: {chat['question']}"
                    if chat.get('sql_query'):
                        context_message += f"\nSQL: ```sql\n{chat['sql_query']}\n```"
                    if chat.get('graph_url'):
                        context_message += f"\nGraph: {chat['graph_url']}"

            # Prepare messages array
            messages = [
                {"role": "system", "content": system_message},
                {"role": "assistant", "content": context_message},
                {"role": "user", "content": request.question}
            ]

            # Get response from OpenAI
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7
            )

            assistant_response = response.choices[0].message.content
            sql_query = self._extract_sql_query(assistant_response)
            graph_code = self._extract_graph_code(assistant_response)

            if not sql_query or not graph_code:
                send_error(channel_id, initial_response['ts'], {}, "Could not generate SQL query or graph code")
                return {"error": "Missing SQL query or graph code"}

            # Execute SQL query
            query_result, query_error = await self.sql_service.execute_query(sql_query)
            if query_error:
                send_error(channel_id, initial_response['ts'], {}, f"SQL Error: {query_error}")
                return {"error": query_error}

            # Generate and save graph
            img_buffer, graph_filename = generate_graph(graph_code, query_result)
            if not img_buffer:
                send_error(channel_id, initial_response['ts'], {}, "Failed to generate graph")
                return {"error": "Graph generation failed"}

            # Upload graph to Slack
            send_graph(channel_id, img_buffer, text)
            
            # Update final message
            update_final_message(channel_id, initial_response['ts'], {"text": "Graph generated successfully! ✨"})
            
            # After successful graph generation
            graph_url = mongodb_service.store_image(img_buffer, graph_filename, user_id)
            
            # Save to chat history
            save_chat_history(
                user_id=user_id,
                command="/graph",
                question=text,
                sql_query=sql_query,
                response="Graph generated successfully",
                ai_response=assistant_response,
                graph_url=graph_url
            )

            return {"success": True}

        except Exception as e:
            logger.error(f"Error in process_graph_command: {str(e)}")
            send_error(channel_id, initial_response['ts'], {}, str(e))
            return {"error": str(e)}

    def generate_and_save_graph(self, graph_code: str, sql_result: dict, user_id: str, question: str) -> Tuple[bytes, str]:
        """Generate graph and save to MongoDB"""
        try:
            # Generate graph
            img_buffer, graph_filename = generate_graph(graph_code, sql_result)
            if not img_buffer:
                raise ValueError("Failed to generate graph")

            # Store in MongoDB
            graph_url = self.mongodb_service.store_image(img_buffer, graph_filename, user_id)
            
            return img_buffer, graph_url
            
        except Exception as e:
            logger.error(f"Error generating graph: {e}")
            plt.close('all')
            return None, None

    async def process_sql_command(self, request: ChatRequest) -> Dict[str, Any]:
        """Process SQL command - return query without executing"""
        try:
            messages = self._prepare_conversation_context(request.user_id, request.question)
            
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7
            )

            sql_query = self._extract_sql_query(response.choices[0].message.content)
            if not sql_query:
                return {
                    "error": "Could not generate SQL query",
                    "sql_query": None
                }

            # Save to history
            save_chat_history(
                user_id=request.user_id,
                command="/sql",
                question=request.question,
                sql_query=sql_query,
                ai_response=response.choices[0].message.content
            )

            return {
                "sql_query": sql_query,
                "error": None
            }

        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            return {
                "error": str(e),
                "sql_query": None
            }

    async def process_ask_command(self, request: ChatRequest) -> ChatResponse:
        """Process ask command - execute query and return results"""
        try:
            # Load schema information
            with open("static/db_structure.json", "r") as f:
                db_structure = json.load(f)

            # 1. System prompt
            system_message = """You are a SQL expert. Generate SQL query to answer the question.
Format your response with the SQL query between ```sql markers.
Focus on generating accurate and efficient queries."""

            # 2. Assistant message with schema and history context
            context_message = "Database schema and tables:\n"
            context_message += f"{json.dumps(db_structure, indent=2)}\n\n"
            
            # Add history context
            history = load_chat_history(request.user_id)
            if history:
                context_message += "Previous conversations:\n"
                for chat in history[-5:]:  # Last 5 conversations
                    context_message += f"\nUser: {chat['question']}"
                    if chat.get('sql_query'):
                        context_message += f"\nGenerated SQL: ```sql\n{chat['sql_query']}\n```"

            # Prepare messages array
            messages = [
                {"role": "system", "content": system_message},
                {"role": "assistant", "content": context_message},
                {"role": "user", "content": request.question}
            ]

            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7
            )

            sql_query = self._extract_sql_query(response.choices[0].message.content)
            if not sql_query:
                return ChatResponse(
                    response="Could not generate SQL query",
                    error="No SQL query generated"
                )

            # Execute query
            query_result, error = await self.sql_service.execute_query(sql_query)
            if error:
                return ChatResponse(
                    response=f"Error executing query: {error}",
                    sql_query=sql_query,
                    error=error
                )

            # Save to history
            save_chat_history(
                user_id=request.user_id,
                command="/ask",
                question=request.question,
                sql_query=sql_query,
                ai_response=response.choices[0].message.content,
                response="Query executed successfully"
            )

            return ChatResponse(
                response="Query executed successfully",
                sql_query=sql_query,
                query_result=query_result
            )

        except Exception as e:
            logger.error(f"Error processing ask command: {e}")
            return ChatResponse(
                response=f"Error: {str(e)}",
                error=str(e)
            )

##################################################################################
##################################################################################
############################# Helper Functions ##################################
##################################################################################
##################################################################################

# Help text generator
def get_help_text():
    return """*Available Commands:*
• `/sql [question]` - Get SQL query for your question without executing it
• `/ask [question]` - Get results by executing SQL query (query won't be shown)
• `/graph [question]` - Generate visualization from query results
• `/help` - Show this help message

*Examples:*
• `/sql How many movies are there per country?`
• `/ask What are the top 10 countries by movie count?`
• `/graph Show me a bar chart of movies by country`"""

##################################################################################
##################################################################################
############################# Message Handlers ##################################
##################################################################################
##################################################################################

# Initial response sender
def send_initial_response(channel_id: str) -> dict:
    """Send initial 'processing' message"""
    return slack_client.chat_postMessage(
        channel=channel_id,
        text="Processing your request... 🧠",
        blocks=[{
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Processing your request... 🧠"
            }
        }]
    )

# Table update handler
def update_with_table(channel_id: str, ts: str, formatted_response: dict):
    """Update message with query results"""
    text = formatted_response.get("text", "No results")
    return slack_client.chat_update(
        channel=channel_id,
        ts=ts,
        text=text,
        blocks=[{
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": text
            }
        }]
    )

##################################################################################
##################################################################################
############################# Graph Handlers ####################################
##################################################################################
##################################################################################

# Graph code validator
def validate_graph_code(code: str) -> bool:
    """Basic validation of graph code for security"""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            # Prevent potentially dangerous operations
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if node.names[0].name not in ['matplotlib.pyplot', 'plt']:
                    return False
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile']:
                        return False
        return True
    except:
        return False

# Graph generator
def generate_graph(graph_code: str, sql_result: dict = None) -> Tuple[Optional[bytes], Optional[str]]:
    """Generate graph from code and return image buffer and filename"""
    try:
        # Clear any existing plots
        plt.clf()
        plt.close('all')
        
        # Create namespace for execution
        local_namespace = {'plt': plt}
        if sql_result:
            local_namespace['sql_result'] = sql_result
        
        # Execute graph code
        if not validate_graph_code(graph_code):
            raise ValueError("Invalid or potentially unsafe graph code")
            
        exec(graph_code, globals(), local_namespace)
        
        # Ensure the plot is fully rendered
        plt.tight_layout()
        
        # Generate unique filename
        graph_filename = f"{uuid.uuid4()}.png"
        
        # Save to memory buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        
        # Clean up
        plt.close('all')
        
        return img_buffer.getvalue(), graph_filename
        
    except Exception as e:
        logger.error(f"Graph generation error: {e}")
        plt.close('all')
        return None, None

##################################################################################
##################################################################################
############################# History Handlers ##################################
##################################################################################
##################################################################################

# Chat history saver
def save_chat_history(user_id: str, command: str, question: str, sql_query: str = None, 
                     response: str = None, ai_response: str = None, graph_url: str = None):
    """Save chat history to JSON file"""
    try:
        history_file = "static/chat_history.json"
        
        # Create history entry
        entry = {
            "user_id": user_id,
            "command": command,
            "question": question,
            "sql_query": sql_query,
            "response": response,
            "ai_response": ai_response,
            "graph_url": graph_url,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Load existing history or create new list
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = []
        except (FileNotFoundError, json.JSONDecodeError):
            history = []
        
        # Add new entry
        history.append(entry)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        
        # Save updated history
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")

# Chat history loader
def load_chat_history(user_id: str, limit: int = 5) -> list:
    """Load chat history from JSON file"""
    try:
        history_file = "static/chat_history.json"
        
        if not os.path.exists(history_file):
            return []
            
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
                if not isinstance(history, list):
                    return []
        except (json.JSONDecodeError, FileNotFoundError):
            return []
        
        # Filter by user_id and get last N entries
        user_history = [
            entry for entry in history 
            if entry['user_id'] == user_id
        ]
        return user_history[-limit:]
        
    except Exception as e:
        logger.error(f"Error loading chat history: {e}")
        return []

##################################################################################
##################################################################################
############################# Table Handlers ####################################
##################################################################################
##################################################################################

def format_query_result(query_result):
    """Format query result as a readable table for Slack"""
    if not query_result:
        return ""
    
    try:
        # Extract columns and rows
        columns = [str(col) for col in query_result['columns']]
        rows = query_result['rows']
        
        if not rows:
            return "No results found"
        
        # Calculate max width for each column
        col_widths = {col: len(col) for col in columns}
        for row in rows:
            for i, value in enumerate(row):
                col_widths[columns[i]] = max(col_widths[columns[i]], len(str(value)))
        
        # Build table
        table = []
        
        # Header
        header = "| " + " | ".join(f"{col:<{col_widths[col]}}" for col in columns) + " |"
        separator = "|" + "|".join("-" * (width + 2) for width in col_widths.values()) + "|"
        
        table.append(header)
        table.append(separator)
        
        # Rows
        for row in rows:
            row_str = "| " + " | ".join(f"{str(value):<{col_widths[columns[i]]}} " 
                          for i, value in enumerate(row)) + "|"
            table.append(row_str)
        
        return "```\n" + "\n".join(table) + "\n```"
        
    except Exception as e:
        logger.error(f"Error formatting table: {e}")
        return str(query_result)

def format_slack_message(response):
    """Format the response for Slack with support for large tables"""
    if not response.query_result:
        return {"type": "text", "text": "No results to display"}
    
    table_handler = TableHandler()
    
    # Analyze data dimensions
    num_rows, num_cols = table_handler.analyze_data(response.query_result)
    
    # Determine format based on data size
    if table_handler.should_use_excel(num_rows, num_cols):
        excel_result = table_handler.create_excel_file(response.query_result)
        if excel_result:
            excel_buffer, summary = excel_result
            return {
                "type": "excel",
                "excel_file": excel_buffer,
                "text": summary
            }
    
    # For smaller tables, use regular formatting
    formatted_table = format_query_result(response.query_result)
    return {
        "type": "table",
        "text": "*Results:*\n" + formatted_table
    }

def send_graph_status(channel_id: str, ts: str, formatted_response: dict):
    """Update message to show graph generation status"""
    text = formatted_response.get("text", "No results")
    return slack_client.chat_update(
        channel=channel_id,
        ts=ts,
        text=text,
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
                    "text": "Generating visualization... 📊"
                }
            }
        ]
    )

def send_graph(channel_id: str, img_buffer: bytes, text: str):
    """Upload graph to Slack"""
    return slack_client.files_upload_v2(
        channel=channel_id,
        file=img_buffer,
        filename="graph.png",
        title=f"Visualization for: {text}",
        initial_comment="Here's your visualization 📈"
    )

def update_final_message(channel_id: str, ts: str, formatted_response: dict):
    """Send final message after graph upload"""
    text = formatted_response.get("text", "No results")
    return slack_client.chat_update(
        channel=channel_id,
        ts=ts,
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

def send_error(channel_id: str, ts: str, formatted_response: dict, error: str):
    """Send error message"""
    text = formatted_response.get("text", "No results")
    return slack_client.chat_update(
        channel=channel_id,
        ts=ts,
        text=f"{text}\n\nError: {error} 😵",
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
                    "text": f"Error: {error} 😵"
                }
            }
        ]
    )

def send_excel_file(channel_id: str, excel_buffer: io.BytesIO, preview: str):
    """Send Excel file with preview"""
    # Send preview message
    slack_client.chat_postMessage(
        channel=channel_id,
        text=preview,
        blocks=[{
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": preview
            }
        }]
    )
    
    # Upload Excel file
    return slack_client.files_upload_v2(
        channel=channel_id,
        file=excel_buffer,
        filename="query_results.xlsx",
        title="Query Results",
        initial_comment="📊 Download complete results"
    )

def update_with_sql(channel_id: str, ts: str, sql_query: str):
    """Update message with SQL query"""
    return slack_client.chat_update(
        channel=channel_id,
        ts=ts,
        text=f"Here's your SQL query:\n```sql\n{sql_query}\n```",
        blocks=[{
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"Here's your SQL query:\n```sql\n{sql_query}\n```"
            }
        }]
    )

def update_with_help(channel_id: str, ts: str, help_text: str):
    """Update message with help text"""
    return slack_client.chat_update(
        channel=channel_id,
        ts=ts,
        text=help_text,
        blocks=[{
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": help_text
            }
        }]
    )

def update_with_processing(channel_id: str, ts: str, message: str):
    """Update message with processing status"""
    return slack_client.chat_update(
        channel=channel_id,
        ts=ts,
        text=message,
        blocks=[{
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": message
            }
        }]
    ) 



##################################################################################
##################################################################################
############################# Route Handlers ####################################
##################################################################################
##################################################################################

@router.post("/command")
async def slack_command(request: Request, db: Session = Depends(get_db)):
    """Handle Slack slash commands"""
    try:
        # Verify request
        form_data = await request.form()
        
        # Extract command info
        command = form_data.get("command", "").strip()
        text = form_data.get("text", "").strip()
        user_id = form_data.get("user_id")
        channel_id = form_data.get("channel_id")
        
        # Send initial response
        initial_response = send_initial_response(channel_id)
        
        # Initialize chat service
        chat_service = SlackChatService(db)
        
        # Handle different commands
        if command == "/help":
            help_text = get_help_text()
            update_with_help(channel_id, initial_response['ts'], help_text)
            return {"response_type": "in_channel"}
            
        elif command == "/sql":
            chat_request = ChatRequest(user_id=f"slack_{user_id}", question=text)
            response = await chat_service.sql_service.get_sql_only(chat_request)
            
            if response.get("error"):
                send_error(channel_id, initial_response['ts'], {}, response["error"])
            else:
                update_with_sql(channel_id, initial_response['ts'], response["sql_query"])
            
        elif command == "/ask":
            chat_request = ChatRequest(user_id=f"slack_{user_id}", question=text)
            response = await chat_service.process_ask_command(chat_request)
            formatted_response = format_slack_message(response)
            update_with_table(channel_id, initial_response['ts'], formatted_response)
            
        elif command == "/graph":
            chat_request = ChatRequest(user_id=f"slack_{user_id}", question=text)
            response = await chat_service.process_graph_command(
                chat_request, slack_client, mongodb_service, 
                text, channel_id, user_id, initial_response
            )
            if response:
                return {"response_type": "in_channel"}
        else:
            update_with_help(channel_id, initial_response['ts'], "Invalid command. Use /help to see available commands.")
        
        return {"response_type": "in_channel"}
        
    except Exception as e:
        logger.error(f"Error processing command: {e}", exc_info=True)
        return JSONResponse(
            status_code=200,  # Slack expects 200 even for errors
            content={
                "response_type": "ephemeral",
                "text": f"Sorry, I encountered an error: {str(e)} 😔"
            }
        )
