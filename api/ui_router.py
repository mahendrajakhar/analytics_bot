from fastapi import APIRouter, Request, Form, Body, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from services.chat_service import ChatService
from database import SessionLocal
from schemas import ChatRequest
import json
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize templates
templates = None

def init_router(template_engine: Jinja2Templates):
    """Initialize router with template engine"""
    global templates
    templates = template_engine

def format_query_result(query_result):
    """Format query result as a readable table"""
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
        
        return "\n".join(table)
        
    except Exception as e:
        logger.error(f"\n\n\nError formatting table: {e}")
        return str(query_result)

def format_query_result_html(query_result):
    """Format query result as an HTML table"""
    if not query_result:
        return ""
    
    try:
        # Extract columns and rows
        columns = [str(col) for col in query_result['columns']]
        rows = query_result['rows']
        
        if not rows:
            return "<p>No results found</p>"
        
        # Build HTML table
        html = ['<table class="query-result-table">']
        
        # Header
        html.append('<thead>')
        html.append('<tr>')
        for col in columns:
            html.append(f'<th>{col}</th>')
        html.append('</tr>')
        html.append('</thead>')
        
        # Body
        html.append('<tbody>')
        for row in rows:
            html.append('<tr>')
            for value in row:
                html.append(f'<td>{str(value)}</td>')
            html.append('</tr>')
        html.append('</tbody>')
        
        html.append('</table>')
        
        return '\n'.join(html)
        
    except Exception as e:
        logger.error(f"Error formatting table: {e}")
        return f"<p>Error formatting results: {str(e)}</p>"

# Load database structure and schema from JSON file
def get_db_structure():
    try:
        with open("static/db_structure.json", "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading DB structure: {e}")
        return {}

db_structure = get_db_structure()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Home page with chat UI"""
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "db_structure": db_structure
    })

@router.post("/api/chat", response_class=JSONResponse)
async def process_command(
    request: Request, 
    body: dict = Body(...),
    db: Session = Depends(get_db)
):
    """Process chat commands from UI"""
    query = body.get("query", "")
    user_id = body.get("user_id", "default_user")

    # Initialize chat service
    chat_service = ChatService(db)

    # Process commands
    if query.startswith("/"):
        command, *rest = query.split(maxsplit=1)
        question = rest[0] if rest else ""

        if command == "/help":
            return {"result": get_help_html()}
        elif command in ["/ask", "/summarise", "/improve"]:
            context_map = {
                "/ask": "Generate a SQL query and optionally a graph for this question: ",
                "/summarise": "Summarize this data: ",
                "/improve": "Improve this graph code: "
            }
            enhanced_question = context_map[command] + question
    else:
        enhanced_question = "Generate a SQL query for this question: " + query

    # Process the chat request
    chat_request = ChatRequest(user_id=user_id, question=enhanced_question)
    response = await chat_service.process_chat(chat_request)

    # Format the response
    formatted_response = ""
    
    # Add query results if present
    if response.query_result:
        formatted_response = format_query_result_html(response.query_result)

    return {
        "result": formatted_response,
        "graph_url": str(response.graph_url) if response.graph_url else None,
        "error": str(response.error) if response.error else None
    }

def get_help_html():
    """Return help text in HTML format"""
    return """<h2>Available Commands</h2>
    <ul>
        <li><strong>/help</strong> - Show help information.</li>
        <li><strong>/ask</strong> - Generate SQL query and/or graph code from your question.</li>
        <li><strong>/summarise</strong> - Summarise the graph or data.</li>
        <li><strong>/improve</strong> - Improve or repair the graph code.</li>
    </ul>""" 