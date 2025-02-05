# main.py
import os
import json
import logging
import uvicorn
from fastapi import FastAPI, Request, Form, Body, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from services.chat_service import ChatService
from database import SessionLocal
from schemas import ChatRequest, ChatResponse

# Initialize FastAPI app
app = FastAPI()

# Mount the static directory (for CSS, JSON data, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Set up logging (optional)
logging.basicConfig(level=logging.INFO)

# Load database structure and schema from JSON file
db_info_path = os.path.join("static", "db_structure.json")
with open(db_info_path, "r") as f:
    db_structure = json.load(f)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -----------------------------------------------------------------------------
# Dummy AI Model Call Function
# Replace this with your actual GPT-4o-mini integration call.
# -----------------------------------------------------------------------------
def call_ai_model(prompt: str) -> str:
    logging.info(f"Calling AI Model with prompt: {prompt}")
    # Here you would integrate your actual LLM inference call.
    # For demonstration, we return a dummy response.
    return f"Generated output for prompt:\n\n{prompt}"

# -----------------------------------------------------------------------------
# Route: Home Page (Chat UI)
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Pass the loaded DB structure to the UI if needed.
    return templates.TemplateResponse("index.html", {"request": request, "db_structure": db_structure})

# -----------------------------------------------------------------------------
# Route: Help Page
# -----------------------------------------------------------------------------
@app.get("/help", response_class=HTMLResponse)
async def help_page(request: Request):
    help_text = """
    <h2>Available Commands</h2>
    <ul>
        <li><strong>/help</strong> - Show help information.</li>
        <li><strong>/ask</strong> - Generate SQL query and/or graph code from your question.</li>
        <li><strong>/summarise</strong> - Summarise the graph or data.</li>
        <li><strong>/improve</strong> - Improve or repair the graph code.</li>
    </ul>
    """
    return templates.TemplateResponse("help.html", {"request": request, "help_text": help_text})

# -----------------------------------------------------------------------------
# Route: /ask Command
# This endpoint handles user queries for SQL and/or graph code generation.
# -----------------------------------------------------------------------------
@app.post("/ask", response_class=JSONResponse)
async def ask_command(request: Request, query: str = Form(...)):
    # Build a prompt that includes the user query and DB information.
    prompt = (
        f"User query: {query}\n\n"
        f"Database Structure: {json.dumps(db_structure.get('structure', {}), indent=2)}\n\n"
        f"Database Schema: {json.dumps(db_structure.get('schema', {}), indent=2)}\n\n"
        "Based on the above, generate a SQL query and, if needed, graph code using a specified library. " # need to specify library
        "The output should be formatted so that the code sections can be parsed and executed."
    )
    logging.info(f"Processing 'prompt': {prompt}")
    output = call_ai_model(prompt)
    return {"result": output}

# -----------------------------------------------------------------------------
# Route: /summarise Command
# This endpoint handles requests for summarising graph or data output.
# -----------------------------------------------------------------------------
@app.post("/summarise", response_class=JSONResponse)
async def summarise_command(request: Request, query: str = Form(...)):
    prompt = (
        f"User request for summarisation: {query}\n\n"
        f"Database Info: {json.dumps(db_structure, indent=2)}\n\n"
        "Please provide a summary explanation of the graph or data, "
        "including key insights and possible recommendations."
    )
    output = call_ai_model(prompt)
    return {"result": output}

# -----------------------------------------------------------------------------
# Route: /improve Command
# This endpoint handles requests to improve or repair existing graph code.
# -----------------------------------------------------------------------------
@app.post("/improve", response_class=JSONResponse)
async def improve_command(request: Request, query: str = Form(...)):
    prompt = (
        f"User request to improve graph code: {query}\n\n"
        f"Database Structure: {json.dumps(db_structure, indent=2)}\n\n"
        "Please provide improved graph code (and necessary SQL query if required) "
        "using the appropriate libraries."
    )
    output = call_ai_model(prompt)
    return {"result": output}

# -----------------------------------------------------------------------------
# Generic Command Processing Endpoint
# This endpoint processes any command by checking the 'command' field.
# -----------------------------------------------------------------------------
@app.post("/api/chat", response_class=JSONResponse)
async def process_command(
    request: Request, 
    body: dict = Body(...),
    db: Session = Depends(get_db)
):
    query = body.get("query", "")
    user_id = body.get("user_id", "default_user")

    # Initialize chat service
    chat_service = ChatService(db)

    # Process commands
    if query.startswith("/"):
        command, *rest = query.split(maxsplit=1)
        question = rest[0] if rest else ""

        if command == "/help":
            return {"result": return_help_html()}
        elif command in ["/ask", "/summarise", "/improve"]:
            # Add command context to the question
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
    logging.debug(f"Final response object: {response}")
    # Format the response for the frontend
    result = {
        "result":       str(response.response),
        "sql_query":    str(response.sql_query),
        "graph_url":    str(response.graph_url),
        "query_result": str(response.query_result),
        "error":        str(response.error)
    }

    return result

# -----------------------------------------------------------------------------
# Return Help HTML
# -----------------------------------------------------------------------------
def return_help_html():
    help_text = """<h2>Available Commands</h2>
    <ul>
        <li><strong>/help</strong> - Show help information.</li>
        <li><strong>/ask</strong> - Generate SQL query and/or graph code from your question.</li>
        <li><strong>/summarise</strong> - Summarise the graph or data.</li>
        <li><strong>/improve</strong> - Improve or repair the graph code.</li>
    </ul>"""
    return help_text

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Run with: uvicorn main:app --reload
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
