from slack_sdk import WebClient
from slack_sdk.signature import SignatureVerifier
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from config import settings
from services.chat_service import ChatService
from database import SessionLocal
from schemas import ChatRequest
import json
import logging
import os
import io
import matplotlib.pyplot as plt
from services.mongodb_service import MongoDBService
import time
import shutil
import pandas as pd
import io
from typing import Dict, Any, List, Tuple
import math
from .slack.message_handlers import (
    send_initial_response, update_with_table, 
    send_graph_status, send_graph, 
    update_final_message, send_error,
    update_with_sql, update_with_help
)
from .slack.formatters import format_slack_message


router = APIRouter()

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Initialize Slack clients
slack_client = WebClient(token=settings.SLACK_BOT_TOKEN)
signature_verifier = SignatureVerifier(settings.SLACK_SIGNING_SECRET)

# Initialize MongoDB service
mongodb_service = MongoDBService()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def generate_and_save_graph(graph_code: str, sql_result: dict) -> tuple:
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
        
        # Clean up
        plt.close('all')
        
        return img_buffer, local_path
    except Exception as e:
        logger.error(f"Error generating graph: {e}")
        plt.close('all')
        return None, None

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
        chat_service = ChatService(db)
        
        # Handle different commands
        if command == "/help":
            help_text = get_help_text()
            update_with_help(channel_id, initial_response['ts'], help_text)
            return {"response_type": "in_channel"}
            
        elif command == "/sql":
            chat_request = ChatRequest(user_id=f"slack_{user_id}", question=text)
            response = await chat_service.get_sql_only(chat_request)
            update_with_sql(channel_id, initial_response['ts'], response.sql_query)
            
        elif command == "/ask":
            chat_request = ChatRequest(user_id=f"slack_{user_id}", question=text)
            response = await chat_service.process_ask_command(chat_request)
            formatted_response = format_slack_message(response)
            update_with_table(channel_id, initial_response['ts'], formatted_response)
            
        elif command == "/graph":
            chat_request = ChatRequest(user_id=f"slack_{user_id}", question=text)
            responses = await chat_service.process_graph_command(chat_request, slack_client, mongodb_service, text,channel_id, user_id, initial_response)
            if responses:
                return 
            # if response.query_result:
            #     try:
            #         send_graph_status(channel_id, initial_response['ts'], {"text": "Generating visualization... 📊"})
                    
            #         # Generate and save graph
            #         img_buffer = io.BytesIO()
            #         plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            #         img_buffer.seek(0)
                    
            #         if img_buffer:
            #             # Upload graph
            #             send_graph(channel_id, img_buffer, text)
            #             update_final_message(channel_id, initial_response['ts'], {"text": "Visualization complete! ✨"})
            #     except Exception as e:
            #         logger.error(f"Error handling graph: {e}", exc_info=True)
            #         send_error(channel_id, initial_response['ts'], {"text": ""}, str(e))
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

@router.post("/events")
async def slack_events(request: Request):
    """Handle Slack events"""
    try:
        body = await request.json()
        
        # Handle URL verification
        if body.get("type") == "url_verification":
            return {"challenge": body.get("challenge")}
            
        return {"ok": True}
        
    except Exception as e:
        logger.error(f"Error handling event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
