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
    update_final_message, send_error, send_excel_file
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

@router.post("/events")
async def slack_events(request: Request, db: Session = Depends(get_db)):
    payload = await request.json()
    
    # Handle URL verification
    if payload.get("type") == "url_verification":
        return JSONResponse(content={"challenge": payload.get("challenge")})
    
    # Verify request signature
    body = await request.body()
    if not signature_verifier.is_valid_request(body, dict(request.headers)):
        raise HTTPException(status_code=400, detail="Invalid Slack request")
    
    event = payload.get("event", {})
    
    # Handle app mention events
    if event.get("type") == "app_mention":
        try:
            channel_id = event.get("channel")
            user_id = event.get("user")
            text = event.get("text").split(">")[-1].strip()
            
            # Send thinking indicator
            slack_client.chat_postMessage(
                channel=channel_id,
                text="Thinking... 🤔"
            )
            
            # Process commands
            if text.startswith("/help"):
                return slack_client.chat_postMessage(
                    channel=channel_id,
                    text=get_help_text(),
                    mrkdwn=True
                )
            
            # Process the request
            chat_service = ChatService(db)
            enhanced_question = text
            chat_request = ChatRequest(user_id=f"slack_{user_id}", question=enhanced_question)
            response = await chat_service.process_chat(chat_request)
            
            # Send initial response
            formatted_response = format_slack_message(response)
            initial_message = slack_client.chat_postMessage(
                channel=channel_id,
                text=formatted_response,
                mrkdwn=True
            )
            
            # Handle graph if present
            if response.graph_url and hasattr(response, 'query_result'):
                try:
                    graph_code = chat_service._extract_graph_code(response.response)
                    if graph_code:
                        img_buffer, local_path = generate_and_save_graph(
                            graph_code, 
                            response.query_result
                        )
                        
                        if img_buffer:
                            # Upload to Slack
                            slack_client.files_upload_v2(
                                channel=channel_id,
                                file=img_buffer,
                                filename="graph.png",
                                title=f"Visualization for: {text}",
                                initial_comment="Here's your visualization 📊"
                            )
                            
                            # Store in MongoDB
                            image_url = mongodb_service.store_image(
                                img_buffer.getvalue(),
                                f"graph_{int(time.time())}.png",
                                f"slack_{user_id}"
                            )
                            
                            # Store chat history
                            mongodb_service.store_chat_history(
                                user_id=f"slack_{user_id}",
                                question=enhanced_question,
                                sql_query=response.sql_query,
                                response=response.response,
                                graph_urls=[image_url]
                            )
                except Exception as e:
                    logger.error(f"Error handling graph: {e}", exc_info=True)
                    slack_client.chat_postMessage(
                        channel=channel_id,
                        text=f"Error generating graph: {str(e)}"
                    )
            
        except Exception as e:
            slack_client.chat_postMessage(
                channel=channel_id,
                text=f"Sorry, I encountered an error: {str(e)}"
            )
            raise HTTPException(status_code=500, detail=str(e))
    
    return JSONResponse(content={"status": "ok"})

async def process_slack_command(text: str, user_id: str, channel_id: str, db: Session):
    """Process Slack command and handle responses"""
    try:
        # Send initial response
        initial_response = send_initial_response(channel_id)
        
        # Process chat request
        chat_service = ChatService(db)
        chat_request = ChatRequest(user_id=f"slack_{user_id}", question=text)
        response = await chat_service.process_chat(chat_request)
        
        # Format response
        formatted_response = format_slack_message(response)
        
        if formatted_response["type"] == "excel":
            # Send Excel file with preview
            send_excel_file(
                channel_id, 
                formatted_response["excel_file"],
                formatted_response["preview"]
            )
            
            # Update message with preview table
            update_with_table(
                channel_id,
                initial_response['ts'],
                {"text": formatted_response["preview_table"]}
            )
        else:
            # Update with regular table
            update_with_table(
                channel_id,
                initial_response['ts'],
                formatted_response
            )
        
        # Handle graph if present
        if response.graph_url and hasattr(response, 'query_result'):
            try:
                graph_code = chat_service._extract_graph_code(response.response)
                if graph_code:
                    # Update status
                    send_graph_status(channel_id, initial_response['ts'], formatted_response)
                    
                    # Generate and save graph
                    img_buffer, local_path = generate_and_save_graph(
                        graph_code, 
                        response.query_result
                    )
                    
                    if img_buffer:
                        # Upload graph
                        send_graph(channel_id, img_buffer, text)
                        
                        # Store in MongoDB
                        image_url = mongodb_service.store_image(
                            img_buffer.getvalue(),
                            f"graph_{int(time.time())}.png",
                            f"slack_{user_id}"
                        )
                        
                        # Store chat history
                        mongodb_service.store_chat_history(
                            user_id=f"slack_{user_id}",
                            question=text,
                            sql_query=response.sql_query,
                            response=response.response,
                            graph_urls=[image_url]
                        )
                        
                        # Final update
                        update_final_message(channel_id, initial_response['ts'], formatted_response)
                    
            except Exception as e:
                logger.error(f"Error handling graph: {e}", exc_info=True)
                send_error(channel_id, initial_response['ts'], formatted_response, str(e))
        
        return {"response_type": "in_channel"}
        
    except Exception as e:
        logger.error(f"Error processing command: {e}", exc_info=True)
        return {
            "response_type": "ephemeral",
            "text": f"Sorry, I encountered an error: {str(e)} 😔"
        }

@router.post("/commands")
async def slack_commands(request: Request, db: Session = Depends(get_db)):
    """Handle Slack slash commands"""
    form_data = await request.form()
    command = form_data.get("command", "").strip()
    text = form_data.get("text", "").strip()
    user_id = form_data.get("user_id")
    channel_id = form_data.get("channel_id")
    
    return await process_slack_command(text, user_id, channel_id, db)

def get_help_text():
    return """*Available Commands:*
• `/help` - Show this help message
• `/ask [question]` - Generate SQL query and graph for your question
• `/summarise [data]` - Summarize data or graph results
• `/improve [code]` - Improve or fix graph code
• Just mention me (@bot) with your question!"""