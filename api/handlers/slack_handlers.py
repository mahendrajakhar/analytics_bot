import logging
from fastapi import Request, HTTPException
from api.config import settings, logger  # Remove get_db import
from api.services.slack_chat_service import SlackChatService
from api.services.sql_service import SQLService
from api.services.mongodb_service import mongodb_service
from api.utils.message_utils import (
    send_initial_response, update_final_message, send_error, 
    get_help_text, update_with_processing
)

# Define command constants
COMMAND_HELP = "/help"
COMMAND_SQL = "/sql"
COMMAND_ASK = "/ask"
COMMAND_GRAPH = "/graph"

async def slack_command(request: Request, response_url: str):
    """Handle Slack slash commands"""
    try:
        form_data = await request.form()
        
        command = form_data.get("command", "").strip()
        text = form_data.get("text", "").strip()
        user_id = form_data.get("user_id", "")
        channel_id = form_data.get("channel_id", "")
        
        initial_response = send_initial_response(channel_id)
        message_ts = initial_response["ts"]
        
        update_with_processing(
            channel_id, 
            message_ts, 
            "Processing your request... 🔄"
        )
        
        sql_service = SQLService()  # Remove db dependency
        chat_service = SlackChatService(sql_service, mongodb_service)
        
        if command == COMMAND_HELP or not text:
            help_text = get_help_text()
            update_final_message(channel_id, message_ts, {"text": help_text})
            return {"text": help_text}
            
        elif command == COMMAND_SQL:
            update_with_processing(channel_id, message_ts, "Generating SQL query... 🤔")
            return await chat_service.process_sql_command(
                {"user_id": user_id, "question": text},
                channel_id,
                message_ts
            )
            
        elif command == COMMAND_ASK:
            update_with_processing(channel_id, message_ts, "Analyzing your question... 🤔")
            return await chat_service.process_ask_command(
                {"user_id": user_id, "question": text},
                channel_id,
                message_ts
            )
            
        elif command == COMMAND_GRAPH:
            update_with_processing(channel_id, message_ts, "Generating visualization... 📊")
            return await chat_service.process_graph_command(
                {"user_id": user_id, "question": text},
                channel_id,
                message_ts
            )
            
        else:
            error_msg = f"Unknown command: {command}"
            send_error(channel_id, message_ts, {}, error_msg)
            return {"error": error_msg}
            
    except Exception as e:
        return {"error": str(e)}
