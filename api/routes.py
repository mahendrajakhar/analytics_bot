from fastapi import APIRouter, Request, BackgroundTasks
from api.handlers.slack_handlers import slack_command
from fastapi.responses import Response
import logging

router = APIRouter()

@router.post("/command")
async def handle_slack_command(request: Request, background_tasks: BackgroundTasks):
    """Handle incoming Slack commands"""
    try:
        # Get form data
        form_data = await request.form()
        response_url = form_data.get("response_url")
        
        # Add command processing to background tasks
        background_tasks.add_task(
            slack_command,
            request,
            response_url
        )
        
        # Return empty response with 200 status
        return Response(status_code=200)
        
    except Exception:
        return Response(status_code=200)
