from fastapi import APIRouter, Request, BackgroundTasks
from api.handlers.slack_handlers import slack_command
from api.handlers.feedback_handlers import save_positive_feedback, process_feedback
from fastapi.responses import Response
import json
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/command")
async def handle_slack_command(request: Request, background_tasks: BackgroundTasks):
    """Handle incoming Slack commands"""
    try:
        # Get form data
        form_data = await request.form()
        response_url = form_data.get("response_url")
        logger.info(f"[form_data] Received command request: {form_data}")
        
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

@router.post("/interactivity")
async def handle_interactivity(request: Request):
    """Handle Slack interactive components"""
    try:
        # Parse the request payload
        form_data = await request.form()
        logger.info(f"[form_data] Received command request: {form_data}")
        payload = json.loads(form_data.get("payload", "{}"))
        
        # Process feedback through centralized handler and await the result
        result = await process_feedback(payload)
        return result
            
    except Exception as e:
        logger.error(f"[INTERACTIVITY] Error handling interactivity: {e}")
        logger.exception("[INTERACTIVITY] Full traceback:")
        return {"text": "Error processing feedback"}
