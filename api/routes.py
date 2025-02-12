from fastapi import APIRouter, Depends, Request, BackgroundTasks
from sqlalchemy.orm import Session
from api.config import get_db
from api.handlers.slack_handlers import slack_command
from fastapi.responses import Response

router = APIRouter()

@router.post("/command")
async def handle_slack_command(request: Request, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Handle incoming Slack commands"""
    try:
        form_data = await request.form()
        response_url = form_data.get("response_url")
        
        background_tasks.add_task(
            slack_command,
            request,
            db,
            response_url
        )
        
        return Response(status_code=200)
        
    except Exception:
        return Response(status_code=200)
