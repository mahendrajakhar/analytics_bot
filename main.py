# main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from bson.objectid import ObjectId
import matplotlib
import json

from api.config import settings, logger
from api.services.mongodb_service import mongodb_service
from api.routes import router as api_router

# Use non-interactive backend for matplotlib
matplotlib.use("Agg")

# Initialize FastAPI app
app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(api_router, prefix="/slack")

# Image serving endpoint
@app.get("/api/images/{image_id}")
async def get_image(image_id: str):
    """Serve images from MongoDB"""
    try:
        image_data, content_type = mongodb_service.get_image(ObjectId(image_id))
        if image_data:
            return Response(content=image_data, media_type=content_type)
        raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        logger.error(f"Error serving image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/slack/events")
async def slack_events(request: Request):
    data = await request.json()
    
    # If Slack sends a challenge, respond with it
    if "challenge" in data:
        return {"challenge": data["challenge"]}
    
    # Handle other event types here
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
