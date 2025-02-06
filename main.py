# main.py
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from api import slack_router, ui_router
from services.mongodb_service import MongoDBService
from bson.objectid import ObjectId
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend to prevent local display

# Initialize FastAPI app
app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Initialize MongoDB service
mongodb_service = MongoDBService()

# Initialize UI router with templates
ui_router.init_router(templates)

# Include routers
app.include_router(ui_router.router)
app.include_router(slack_router.router, prefix="/slack")

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
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
