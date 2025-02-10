# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from api import slack_router
from api.slack_router import MongoDBService, settings
from bson.objectid import ObjectId
import matplotlib
import logging

# Configure logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("fastapi").setLevel(logging.WARNING)

# Set basic logging configuration
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

matplotlib.use("Agg")

# Initialize FastAPI app
app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize MongoDB service
mongodb_service = MongoDBService()

# Include routers
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
