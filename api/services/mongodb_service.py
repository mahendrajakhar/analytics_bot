from pymongo import MongoClient
from bson.binary import Binary
from datetime import datetime
from api.config import settings, logger
from bson.objectid import ObjectId

class MongoDBService:
    def __init__(self):
        self.uri = settings.MONGODB_URI
        self.client = MongoClient(self.uri)
        self.db = self.client.analytics_bot
        self.images = self.db.images
        self.history = self.db.chat_history

    def save_graph(self, graph_doc: dict) -> str:
        """Save graph to MongoDB and return its ID"""
        try:
            # Convert image data to Binary
            if 'image_data' in graph_doc:
                graph_doc['image_data'] = Binary(graph_doc['image_data'])
            
            # Add timestamp if not present
            if 'created_at' not in graph_doc:
                graph_doc['created_at'] = datetime.utcnow()
                
            result = self.images.insert_one(graph_doc)
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error saving graph to MongoDB: {e}")
            return None

    def get_graph(self, graph_id: str):
        """Retrieve graph from MongoDB by ID"""
        try:
            return self.images.find_one({"_id": ObjectId(graph_id)})
        except Exception as e:
            logger.error(f"Error retrieving graph from MongoDB: {e}")
            return None

    def store_image(self, image_data: bytes, filename: str, user_id: str) -> str:
        """Store image in MongoDB and return its ID"""
        try:
            image_binary = Binary(image_data)
            image_doc = {
                "filename": filename,
                "image_data": image_binary,
                "user_id": user_id,
                "created_at": datetime.utcnow(),
                "content_type": "image/png"
            }
            result = self.images.insert_one(image_doc)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error storing image in MongoDB: {e}")
            return None

    def get_image(self, image_id: str):
        """Retrieve image from MongoDB by ID"""
        try:
            return self.images.find_one({"_id": ObjectId(image_id)})
        except Exception as e:
            logger.error(f"Error retrieving image from MongoDB: {e}")
            return None

# Initialize service
mongodb_service = MongoDBService()
