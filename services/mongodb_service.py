from pymongo import MongoClient
from bson.binary import Binary
from bson.objectid import ObjectId
import io
import logging
from datetime import datetime
from config import settings

logger = logging.getLogger(__name__)

class MongoDBService:
    def __init__(self):
        self.uri = settings.MONGODB_URI
        self.client = MongoClient(self.uri)
        self.db = self.client.analytics_bot
        self.images = self.db.images
        self.history = self.db.chat_history

    def store_image(self, image_data: bytes, filename: str, user_id: str) -> str:
        """Store image in MongoDB and return its URL"""
        try:
            logger.info("\n\n\nStarting MongoDB image storage process...")
            
            # Convert image to Binary
            logger.info("\n\n\nConverting image to Binary format...")
            image_binary = Binary(image_data)
            
            # Create image document
            logger.info("\n\n\nCreating image document...")
            image_doc = {
                "filename": filename,
                "data": image_binary,
                "user_id": user_id,
                "created_at": datetime.utcnow(),
                "content_type": "image/png"
            }
            
            # Insert into MongoDB
            logger.info("\n\n\nInserting document into MongoDB...")
            result = self.images.insert_one(image_doc)
            
            # Generate URL
            image_url = f"/api/images/{result.inserted_id}"
            logger.info(f"\n\n\nImage stored successfully. Generated URL: {image_url}")
            
            return image_url

        except Exception as e:
            logger.error(f"\n\n\nError storing image in MongoDB: {e}")
            raise

    def get_image(self, image_id: str):
        """Retrieve image from MongoDB"""
        try:
            logger.info(f"\n\n\nAttempting to retrieve image {image_id} from MongoDB...")
            image_doc = self.images.find_one({"_id": image_id})
            if image_doc:
                logger.info("\n\n\nImage retrieved successfully")
                return image_doc["data"], image_doc["content_type"]
            logger.warning("\n\n\nImage not found in MongoDB")
            return None, None
        except Exception as e:
            logger.error(f"\n\n\nError retrieving image from MongoDB: {e}")
            raise

    def store_chat_history(self, user_id: str, question: str, sql_query: str, 
                          response: str, graph_urls: list = None):
        """Store chat history with graph URLs"""
        try:
            history_doc = {
                "user_id": user_id,
                "question": question,
                "sql_query": sql_query,
                "response": response,
                "graph_urls": graph_urls or [],
                "timestamp": datetime.utcnow()
            }
            
            self.history.insert_one(history_doc)
        except Exception as e:
            logger.error(f"Error storing chat history: {e}")
            raise 