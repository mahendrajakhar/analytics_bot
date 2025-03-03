import json
import os
from datetime import datetime
from typing import Dict, Any
from api.config import logger, feedback_langchain_service
from api.handlers.history_handlers import load_chat_history

FEEDBACK_DIR = "static/feedback"
os.makedirs(FEEDBACK_DIR, exist_ok=True)

async def process_feedback(payload: Dict[str, Any]) -> Dict[str, str]:
    """Process feedback from Slack interactive components"""
    try:
        
        # Get action details
        action_id = payload.get("actions", [{}])[0].get("action_id")
        
        if action_id == "feedback_helpful":
            # Get message details from payload
            message = payload.get("message", {})
            user = payload.get("user", {})
            message_ts = message.get("ts", "")
            
            # Load chat history to determine command type
            history = load_chat_history(user.get("id"))
            history_entry = None
            
            # Find the matching entry in history using message_ts
            for entry in reversed(history):
                if entry.get("message_ts") == message_ts:
                    history_entry = entry
                    break
            
            if not history_entry:
                return {"text": "Could not process feedback. History entry not found."}
            
            # Get command type from history entry
            command_type = history_entry.get("command")
            if not command_type:
                return {"text": "Could not process feedback. Command type not found."}
            
            # Prepare feedback data
            feedback_data = {
                "command_type": command_type,
                "user_id": user.get("id"),
                "user_name": user.get("name", ""),
                "message_ts": message_ts
            }
            
            # Save feedback
            save_result = await save_positive_feedback(feedback_data)
            logger.info(f"[FEEDBACK] Save result: {save_result}")
            
            return {"text": "Thanks for your feedback! 🙌"}
            
        elif action_id == "feedback_not_helpful":
            logger.info("[FEEDBACK] Received negative feedback")
            return {"text": "Thanks for your feedback. We'll work on improving! 🎯"}
            
        else:
            logger.error(f"[FEEDBACK] Unknown action_id: {action_id}")
            return {"text": "Invalid feedback action"}
            
    except Exception as e:
        logger.error(f"[FEEDBACK] Error processing feedback: {e}")
        logger.exception("[FEEDBACK] Full traceback:")
        return {"text": "Error processing feedback"}

async def save_positive_feedback(feedback_data: Dict[str, Any]) -> bool:
    """Save positive feedback to appropriate JSON file based on command type"""
    try:
        logger.info("[FEEDBACK] Starting to save positive feedback")
        logger.info(f"[FEEDBACK] Received feedback data: {json.dumps(feedback_data, indent=2)}")
        
        # Validate required fields
        command_type = feedback_data.get("command_type")
        message_ts = feedback_data.get("message_ts")
        user_id = feedback_data.get("user_id")
        user_name = feedback_data.get("user_name", "")
        
        if not command_type or not message_ts or not user_id:
            logger.error("[FEEDBACK] Missing required feedback data")
            logger.error(f"[FEEDBACK] command_type: {command_type}, message_ts: {message_ts}, user_id: {user_id}")
            return False

        # Get message from chat history
        logger.info(f"[FEEDBACK] Loading chat history for user: {user_id}")
        history = load_chat_history(user_id)
        history_entry = None
        
        # Find the matching entry in history
        logger.info(f"[FEEDBACK] Searching for message with ts: {message_ts}")
        for entry in reversed(history):
            if entry.get("message_ts") == message_ts:
                history_entry = entry
                logger.info("[FEEDBACK] Found matching history entry")
                logger.debug(f"[FEEDBACK] History entry: {json.dumps(entry, indent=2)}")
                break
        
        if not history_entry:
            logger.error(f"[FEEDBACK] No history entry found for message_ts: {message_ts}")
            return False

        # Determine file path based on command type
        filename = f"positive_{command_type.replace('/', '')}_feedback.json"
        filepath = os.path.join(FEEDBACK_DIR, filename)
        logger.info(f"[FEEDBACK] Using feedback file: {filepath}")

        # Prepare base feedback entry with common fields
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "user_name": user_name,
            "question": history_entry.get("question", ""),
            "sql_query": history_entry.get("sql_query", "")
        }

        # Add command-specific data
        if command_type == "/graph":
            logger.info("[FEEDBACK] Processing graph command feedback")
            # Extract visualization code from AI response
            ai_response = history_entry.get("ai_response", "")
            if "```python" in ai_response:
                code_parts = ai_response.split("```python")
                if len(code_parts) > 1:
                    viz_code = code_parts[1].split("```")[0].strip()
                    entry.update({
                        "visualization_code": viz_code,
                        "graph_url": history_entry.get("graph_url", "")
                    })
                    logger.info("[FEEDBACK] Added visualization code and graph URL to feedback")
            
        elif command_type == "/ask":
            logger.info("[FEEDBACK] Processing ask command feedback")
            entry.update({
                "response": history_entry.get("response", ""),
                "ai_response": history_entry.get("ai_response", "")
            })
            
        elif command_type == "/sql":
            logger.info("[FEEDBACK] Processing sql command feedback")
            entry.update({
                "ai_response": history_entry.get("ai_response", "")
            })

        logger.info(f"[FEEDBACK] Prepared feedback entry: {json.dumps(entry, indent=2)}")

        # Load existing feedback
        existing_feedback = []
        if os.path.exists(filepath):
            logger.info("[FEEDBACK] Loading existing feedback file")
            with open(filepath, 'r') as f:
                existing_feedback = json.load(f)
                logger.info(f"[FEEDBACK] Loaded {len(existing_feedback)} existing entries")

        # Add new feedback
        existing_feedback.append(entry)
        logger.info("[FEEDBACK] Added new entry to feedback list")

        # Save updated feedback
        logger.info("[FEEDBACK] Saving updated feedback file")
        with open(filepath, 'w') as f:
            json.dump(existing_feedback, f, indent=2)

        # Add to vector index
        try:
            await feedback_langchain_service.add_feedback_to_index(entry)
            logger.info("[FEEDBACK] Added feedback to vector index")
        except Exception as e:
            logger.error(f"[FEEDBACK] Error adding to vector index: {e}")

        return True

    except Exception as e:
        logger.error(f"[FEEDBACK] Error saving feedback: {e}")
        logger.exception("[FEEDBACK] Full traceback:")
        return False
