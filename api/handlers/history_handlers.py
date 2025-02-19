import json
import os
from datetime import datetime
from typing import List, Dict
from api.config import get_chat_history_path, logger

def save_chat_history(user_id: str, command: str, question: str, 
                     message_ts: str, sql_query: str = None, 
                     response: str = None, ai_response: str = None, 
                     graph_url: str = None) -> None:
    """Save chat history to JSON file"""
    try:
        history_file = get_chat_history_path(user_id)
        
        # Load existing history
        history = []
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        
        # Add new entry
        history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'message_ts': message_ts,
            'command': command,
            'question': question,
            'sql_query': sql_query,
            'response': response,
            'ai_response': ai_response,
            'graph_url': graph_url
        })
        
        # Keep only last 100 entries
        history = history[-100:]
        
        # Save updated history
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")

def load_chat_history(user_id: str) -> List[Dict]:
    """Load chat history from JSON file"""
    try:
        history_file = get_chat_history_path(user_id)
        if not os.path.exists(history_file):
            return []
            
        with open(history_file, 'r') as f:
            return json.load(f)
            
    except Exception as e:
        logger.error(f"Error loading chat history: {e}")
        return []
