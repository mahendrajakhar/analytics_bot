import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from api.config import logger

def save_chat_history(user_id: str, command: str, question: str, sql_query: Optional[str] = None, 
                     response: Optional[str] = None, ai_response: Optional[str] = None, graph_url: Optional[str] = None):
    """Save chat history to centralized JSON file"""
    try:
        history_file = "static/chat_history.json"
        os.makedirs("static", exist_ok=True)
        
        # Load existing history
        history = {}
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
            except json.JSONDecodeError:
                history = {}
        
        # Initialize user history if not exists
        if user_id not in history:
            history[user_id] = []
        
        # Add new entry
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "command": command,
            "question": question,
            "sql_query": sql_query,
            "response": response,
            "ai_response": ai_response,
        }
        
        # Only add graph_url if it exists
        if graph_url:
            entry["graph_url"] = graph_url
        
        # Add to user's history
        history[user_id].append(entry)
        
        # Keep only last 50 entries per user
        history[user_id] = history[user_id][-20:]
        
        # Save updated history
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
            
    except Exception:
        pass

def load_chat_history(user_id: str, limit: int = 5) -> List[Dict]:
    """Load chat history from centralized JSON file"""
    try:
        history_file = "static/chat_history.json"
        
        if not os.path.exists(history_file):
            return []
        
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
                
            # Get user's history
            user_history = history.get(user_id, [])
            
            # Return the most recent entries up to the limit
            return user_history[-limit:]
            
        except json.JSONDecodeError:
            return []
            
    except Exception:
        return []
