from slack_sdk import WebClient
import logging
from config import settings
import io

logger = logging.getLogger(__name__)
slack_client = WebClient(token=settings.SLACK_BOT_TOKEN)

def send_initial_response(channel_id: str) -> dict:
    """Send initial 'processing' message"""
    return slack_client.chat_postMessage(
        channel=channel_id,
        text="Processing your request... 🧠",
        blocks=[{
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Processing your request... 🧠"
            }
        }]
    )

def update_with_table(channel_id: str, ts: str, formatted_response: dict):
    """Update message with query results"""
    text = formatted_response.get("text", "No results")
    return slack_client.chat_update(
        channel=channel_id,
        ts=ts,
        text=text,
        blocks=[{
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": text
            }
        }]
    )

def send_graph_status(channel_id: str, ts: str, formatted_response: dict):
    """Update message to show graph generation status"""
    text = formatted_response.get("text", "No results")
    return slack_client.chat_update(
        channel=channel_id,
        ts=ts,
        text=f"{text}\n\nGenerating visualization... 📊",
        blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": text
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "Generating visualization... 📊"
                }
            }
        ]
    )

def send_graph(channel_id: str, img_buffer: bytes, text: str):
    """Upload graph to Slack"""
    return slack_client.files_upload_v2(
        channel=channel_id,
        file=img_buffer,
        filename="graph.png",
        title=f"Visualization for: {text}",
        initial_comment="Here's your visualization 📈"
    )

def update_final_message(channel_id: str, ts: str, formatted_response: dict):
    """Send final message after graph upload"""
    text = formatted_response.get("text", "No results")
    return slack_client.chat_update(
        channel=channel_id,
        ts=ts,
        text=f"{text}\n\nVisualization complete! ✨",
        blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": text
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "Visualization complete! ✨"
                }
            }
        ]
    )

def send_error(channel_id: str, ts: str, formatted_response: dict, error: str):
    """Send error message"""
    text = formatted_response.get("text", "No results")
    return slack_client.chat_update(
        channel=channel_id,
        ts=ts,
        text=f"{text}\n\nError: {error} 😵",
        blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": text
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"Error: {error} 😵"
                }
            }
        ]
    )

def send_excel_file(channel_id: str, excel_buffer: io.BytesIO, preview: str):
    """Send Excel file with preview"""
    # Send preview message
    slack_client.chat_postMessage(
        channel=channel_id,
        text=preview,
        blocks=[{
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": preview
            }
        }]
    )
    
    # Upload Excel file
    return slack_client.files_upload_v2(
        channel=channel_id,
        file=excel_buffer,
        filename="query_results.xlsx",
        title="Query Results",
        initial_comment="📊 Download complete results"
    ) 