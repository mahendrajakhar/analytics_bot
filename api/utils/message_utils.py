from slack_sdk import WebClient
from slack_sdk.signature import SignatureVerifier
from api.config import settings, logger
import io
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Initialize Slack client
slack_client = WebClient(token=settings.SLACK_BOT_TOKEN)
signature_verifier = SignatureVerifier(settings.SLACK_SIGNING_SECRET)

class TableHandler:
    """Handles large table data formatting and conversion"""
    
    MAX_COLUMNS = 3
    MAX_ROWS = 15
    
    @staticmethod
    def analyze_data(query_result: Dict[str, Any]) -> Tuple[int, int]:
        """Analyze the data dimensions"""
        if not query_result or 'rows' not in query_result:
            return 0, 0
            
        columns = query_result.get('columns', [])
        rows = query_result.get('rows', [])
        
        return len(rows), len(columns)

    @staticmethod
    def should_use_excel(rows: int, columns: int) -> bool:
        """Determine if data should be sent as Excel file"""
        return rows > TableHandler.MAX_ROWS or columns > TableHandler.MAX_COLUMNS

    @staticmethod
    def create_excel_file(query_result: Dict[str, Any]) -> Optional[Tuple[io.BytesIO, str]]:
        """Create Excel file from query results"""
        try:
            columns = [str(col) for col in query_result['columns']]
            rows = query_result['rows']
            
            # Create DataFrame with all data
            df = pd.DataFrame(rows, columns=columns)
            
            # Create Excel file in memory
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Query Results', index=False)
                
                # Auto-adjust columns width
                worksheet = writer.sheets['Query Results']
                for idx, col in enumerate(df.columns):
                    max_length = max(
                        df[col].astype(str).apply(len).max(),
                        len(str(col))
                    ) + 2
                    worksheet.set_column(idx, idx, max_length)

            excel_buffer.seek(0)
            
            # Create summary text
            summary = (
                "*Query Results Summary:*\n"
                f"• Total Rows: {len(df)}\n"
                f"• Total Columns: {len(df.columns)}\n\n"
                "📊 Complete results available in the Excel file above."
            )
            
            return excel_buffer, summary
            
        except Exception as e:
            logger.error(f"Error creating Excel file: {e}")
            return None

def format_query_result(query_result: Dict[str, Any]) -> Dict[str, Any]:
    """Format query result for Slack with support for large tables"""
    if not query_result:
        return {"type": "text", "text": "No results to display"}
    
    table_handler = TableHandler()
    
    # Analyze data dimensions
    num_rows, num_cols = table_handler.analyze_data(query_result)
    
    # Determine format based on data size
    if table_handler.should_use_excel(num_rows, num_cols):
        excel_result = table_handler.create_excel_file(query_result)
        if excel_result:
            excel_buffer, summary = excel_result
            return {
                "type": "excel",
                "excel_file": excel_buffer,
                "text": summary
            }
    
    # For smaller tables, use regular formatting
    try:
        # Extract columns and rows
        columns = [str(col) for col in query_result['columns']]
        rows = query_result['rows']
        
        if not rows:
            return {"type": "text", "text": "No results found"}
        
        # Calculate max width for each column
        col_widths = {col: len(col) for col in columns}
        for row in rows:
            for i, value in enumerate(row):
                col_widths[columns[i]] = max(col_widths[columns[i]], len(str(value)))
        
        # Build table
        table = []
        
        # Header
        header = "| " + " | ".join(f"{col:<{col_widths[col]}}" for col in columns) + " |"
        separator = "|" + "|".join("-" * (width + 2) for width in col_widths.values()) + "|"
        
        table.append(header)
        table.append(separator)
        
        # Rows
        for row in rows:
            row_str = "| " + " | ".join(f"{str(value):<{col_widths[columns[i]]}} " 
                          for i, value in enumerate(row)) + "|"
            table.append(row_str)
        
        return {
            "type": "table",
            "text": "*Results:*\n```\n" + "\n".join(table) + "\n```"
        }
        
    except Exception as e:
        logger.error(f"Error formatting table: {e}")
        return {"type": "text", "text": str(query_result)}

def send_initial_response(channel_id: str) -> dict:
    """Send initial 'processing' message"""
    try:
        response = slack_client.chat_postMessage(
            channel=channel_id,
            text="Processing your request... 🧠"
        )
        return response
    except Exception as e:
        logger.error(f"Error sending initial response: {e}")
        raise

def update_with_table(channel_id: str, ts: str, formatted_response: dict):
    """Update message with query results"""
    try:
        slack_client.chat_update(
            channel=channel_id,
            ts=ts,
            text=formatted_response.get('text', ''),
            blocks=formatted_response.get('blocks', [])
        )
    except Exception as e:
        logger.error(f"Error updating with table: {e}")
        raise

def send_graph_status(channel_id: str, ts: str, formatted_response: dict):
    """Update message to show graph generation status"""
    try:
        response = slack_client.chat_update(
            channel=channel_id,
            ts=ts,
            text=formatted_response.get("text", "Generating visualization... 📊"),
            blocks=[
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": formatted_response.get("text", "Generating visualization... 📊")
                    }
                }
            ]
        )
        if not response["ok"]:
            raise Exception(f"Error updating graph status: {response.get('error')}")
            
    except Exception as e:
        logger.error(f"Error updating graph status: {e}")
        raise

def send_graph(channel_id: str, img_buffer: bytes, text: str):
    """Upload graph to Slack"""
    try:
        # Upload the image file
        response = slack_client.files_upload_v2(
            channel=channel_id,
            file=img_buffer,
            filename="graph.png",
            title=f"Visualization for: {text}",
            initial_comment="Here's your visualization 📈"
        )
        
        if not response["ok"]:
            raise Exception(f"Error uploading graph: {response.get('error')}")
            
        return response
        
    except Exception as e:
        logger.error(f"Error sending graph: {e}")
        raise

def update_final_message(channel_id: str, ts: str, formatted_response: dict):
    """Update final message after graph upload"""
    try:
        # Create blocks array
        blocks = []
        
        # Add text block if present
        if formatted_response.get("text"):
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": formatted_response["text"]
                }
            })
        
        # Add image block if graph_url is present
        if formatted_response.get("graph_url"):
            blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": formatted_response['text']
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
        
        # Update the message with text only
        response = slack_client.chat_update(
            channel=channel_id,
            ts=ts,
            text=formatted_response.get("text", "Query completed"),
            blocks=blocks
        )
        
        if not response["ok"]:
            logger.error(f"Error updating message: {response.get('error')}")
            raise Exception(f"Failed to update message: {response.get('error')}")
            
    except Exception as e:
        logger.error(f"Error updating final message: {e}")
        # Fallback to simpler message if update fails
        try:
            slack_client.chat_update(
                channel=channel_id,
                ts=ts,
                text="Query completed. Check the visualization above! 📊",
                blocks=[{
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "Query completed. Check the visualization above! 📊"
                    }
                }]
            )
        except Exception as fallback_error:
            logger.error(f"Error sending fallback message: {fallback_error}")
            raise

def send_error(channel_id: str, ts: str, formatted_response: dict, error: str):
    """Send error message"""
    try:
        slack_client.chat_update(
            channel=channel_id,
            ts=ts,
            text=f"Error: {error}",
            blocks=formatted_response.get('blocks', [])
        )
    except Exception as e:
        logger.error(f"Error sending error message: {e}")
        raise

def send_excel_file(channel_id: str, excel_buffer: io.BytesIO, preview: str):
    """Send Excel file with preview"""
    try:
        return slack_client.files_upload_v2(
            channel=channel_id,
            file=excel_buffer,
            title="Query Results",
            filename="query_results.xlsx",
            initial_comment=f"Here are your query results:\n```\n{preview}\n```"
        )
    except Exception as e:
        logger.error(f"Error sending Excel file: {e}")
        raise

def send_graph_image(channel_id: str, image_buffer: io.BytesIO, summary: str):
    """Send graph image to Slack channel"""
    try:
        # Upload image file
        response = slack_client.files_upload_v2(
            channel=channel_id,
            file=image_buffer,
            filename="graph.png",
            title="Data Visualization",
            initial_comment=summary
        )
        if not response["ok"]:
            raise Exception(f"Error uploading graph: {response.get('error')}")
            
    except Exception as e:
        logger.error(f"Error sending graph image: {e}")
        raise

def format_slack_message(response: str) -> dict:
    """Format the response for Slack with support for large tables"""
    blocks = []
    if response:
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": response}
        })
    return {"text": response, "blocks": blocks}

def update_with_processing(channel_id: str, ts: str, message: str):
    """Update message with processing status"""
    try:
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message
                }
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "⏳ Processing..."
                    }
                ]
            }
        ]
        
        slack_client.chat_update(
            channel=channel_id,
            ts=ts,
            text=message,
            blocks=blocks
        )
    except Exception as e:
        logger.error(f"Error updating processing status: {e}")
        raise

def get_help_text():
    """Get help text for available commands"""
    return """
Available commands:
• `/sql [question]` - Get the SQL query for your question without executing it
• `/ask [question]` - Execute the SQL query and get the results
• `/graph [question]` - Generate a graph based on your question
• `/help` - Show this help message

Examples:
• `/sql Show me the total sales by region`
• `/ask What were the top 5 products sold last month?`
• `/graph Show me a bar chart of monthly revenue`
"""
