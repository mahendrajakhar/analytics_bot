from slack_sdk import WebClient
from slack_sdk.signature import SignatureVerifier
from api.config import settings, logger
import io
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import logging
import tempfile
import math

logger = logging.getLogger(__name__)

# Initialize Slack client
slack_client = WebClient(token=settings.SLACK_BOT_TOKEN)
signature_verifier = SignatureVerifier(settings.SLACK_SIGNING_SECRET)

class TableHandler:
    """Handles large table data formatting and conversion"""
    
    MAX_COLUMNS = 10  # Updated from 3 to 10
    MAX_ROWS = 40    # Updated from 15 to 30
    
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

def convert_timezone_aware_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timezone-aware datetime columns to timezone-naive"""
    for column in df.select_dtypes(include=['datetime64[ns, UTC]']).columns:
        df[column] = df[column].dt.tz_localize(None)
    return df

def format_query_result(query_result: Dict[str, Any]) -> Dict[str, Any]:
    """Format query results for Slack display"""
    try:
        if not query_result or not query_result.get('rows'):
            return {"text": "No results found.", "type": "text"}

        df = pd.DataFrame(query_result['rows'], columns=query_result['columns'])
        
        # Convert timezone-aware datetime columns
        df = convert_timezone_aware_columns(df)
        
        total_rows = len(df)
        total_cols = len(df.columns)
        
        # For results within limits, return as code block
        if total_rows <= TableHandler.MAX_ROWS and total_cols <= TableHandler.MAX_COLUMNS:
            # Format datetime columns for display
            for col in df.select_dtypes(include=['datetime64']).columns:
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            table_str = df.to_string(index=False)
            return {
                "text": f"Query returned {total_rows} rows and {total_cols} columns:\n```{table_str}```",
                "type": "text"
            }
        
        # For larger results, create Excel file
        # Calculate sheets needed (max 1M rows per sheet)
        MAX_ROWS_PER_SHEET = 1000000
        total_sheets = math.ceil(total_rows / MAX_ROWS_PER_SHEET)
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
            if total_sheets == 1:
                # Single sheet case
                df.to_excel(temp_file.name, index=False, sheet_name='Results')
            else:
                # Multiple sheets case
                with pd.ExcelWriter(temp_file.name, engine='xlsxwriter') as writer:
                    for i in range(total_sheets):
                        start_idx = i * MAX_ROWS_PER_SHEET
                        end_idx = min((i + 1) * MAX_ROWS_PER_SHEET, total_rows)
                        sheet_name = f'Results_Part_{i+1}'
                        
                        # Get the slice of data for this sheet
                        sheet_df = df.iloc[start_idx:end_idx].copy()
                        
                        sheet_df.to_excel(
                            writer, 
                            sheet_name=sheet_name,
                            index=False
                        )
                        
                        # Optional: Adjust column widths for better readability
                        worksheet = writer.sheets[sheet_name]
                        for idx, col in enumerate(sheet_df.columns):
                            max_length = max(
                                sheet_df[col].astype(str).str.len().max(),
                                len(str(col))
                            )
                            worksheet.set_column(idx, idx, min(max_length + 2, 50))

        return {
            "text": (f"Query returned {total_rows} rows and {total_cols} columns. "
                    f"Due to size (>30 rows or >10 columns), data has been exported to Excel "
                    f"and split into {total_sheets} sheet{'s' if total_sheets > 1 else ''}."),
            "type": "excel",
            "excel_file": temp_file.name
        }

    except Exception as e:
        logger.error(f"Error formatting query result: {e}")
        logger.exception("Full traceback:")  # Add full traceback logging
        return {
            "text": f"Error creating result format: {str(e)}",
            "type": "text"
        }

def send_user_question_in_initial_response(channel_id: str, question: str, command: str) -> dict:
    """Send initial response with stylized user question"""
    try:
        # Create a stylized question block with command context
        command_emoji = {
            "/sql": "🔍",
            "/ask": "💡",
            "/graph": "📊",
            "/help": "ℹ️"
        }.get(command, "❓")
        
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"```{command} {command_emoji} {question}```"
                }
            },
            {
                "type": "divider"
            }
        ]

        # Send the message
        response = slack_client.chat_postMessage(
            channel=channel_id,
            text=f"Question: {question}",
            blocks=blocks
        )
        
        if not response["ok"]:
            logger.error(f"Error sending initial response: {response.get('error')}")
            raise Exception(f"Failed to send initial response: {response.get('error')}")
            
        return response

    except Exception as e:
        logger.error(f"Error sending initial response: {e}")
        # Send a simpler fallback message
        try:
            return slack_client.chat_postMessage(
                channel=channel_id,
                text=f"Processing: {question}"
            )
        except Exception as fallback_error:
            logger.error(f"Error sending fallback message: {fallback_error}")
            raise

def send_initial_response(channel_id: str, question: str) -> dict:
    """Send initial 'processing' message with user's question"""
    try:
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Question:*\n{question}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "Processing your request... 🧠"
                }
            }
        ]
        
        response = slack_client.chat_postMessage(
            channel=channel_id,
            text=f"Question: {question}\nProcessing...",
            blocks=blocks
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
    """Update final message with feedback buttons"""
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
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "Visualization complete! ✨"
                }
            })

        # Add feedback buttons
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "👍 Helpful",
                        "emoji": True
                    },
                    "value": "thumbs_up",
                    "action_id": "feedback_helpful"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "👎 Not Helpful",
                        "emoji": True
                    },
                    "value": "thumbs_down",
                    "action_id": "feedback_not_helpful"
                }
            ]
        })
        
        # Update the message
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
                text="Query completed",
                blocks=[{
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "Query completed"
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

def update_help_message(channel_id: str, ts: str, formatted_response: dict):
    """Update help message without feedback buttons"""
    try:
        # Update the message
        response = slack_client.chat_update(
            channel=channel_id,
            ts=ts,
            text=formatted_response["text"],
            blocks=[]
        )
        
        if not response["ok"]:
            logger.error(f"Error updating help message: {response.get('error')}")
            raise Exception(f"Failed to update help message: {response.get('error')}")
            
    except Exception as e:
        logger.error(f"Error updating help message: {e}")
        try: send_error(channel_id, ts, formatted_response, "Error updating help message")
        except Exception as fallback_error:
            logger.error(f"Error sending fallback help message: {fallback_error}")
            raise
