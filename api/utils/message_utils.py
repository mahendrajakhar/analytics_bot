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

        # Convert to DataFrame for easier handling
        df = pd.DataFrame(query_result['rows'], columns=query_result['columns'])
        
        # Format columns based on name and content
        for col in df.columns:
            # Check if column is an ID field
            is_id_field = any(id_term in col.lower() for id_term in ['_id', 'id_', 'id'])
            
            # Format based on column type and name
            if df[col].dtype == 'datetime64[ns]' or 'datetime' in str(df[col].dtype).lower():
                # Format datetime without timezone
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            elif 'date' in col.lower():
                # Try to convert string dates to proper format
                try:
                    df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
                except:
                    pass
            elif is_id_field:
                # Handle ID columns - convert to integer without formatting
                df[col] = df[col].apply(lambda x: str(int(float(x))) if pd.notnull(x) and str(x).strip() else '')
            elif df[col].dtype in ['int64', 'float64']:
                # Format other numeric columns with thousand separators
                df[col] = df[col].apply(lambda x: f"{x:,}" if pd.notnull(x) else '')
            
            # Ensure all values are strings and handle None/NaN
            df[col] = df[col].fillna('').astype(str)

        total_rows = len(df)
        total_cols = len(df.columns)
        
        # For results within limits, return as formatted table
        if total_rows <= TableHandler.MAX_ROWS and total_cols <= TableHandler.MAX_COLUMNS:
            # Create formatted table string
            table_str = "```\n"
            
            # Add headers with proper spacing
            col_widths = {col: max(len(str(col)), df[col].str.len().max()) 
                         for col in df.columns}
            
            # Format header
            header = " | ".join(f"{col:<{col_widths[col]}}" for col in df.columns)
            separator = "-" * len(header)
            
            table_str += f"{header}\n{separator}\n"
            
            # Format rows
            for _, row in df.iterrows():
                formatted_row = " | ".join(f"{str(val):<{col_widths[col]}}" 
                                         for col, val in row.items())
                table_str += f"{formatted_row}\n"
            
            table_str += "```"
            
            return {
                "text": f"Query returned {total_rows} rows and {total_cols} columns:\n{table_str}",
                "type": "text"
            }
        
        # For larger results, create Excel file
        excel_buffer = io.BytesIO()
        
        # Create Excel writer with xlsxwriter engine for better formatting
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Results', index=False)
            
            # Get workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Results']
            
            # Add formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#D7E4BC',
                'border': 1,
                'text_wrap': True
            })
            
            # Auto-fit columns
            for idx, col in enumerate(df.columns):
                max_length = max(
                    df[col].astype(str).str.len().max(),
                    len(str(col))
                ) + 2
                worksheet.set_column(idx, idx, min(max_length, 50))
                
                # Write headers with format
                worksheet.write(0, idx, col, header_format)

        excel_buffer.seek(0)
        
        # Create preview of data
        preview_df = df.head(5)
        preview_str = "```\n"
        
        # Format preview headers
        preview_col_widths = {col: max(len(str(col)), preview_df[col].str.len().max()) 
                            for col in preview_df.columns}
        preview_header = " | ".join(f"{col:<{preview_col_widths[col]}}" 
                                  for col in preview_df.columns)
        preview_separator = "-" * len(preview_header)
        
        preview_str += f"{preview_header}\n{preview_separator}\n"
        
        # Format preview rows
        for _, row in preview_df.iterrows():
            formatted_row = " | ".join(f"{str(val):<{preview_col_widths[col]}}" 
                                     for col, val in row.items())
            preview_str += f"{formatted_row}\n"
        
        preview_str += f"```\n(Showing 5 of {total_rows} rows)"

        return {
            "text": preview_str,
            "type": "excel",
            "excel_file": excel_buffer
        }

    except Exception as e:
        logger.error(f"Error formatting query result: {e}")
        return {"text": f"Error formatting results: {str(e)}", "type": "text"}

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

        # Add feedback and summarize buttons
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
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "📝 Summarize",
                        "emoji": True
                    },
                    "value": "summarize",
                    "action_id": "summarize"
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

def send_excel_file(channel_id: str, excel_buffer: io.BytesIO, preview: str, history_entry: dict = None):
    """Send Excel file with preview"""
    try:
        file_response = slack_client.files_upload_v2(
            channel=channel_id,
            file=excel_buffer,
            title="Query Results",
            filename="query_results.xlsx",
            initial_comment=f"Here are your query results:\n```\n{preview}\n```"
        )
        
        # Then send a follow-up message with buttons
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "Excel file has been uploaded with the query results."
                }
            },
            {
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
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "📝 Summarize",
                            "emoji": True
                        },
                        "value": "summarize",
                        "action_id": "summarize"
                    }
                ]
            }
        ]
        
        # Send follow-up message with buttons
        button_response = slack_client.chat_postMessage(
            channel=channel_id,
            text="Excel file has been uploaded with the query results.",
            blocks=blocks
        )

        # If we have a history entry, update it with the new message timestamp
        if history_entry and button_response['ok']:
            from api.handlers.history_handlers import save_chat_history
            
            # Create a new history entry with the button message timestamp
            save_chat_history(
                user_id=history_entry['user_id'],
                command=history_entry['command'],
                question=history_entry['question'],
                message_ts=button_response['ts'],  # Use the new message timestamp
                sql_query=history_entry.get('sql_query'),
                response=history_entry.get('response'),
                ai_response=history_entry.get('ai_response')
            )
        
        return file_response
        
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
