from .table_handler import TableHandler
import logging

logger = logging.getLogger(__name__)

def format_query_result(query_result):
    """Format query result as a readable table for Slack"""
    if not query_result:
        return ""
    
    try:
        # Extract columns and rows
        columns = [str(col) for col in query_result['columns']]
        rows = query_result['rows']
        
        if not rows:
            return "No results found"
        
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
        
        return "```\n" + "\n".join(table) + "\n```"
        
    except Exception as e:
        logger.error(f"Error formatting table: {e}")
        return str(query_result)

def format_slack_message(response):
    """Format the response for Slack with support for large tables"""
    if not response.query_result:
        return {"type": "text", "text": "No results to display"}
    
    table_handler = TableHandler()
    
    # Analyze data dimensions
    num_rows, num_cols = table_handler.analyze_data(response.query_result)
    
    # Determine format based on data size
    if table_handler.should_use_excel(num_rows, num_cols):
        excel_result = table_handler.create_excel_file(response.query_result)
        if excel_result:
            excel_buffer, summary = excel_result
            return {
                "type": "excel",
                "excel_file": excel_buffer,
                "text": summary
            }
    
    # For smaller tables, use regular formatting
    formatted_table = format_query_result(response.query_result)
    return {
        "type": "table",
        "text": "*Results:*\n" + formatted_table
    } 