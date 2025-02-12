import pandas as pd
import io
from typing import Dict, Any

class TableHandler:
    MAX_COLUMNS = 3
    MAX_ROWS = 15

    @staticmethod
    def analyze_data(query_result: Dict[str, Any]):
        """Analyze the data dimensions"""
        if not query_result or 'rows' not in query_result:
            return 0, 0
        rows = len(query_result['rows'])
        columns = len(query_result['columns']) if query_result['columns'] else 0
        return rows, columns

    @staticmethod
    def should_use_excel(rows: int, columns: int) -> bool:
        """Determine if data should be sent as Excel file"""
        return rows > TableHandler.MAX_ROWS or columns > TableHandler.MAX_COLUMNS

    @staticmethod
    def create_excel_file(query_result: Dict[str, Any]) -> io.BytesIO:
        """Create Excel file from query results"""
        df = pd.DataFrame(query_result['rows'], columns=query_result['columns'])
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Query Results', index=False)
        excel_buffer.seek(0)
        return excel_buffer

    @staticmethod
    def format_preview(query_result: Dict[str, Any], max_rows: int = 5) -> str:
        """Format a preview of the data"""
        if not query_result or not query_result['rows']:
            return "No results found"

        preview_rows = query_result['rows'][:max_rows]
        columns = query_result['columns']
        
        # Format header
        header = " | ".join(str(col) for col in columns)
        separator = "-" * len(header)
        
        # Format rows
        formatted_rows = []
        for row in preview_rows:
            formatted_row = " | ".join(str(cell) for cell in row)
            formatted_rows.append(formatted_row)
        
        # Combine all parts
        preview = f"{header}\n{separator}\n" + "\n".join(formatted_rows)
        
        if len(query_result['rows']) > max_rows:
            preview += f"\n... and {len(query_result['rows']) - max_rows} more rows"
        
        return preview
