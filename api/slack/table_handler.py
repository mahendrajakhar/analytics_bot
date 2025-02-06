import pandas as pd
import io
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

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

    @staticmethod
    def format_preview(query_result: Dict[str, Any], max_rows: int = 5) -> str:
        """Format a preview of the data"""
        columns = [str(col) for col in query_result['columns']]
        rows = query_result['rows'][:max_rows]
        
        preview = ["```"]
        
        # Headers
        header = " | ".join(f"{col[:15]:<15}" for col in columns)
        separator = "-" * (17 * len(columns))
        
        preview.append(header)
        preview.append(separator)
        
        # Rows
        for row in rows:
            row_str = " | ".join(f"{str(val)[:15]:<15}" for val in row)
            preview.append(row_str)
        
        preview.append("```")
        return "\n".join(preview) 