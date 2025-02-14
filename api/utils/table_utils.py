import pandas as pd
import io
from typing import Dict, Any, Tuple

class TableHandler:
    MAX_PREVIEW_COLUMNS = 5  # Only for preview display
    MAX_PREVIEW_ROWS = 20     # Only for preview display
    
    @staticmethod
    def analyze_data_size(query_result: Dict[str, Any]) -> Tuple[int, int]:
        """Analyze data dimensions"""
        if not query_result or 'rows' not in query_result:
            return 0, 0
            
        rows = len(query_result['rows'])
        columns = len(query_result['columns']) if query_result['columns'] else 0
        return rows, columns

    @staticmethod
    def create_excel_file(query_result: Dict[str, Any], question: str) -> Tuple[io.BytesIO, str]:
        """Create formatted Excel file from query results"""
        df = pd.DataFrame(query_result['rows'], columns=query_result['columns'])
        
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            # Write main data with optimized settings for large datasets
            df.to_excel(writer, sheet_name='Query Results', index=False)
            
            # Get workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Query Results']
            
            # Add formats
            header_format = workbook.add_format({
                'bold': True,
                'fg_color': '#D7E4BC',
                'border': 1,
                'text_wrap': True
            })
            
            # Auto-fit columns (with max width limit for very wide content)
            for idx, col in enumerate(df.columns):
                # Sample only first 1000 rows for column width calculation
                sample = df[col].head(1000).astype(str)
                max_length = max(
                    sample.str.len().max(),
                    len(str(col))
                ) + 2
                worksheet.set_column(idx, idx, min(max_length, 50))
            
            # Apply header format
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                
            # Add metadata sheet
            metadata = workbook.add_worksheet('Info')
            metadata.write('A1', 'Query Information')
            metadata.write('A2', 'Question:')
            metadata.write('B2', question)
            metadata.write('A3', 'Total Rows:')
            metadata.write('B3', len(df))
            metadata.write('A4', 'Total Columns:')
            metadata.write('B4', len(df.columns))
            
        excel_buffer.seek(0)
        return excel_buffer, f"query_results_{len(df.columns)}cols_{len(df)}rows.xlsx"

    @staticmethod
    def format_preview(query_result: Dict[str, Any]) -> str:
        """Format a preview of the data for Slack message"""
        if not query_result or not query_result['rows']:
            return "No results found"

        total_rows = len(query_result['rows'])
        total_cols = len(query_result['columns'])
        
        preview_rows = query_result['rows'][:TableHandler.MAX_PREVIEW_ROWS]
        preview_cols = query_result['columns'][:TableHandler.MAX_PREVIEW_COLUMNS]
        
        # Format header
        header = " | ".join(str(col) for col in preview_cols)
        separator = "-" * len(header)
        
        # Format preview rows
        formatted_rows = []
        for row in preview_rows:
            preview_values = [str(row[i]) for i in range(min(len(row), TableHandler.MAX_PREVIEW_COLUMNS))]
            formatted_row = " | ".join(preview_values)
            formatted_rows.append(formatted_row)
        
        # Combine parts with summary
        preview = (
            f"Total: {total_rows} rows × {total_cols} columns\n\n"
            f"{header}\n{separator}\n"
            f"{chr(10).join(formatted_rows)}\n"
        )
        
        if total_rows > TableHandler.MAX_PREVIEW_ROWS:
            preview += f"\n... and {total_rows - TableHandler.MAX_PREVIEW_ROWS} more rows"
        if total_cols > TableHandler.MAX_PREVIEW_COLUMNS:
            preview += f"\n... and {total_cols - TableHandler.MAX_PREVIEW_COLUMNS} more columns"
            
        return preview
