import json
import pandas as pd
import requests
from typing import List, Dict, Any, Optional, Tuple
from api.config import settings
from databricks import sql as db_sql
from datetime import datetime
from contextlib import contextmanager

class DatabricksAnalyzer:
    def __init__(self):
        """Initialize the Databricks analyzer with connection details."""
        self.connection_params = {
            "server_hostname": settings.DATABRICKS_HOST,
            "http_path": settings.DATABRICKS_HTTP_PATH,
            "access_token": settings.DATABRICKS_TOKEN
        }
        self.connection = None

    @contextmanager
    def get_connection(self):
        """Context manager for database connection."""
        connection = None
        try:
            connection = db_sql.connect(
                server_hostname=self.connection_params.get('server_hostname'),
                http_path=self.connection_params.get('http_path'),
                access_token=self.connection_params.get('access_token')
            )
            yield connection
        finally:
            if connection:
                connection.close()

    def safe_value_convert(self, value: Any) -> Any:
        """Safely convert values to appropriate string representations."""
        if value is None:
            return None
            
        if isinstance(value, datetime):
            try:
                # Try different datetime formats
                format_attempts = [
                    '%Y-%m-%d %H:%M:%S.%f',  # With microseconds
                    '%Y-%m-%d %H:%M:%S',      # Without microseconds
                    '%Y-%m-%d'                # Just date
                ]
                
                for fmt in format_attempts:
                    try:
                        return value.strftime(fmt)
                    except ValueError:
                        continue
                        
                # If all formats fail, just convert to string
                return str(value)
                
            except Exception as e:
                print(f"Error converting datetime: {str(e)}")
                return str(value)
                
        if isinstance(value, (int, float)):
            return value
            
        # For all other types, convert to string
        return str(value)
    
    def execute_query(self, query: str) -> Optional[Dict]:
        """Execute a query and return results."""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    
                    # Get column names if available
                    columns = []
                    if cursor.description:
                        columns = [desc[0] for desc in cursor.description]
                    
                    # Fetch results
                    results = cursor.fetchall()
                    
                    # Convert to list of lists for easier handling
                    processed_results = []
                    for row in results:
                        processed_row = []
                        for value in row:
                            try:
                                # Handle each value individually
                                processed_value = self.safe_value_convert(value)
                                processed_row.append(processed_value)
                            except Exception as e:
                                print(f"Error processing value {value}: {str(e)}")
                                processed_row.append(str(value))
                        processed_results.append(tuple(processed_row))
                    
                    return {
                        "results": {
                            "schema": [{"name": col} for col in columns],
                            "data": processed_results
                        }
                    }
                    
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            print(f"Query was: {query}")
            return None

    def get_sample_rows(self, table_name: str, num_rows: int = 10) -> List[Dict]:
        """Get sample rows from a table."""
        try:
            # Try without ORDER BY RAND() first as it might be causing issues
            query = f"""
            SELECT *
            FROM {table_name}
            LIMIT {num_rows}
            """
            result = self.execute_query(query)
            
            if not result or "results" not in result:
                return []
                
            columns = [col["name"] for col in result["results"]["schema"]]
            rows = result["results"]["data"]
            
            # Convert rows to dictionaries with proper value handling
            return [
                {
                    col: self.safe_value_convert(val) 
                    for col, val in zip(columns, row)
                }
                for row in rows
            ]
                
        except Exception as e:
            print(f"Error getting sample rows for {table_name}: {str(e)}")
            return []

    def get_column_unique_counts(self, table_name: str) -> Dict[str, int]:
        """Get count of unique values for each column in a table."""
        try:
            # First get columns
            columns = self.get_column_info(table_name)
            if not columns:
                return {}
            
            unique_counts = {}
            for col in columns:
                column_name = col["name"]
                query = f"""
                SELECT COUNT(DISTINCT {column_name}) as count
                FROM {table_name}
                """
                
                try:
                    result = self.execute_query(query)
                    if result and "results" in result and result["results"]["data"]:
                        unique_counts[column_name] = result["results"]["data"][0][0] or 0
                except Exception as e:
                    print(f"Error getting unique count for column {column_name}: {str(e)}")
                    unique_counts[column_name] = 0
                    
            return unique_counts
            
        except Exception as e:
            print(f"Error in get_column_unique_counts: {str(e)}")
            return {}

    def get_unique_values(self, table_name: str, column_name: str) -> List[Any]:
        """Get all unique values for a specific column."""
        try:
            query = f"""
            SELECT DISTINCT {column_name}
            FROM {table_name}
            WHERE {column_name} IS NOT NULL
            ORDER BY {column_name}
            LIMIT 50
            """
            
            result = self.execute_query(query)
            if result and "results" in result:
                return [row[0] for row in result["results"]["data"]]
        except Exception as e:
            print(f"Error getting unique values for column {column_name}: {str(e)}")
        return []

    def get_column_info(self, table_name: str) -> List[Dict[str, str]]:
        """Get column information for a table."""
        try:
            # Parse the table name components
            parts = table_name.split('.')
            if len(parts) == 3:
                catalog, schema, table = parts
            else:
                return []

            query = f"""
            SELECT column_name, data_type
            FROM {catalog}.information_schema.columns
            WHERE table_catalog = '{catalog}'
              AND table_schema = '{schema}'
              AND table_name = '{table}'
            ORDER BY ordinal_position
            """
            
            result = self.execute_query(query)
            if result and "results" in result:
                return [{"name": row[0], "type": row[1]} for row in result["results"]["data"]]
        except Exception as e:
            print(f"Error getting column info: {str(e)}")
        return []

    def analyze_tables(self, tables: List[str]) -> Dict:
        """Analyze tables and return results."""
        results = {}
        for table_name in tables:
            print(f"\nAnalyzing table: {table_name}")
            
            # Get column info first
            print("Getting column info...")
            columns = self.get_column_info(table_name)
            if not columns:
                print(f"Skipping table {table_name} - couldn't get column info")
                continue

            print("Getting sample rows...")
            sample_rows = self.get_sample_rows(table_name)
            
            print("Getting unique counts...")
            unique_counts = self.get_column_unique_counts(table_name)
            
            table_data = {
                "columns": columns,
                "sample_rows": sample_rows,
                "column_unique_counts": unique_counts,
                "columns_with_few_uniques": {}
            }

            print("Getting unique values for columns with less than 50 unique values...")
            # Get unique values for columns with less than 50 unique values
            for column, count in unique_counts.items():
                if count is not None and count < 50:
                    unique_values = self.get_unique_values(table_name, column)
                    if unique_values:
                        table_data["columns_with_few_uniques"][column] = unique_values

            results[table_name] = table_data
            print(f"Completed analysis for table: {table_name}")

        return results

def main():
    try:
        # Initialize analyzer
        analyzer = DatabricksAnalyzer()
        
        # Get list of tables or use specific tables
        # tables = analyzer.get_tables()
        tables = ['analytics.super.premium_user_all_orders']
        print(f"Found tables: {tables}")
        
        # Analyze tables
        results = analyzer.analyze_tables(tables)
        
        # Save results to JSON file
        output_file = "tables_data_for_context.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nAnalysis complete. Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()