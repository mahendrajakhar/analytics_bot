import requests
import pandas as pd
from typing import Dict, Optional, Tuple
from api.config import settings

class DatabricksService:
    def __init__(self):
        self.hostname = settings.DATABRICKS_HOST
        self.http_path = settings.DATABRICKS_HTTP_PATH
        self.token = settings.DATABRICKS_TOKEN
        self.api_url = f"https://{self.hostname}{self.http_path}"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

    async def execute_query(self, sql_query: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Execute SQL query on Databricks and return results"""
        try:
            # Execute query using Databricks SQL API
            data = {
                "statement": sql_query,
                "wait_timeout": "50s"
            }
            
            # Make request to Databricks SQL endpoint
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=data
            )
            
            if response.status_code != 200:
                return None, f"Databricks API error: {response.text}"

            result = response.json()
            
            if "error" in result:
                return None, f"Query error: {result['error']}"
            
            # Extract column names and data
            if "results" in result:
                columns = [col["name"] for col in result["results"]["schema"]]
                rows = result["results"]["data"]
                
                return {
                    "columns": columns,
                    "rows": rows
                }, None
            
            return None, "No results returned from query"

        except Exception as e:
            return None, str(e)

    def get_table_schema(self) -> str:
        """Get schema information from Databricks"""
        try:
            # Query to get table schema information
            schema_query = """
            SELECT table_schema, table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema NOT IN ('information_schema', 'sys')
            ORDER BY table_schema, table_name, ordinal_position
            """
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"statement": schema_query}
            )
            
            if response.status_code != 200:
                return "Error fetching schema information"

            result = response.json()
            
            if "error" in result:
                return f"Error: {result['error']}"
            
            # Format schema information
            schema_text = []
            current_table = None
            
            for row in result["results"]["data"]:
                schema, table, column, data_type = row
                
                if f"{schema}.{table}" != current_table:
                    if current_table:
                        schema_text.append("")  # Add blank line between tables
                    current_table = f"{schema}.{table}"
                    schema_text.append(f"Table: {schema}.{table}")
                
                schema_text.append(f"  - {column} ({data_type})")
            
            return "\n".join(schema_text)
            
        except Exception as e:
            return f"Error getting schema: {str(e)}"

# Initialize service
databricks_service = DatabricksService() 