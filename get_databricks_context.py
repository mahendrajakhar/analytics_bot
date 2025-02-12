import json
import pandas as pd
import requests
from typing import List, Dict, Any
from api.config import settings

class DatabricksAnalyzer:
    def __init__(self):
        """Initialize the Databricks analyzer with connection details."""
        print("Initializing DatabricksAnalyzer...")
        self.hostname = settings.DATABRICKS_HOST
        self.http_path = settings.DATABRICKS_HTTP_PATH
        self.token = settings.DATABRICKS_TOKEN
        self.api_url = f"https://{self.hostname}{self.http_path}"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        print(f"Initialized with hostname: {self.hostname}")

    def execute_query(self, query: str) -> Dict:
        """Execute a query and return results."""
        try:
            print(f"Executing query: {query}")
            print(f"Sending request to: {self.api_url}")
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"statement": query}
            )
            print(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Query failed with status code: {response.status_code}")
                raise Exception(f"Query failed: {response.text}")
            
            result = response.json()
            if "error" in result:
                print(f"Query returned error: {result['error']}")
                raise Exception(f"Query error: {result['error']}")
            
            print("Query executed successfully")    
            return result
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            return None

    def get_tables(self) -> List[str]:
        """Get list of all tables in the database."""
        print("Getting list of tables...")
        query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema NOT IN ('information_schema', 'sys')
        """
        result = self.execute_query(query)
        if result and "results" in result:
            tables = [row[0] for row in result["results"]["data"]]
            print(f"Found {len(tables)} tables")
            return tables
        print("No tables found")
        return []

    def get_sample_rows(self, table_name: str, num_rows: int = 1) -> List[Dict]:
        """Get sample rows from a table."""
        print(f"Getting {num_rows} sample rows from table: {table_name}")
        query = f"SELECT * FROM {table_name} LIMIT 1"
        result = self.execute_query(query)
        
        if not result or "results" not in result:
            print("No results found")
            return []
            
        columns = [col["name"] for col in result["results"]["schema"]]
        rows = result["results"]["data"]
        print(f"Retrieved {len(rows)} rows")
        
        return [dict(zip(columns, row)) for row in rows]

    def get_column_unique_counts(self, table_name: str) -> Dict[str, int]:
        """Get count of unique values for each column in a table."""
        print(f"Getting unique value counts for table: {table_name}")
        columns_query = f"""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = '{table_name.split('.')[-1]}'
        """
        columns_result = self.execute_query(columns_query)
        if not columns_result or "results" not in columns_result:
            print("No columns found")
            return {}
            
        columns = [row[0] for row in columns_result["results"]["data"]]
        print(f"Found {len(columns)} columns")
        
        unique_counts = {}
        for column in columns:
            print(f"Counting unique values for column: {column}")
            count_query = f"""
            SELECT COUNT(DISTINCT {column}) 
            FROM (
                SELECT {column}
                FROM {table_name}
                LIMIT 1
            )
            """
            result = self.execute_query(count_query)
            if result and "results" in result:
                unique_counts[column] = result["results"]["data"][0][0]
                print(f"Column {column} has {unique_counts[column]} unique values")
                
        return unique_counts

    def get_unique_values(self, table_name: str, column_name: str) -> List[Any]:
        """Get all unique values for a specific column."""
        print(f"Getting unique values for column {column_name} in table {table_name}")
        query = f"""
        SELECT DISTINCT {column_name} 
        FROM (
            SELECT {column_name}
            FROM {table_name}
            LIMIT 1
        )
        ORDER BY {column_name}
        """
        result = self.execute_query(query)
        
        if result and "results" in result:
            values = [row[0] for row in result["results"]["data"]]
            print(f"Found {len(values)} unique values")
            return values
        print("No unique values found")
        return []

    def analyze_tables(self, tables: List[str]) -> Dict:
        """Analyze all tables and return results."""
        print(f"Starting analysis of {len(tables)} tables")
        results = {}
        for table_name in tables:
            print(f"\nAnalyzing table: {table_name}")
            table_data = {
                "sample_rows": self.get_sample_rows(table_name),
                "column_unique_counts": self.get_column_unique_counts(table_name),
                "columns_with_few_uniques": {}
            }
            
            print("Analyzing columns with few unique values...")
            for column, count in table_data["column_unique_counts"].items():
                if count < 50:
                    print(f"Getting unique values for column {column} (count: {count})")
                    unique_values = self.get_unique_values(table_name, column)
                    table_data["columns_with_few_uniques"][column] = unique_values
            
            results[table_name] = table_data
            print(f"Completed analysis for table: {table_name}")
        
        return results

def main():
    try:
        print("Starting main execution...")
        # Initialize analyzer
        analyzer = DatabricksAnalyzer()
        
        # Get list of tables
        # tables = analyzer.get_tables()
        tables = ['analytics.super.raw_listening_impression_combined_show_level']
        print(f"Found tables: {tables}")
        
        # Analyze tables
        print("Starting table analysis...")
        results = analyzer.analyze_tables(tables)
        
        # Save results to JSON file
        output_file = "databricks_db_structure.txt"
        print(f"Preparing to write results to {output_file}")
        
        # Convert results to formatted text
        print("Converting results to formatted text...")
        text_content = []
        for table_name, table_data in results.items():
            text_content.append(f"Table: {table_name}")
            
            # Add sample row information
            if table_data["sample_rows"]:
                text_content.append("Sample data:")
                for row in table_data["sample_rows"][:3]:  # Show first 3 samples
                    text_content.append(f"  {row}")
            
            # Add column information
            text_content.append("Columns:")
            for column, unique_count in table_data["column_unique_counts"].items():
                text_content.append(f"  - {column}")
                text_content.append(f"    Unique values: {unique_count}")
                
                # Add unique values for columns with few distinct values
                if column in table_data["columns_with_few_uniques"]:
                    values = table_data["columns_with_few_uniques"][column]
                    text_content.append(f"    Possible values: {', '.join(map(str, values))}")
            
            text_content.append("")  # Add blank line between tables
        
        # Write to file
        print(f"Writing results to {output_file}...")
        with open(output_file, 'w') as f:
            f.write("\n".join(text_content))
        
        print(f"Analysis complete. Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main() 