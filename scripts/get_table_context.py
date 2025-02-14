import json
import pandas as pd
from sqlalchemy import create_engine, text
import os
from typing import List, Dict, Any

class DatabaseAnalyzer:
    def __init__(self, connection_string: str):
        """Initialize the database analyzer with a connection string."""
        self.engine = create_engine(connection_string)

    def get_tables(self) -> List[str]:
        """Get list of all tables in the database."""
        with self.engine.connect() as conn:
            query = text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            result = conn.execute(query)
            return [row[0] for row in result]

    def get_sample_rows(self, table_name: str, num_rows: int = 10) -> List[Dict]:
        """Get sample rows from a table."""
        query = text(f"SELECT * FROM {table_name} LIMIT {num_rows}")
        df = pd.read_sql(query, self.engine)
        return df.to_dict(orient='records')

    def get_column_unique_counts(self, table_name: str) -> Dict[str, int]:
        """Get count of unique values for each column in a table."""
        with self.engine.connect() as conn:
            # Get columns first
            query = text(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
            """)
            columns = [row[0] for row in conn.execute(query)]
            
            # Get unique counts for each column
            unique_counts = {}
            for column in columns:
                query = text(f"SELECT COUNT(DISTINCT {column}) FROM {table_name}")
                result = conn.execute(query)
                count = result.scalar()
                unique_counts[column] = count
                
            return unique_counts

    def get_unique_values(self, table_name: str, column_name: str) -> List[Any]:
        """Get all unique values for a specific column."""
        query = text(f"SELECT DISTINCT {column_name} FROM {table_name} ORDER BY {column_name}")
        df = pd.read_sql(query, self.engine)
        return df[column_name].tolist()

    def analyze_tables(self, tables: List[str]) -> Dict:
        """Analyze all tables and return results."""
        results = {}
        for table_name in tables:
            print(f"Analyzing table: {table_name}")
            table_data = {
                "sample_rows": self.get_sample_rows(table_name),
                "column_unique_counts": self.get_column_unique_counts(table_name),
                "columns_with_few_uniques": {}
            }
            
            # Get unique values for columns with less than 50 unique values
            for column, count in table_data["column_unique_counts"].items():
                if count < 50:
                    unique_values = self.get_unique_values(table_name, column)
                    table_data["columns_with_few_uniques"][column] = unique_values
            
            results[table_name] = table_data
        
        return results

def main():
    # Connection string - modify as needed
    connection_string = "mysql+pymysql://root:mahensql@localhost:3306/datasets_db"
    
    try:
        # Initialize analyzer
        analyzer = DatabaseAnalyzer(connection_string)
        
        # List of tables to analyze
        tables = ["spotify", "netflix", "nba_games", "olympics_athletes"]
        
        # Analyze tables
        results = analyzer.analyze_tables(tables)
        
        # Save results to JSON file
        output_file = "tables_data_for_context.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Analysis complete. Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()