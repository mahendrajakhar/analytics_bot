from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from config import settings
import logging

logger = logging.getLogger(__name__)

class SQLService:
    def __init__(self):
        # Enhanced MySQL connection with connection pooling
        self.engine = create_engine(
            settings.DATABASE_URL,
            pool_size=10,  # Adjust based on your needs
            max_overflow=20,
            pool_pre_ping=True  # Added for connection health check
        )

    async def execute_query(self, query: str):
        try:
            # Basic SQL injection prevention
            if any(keyword.lower() in query.lower() 
                  for keyword in ['drop', 'truncate', 'delete', 'update', 'insert']):
                raise ValueError("Data modification queries are not allowed")

            with self.engine.connect() as connection:
                result = connection.execute(text(query))
                columns = result.keys()
                rows = result.fetchall()
                return {"columns": columns, "rows": [dict(zip(columns, row)) for row in rows]}, None
        except SQLAlchemyError as e:
            return None, str(e)
        except Exception as e:
            return None, str(e)

    async def get_table_schema(self):
        schema_query = """
        SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, COLUMN_DEFAULT, IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
        ORDER BY TABLE_NAME, ORDINAL_POSITION;
        """
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(schema_query))
                schema_info = {}

                for row in result:
                    table_name = row.TABLE_NAME
                    if table_name not in schema_info:
                        schema_info[table_name] = []

                    schema_info[table_name].append({
                        "column_name": row.COLUMN_NAME,
                        "data_type": row.DATA_TYPE,
                        "is_nullable": row.IS_NULLABLE,
                        "default": row.COLUMN_DEFAULT
                    })

                return schema_info
        except Exception as e:
            # logger.error(f"Error fetching schema: {e}")
            return {} 