import pandas as pd
from sqlalchemy import create_engine
from config import settings
import json
import os
import logging
from sqlalchemy import text  # Add this import at the top


# MySQL Connection
engine = create_engine(settings.DATABASE_URL)

# JSON Structure Update Path
db_structure_path = './static/db_structure.json'

def load_db_structure():
    if os.path.exists(db_structure_path):
        with open(db_structure_path, 'r') as file:
            return json.load(file)
    return {"structure": {}, "schema": {}}

def save_db_structure(db_structure):
    with open(db_structure_path, 'w') as file:
        json.dump(db_structure, file, indent=4)

def import_csv(file_path, table_name, database_name='datasets_db'):
    df = pd.read_csv(file_path)

    try:
        # Import CSV to MySQL
        df.to_sql(table_name, con=engine, if_exists='replace', index=False, schema='datasets_db')
    except Exception as e:
        logging.error(f"Error importing CSV to MySQL: {e}")
        raise

    # Get Schema Information
    schema_info = []
    with engine.connect() as connection:
        result = connection.execute(text(f"DESCRIBE {table_name}"))
        for row in result:
            schema_info.append({
                "column_name": row[0],
                "data_type": row[1],
                "is_nullable": row[2],
                "key": row[3],
                "default": row[4],
                "extra": row[5]
            })

    # Update JSON Structure
    db_structure = load_db_structure()

    if database_name not in db_structure['structure']:
        db_structure['structure'][database_name] = []

    db_structure['structure'][database_name].append({"tablename": table_name, "description": f"Dataset imported from {file_path}"})

    if database_name not in db_structure['schema']:
        db_structure['schema'][database_name] = {}

    db_structure['schema'][database_name][table_name] = schema_info

    save_db_structure(db_structure)

    print(f"✅ Successfully imported '{file_path}' into MySQL table '{table_name}'.")

if __name__ == "__main__":
    files = {
        "spotify_tracks.csv": "spotify",
        "netflix1.csv": "netflix",
        "Games.csv": "nba_games",
        "athlete_events.csv": "olympics_athletes"
    }

    for file_path, table_name in files.items():
        import_csv('csv-files/'+file_path, table_name)