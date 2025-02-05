import mysql.connector
import json

# Database connection configuration
config = {
    'user': 'root',
    'password': 'mahensql',
    'host': 'localhost',
    'port': 3306
}

# Connect to MySQL server
connection = mysql.connector.connect(**config)
cursor = connection.cursor()

# Fetch all database names
cursor.execute("SHOW DATABASES")
databases = [db[0] for db in cursor.fetchall()]

# Data structure to store database details
db_structure = {
    "structure": {},
    "schema": {}
}

# Loop through each database to get tables and schemas
for db in databases:
    if db in ['information_schema', 'mysql', 'performance_schema', 'sys']:
        continue  # Skip system databases

    db_structure["structure"][db] = []
    db_structure["schema"][db] = {}

    # Switch to the database
    cursor.execute(f"USE `{db}`")

    # Get all tables in the current database
    cursor.execute("SHOW TABLES")
    tables = [table[0] for table in cursor.fetchall()]

    for table in tables:
        # Add table info to structure
        db_structure["structure"][db].append({"tablename": table})

        # Get schema for each table
        cursor.execute(f"DESCRIBE `{table}`")
        schema_details = cursor.fetchall()

        # Prepare schema data
        schema_info = []
        for column in schema_details:
            schema_info.append({
                "column_name": column[0],
                "data_type": column[1],
                "is_nullable": column[2],
                "key": column[3],
                "default": column[4],
                "extra": column[5]
            })

        # Add schema details to db_structure
        db_structure["schema"][db][table] = schema_info

# Save the data into a JSON file
with open('static/db_structure.json', 'w') as json_file:
    json.dump(db_structure, json_file, indent=4)

# Close the database connection
cursor.close()
connection.close()

print("Database structure and schema exported to 'static/db_structure.json'")