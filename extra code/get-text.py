import os

def get_code_from_directory(root_dir, output_file):
    # Define directories and file extensions to ignore
    ignore_dirs = {'venv', '__pycache__','.git','csv-files'
                    # add more directories to ignore here (, 'dir_name')
                   
                   
                   }
    ignore_files = {'.DS_Store','.csv', '.md', '.gitkeep', '.yml', '.txt',
                    '.cfg', '__init__.py','get-text.py','pytest.ini',
                    'Makefile','pyproject.toml','.gitignore','.env',
                    'check_db_connection.py','code-to-get-mysql-tables-and-schema.py',
                    'import-csv-to-mysql-dataset.py', '.cursorignore', 'styles.css'
                    ,'db_structure.json','requirements.txt','chat_history.json'
                    
                    # add more files to ignore here (, 'file_name')
                    }

    # Open the output file where we'll store the results
    with open(output_file, 'w') as outfile:
        # Walk through the directory structure
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Remove ignored directories from traversal
            dirnames[:] = [d for d in dirnames if d not in ignore_dirs]

            for filename in filenames:
                # Ignore specific file types
                if any(filename.endswith(ext) for ext in ignore_files):
                    continue

                file_path = os.path.join(dirpath, filename)

                try:
                    # Read the content of the file
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()

                    # Write the file name and its content to the output file
                    outfile.write(f"File: {file_path}\n")
                    outfile.write("=" * 50 + "\n")
                    outfile.write(content)
                    outfile.write("\n" + "=" * 50 + "\n\n")

                except Exception as e:
                    # Handle exceptions like unreadable files
                    outfile.write(f"Error reading file {file_path}: {str(e)}\n")
                    outfile.write("=" * 50 + "\n\n")


# Specify the root directory and the output text file
root_directory = './'  # Current directory or specify your target directory
output_text_file = 'all_code.txt'

# Call the function
get_code_from_directory(root_directory, output_text_file)

print(f"All code has been written to {output_text_file}")