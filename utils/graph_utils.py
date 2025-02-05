import matplotlib.pyplot as plt
import uuid
import os
from typing import Dict, Optional
import ast

def validate_graph_code(code: str) -> bool:
    """Basic validation of graph code for security"""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            # Prevent potentially dangerous operations
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if node.names[0].name not in ['matplotlib.pyplot', 'plt']:
                    return False
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile']:
                        return False
        return True
    except:
        return False

def generate_graph(graph_code: str, sql_result: Optional[Dict] = None) -> str:
    try:
        if not validate_graph_code(graph_code):
            raise ValueError("Invalid or potentially unsafe graph code")

        # Make SQL result available to the graph code
        if sql_result:
            locals()['sql_result'] = sql_result
            
        exec(graph_code)  # Note: Code is validated above
        graph_filename = f"{uuid.uuid4()}.png"
        graph_path = os.path.join("static", graph_filename)
        plt.savefig(graph_path)
        plt.close()
        return f"/static/{graph_filename}"
    except Exception as e:
        print(f"Graph generation error: {e}")  # For debugging
        return None 