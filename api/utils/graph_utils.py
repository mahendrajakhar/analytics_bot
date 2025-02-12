import matplotlib.pyplot as plt
import io
import ast
from config import logger

def validate_graph_code(code: str):
    """Basic validation of graph code for security"""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            # Check for potentially dangerous operations
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if any(name.name != 'matplotlib.pyplot' for name in node.names):
                    raise ValueError("Only matplotlib.pyplot imports are allowed")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['eval', 'exec', 'compile']:
                        raise ValueError("Potentially dangerous operation detected")
        return True
    except Exception as e:
        logger.error(f"Graph code validation error: {e}")
        return False

def generate_graph(graph_code: str, sql_result: dict = None):
    """Generate graph from code and return image buffer and filename"""
    try:
        # Clear any existing plots
        plt.clf()
        
        # Create a new figure
        plt.figure(figsize=(10, 6))
        
        # Execute the graph code with the SQL result data
        if sql_result:
            local_vars = {'plt': plt, 'data': sql_result}
            exec(graph_code, {}, local_vars)
        else:
            local_vars = {'plt': plt}
            exec(graph_code, {}, local_vars)
        
        # Add grid and tight layout
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save plot to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        # Generate unique filename
        filename = f"graph_{hash(graph_code)}.png"
        
        return buf, filename
    except Exception as e:
        logger.error(f"Error generating graph: {e}")
        raise
    finally:
        plt.close('all')
