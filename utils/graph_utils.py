import matplotlib.pyplot as plt
import uuid
import os
import io
import base64
from bson.binary import Binary
from services.mongodb_service import MongoDBService  # Ensure MongoDB service is available

mongodb_service = MongoDBService()

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

def generate_graph(graph_code: str, sql_result: dict = None) -> str:
    """Generate graph from code and save it"""
    try:
        # Clear any existing plots
        plt.clf()
        plt.close('all')
        
        # Create a new figure with specific size
        plt.figure(figsize=(10, 6))
        
        # Create a local namespace for execution
        local_namespace = {
            'plt': plt,
            'sql_result': sql_result
        }
        
        # Execute the graph code in the local namespace
        exec(graph_code, globals(), local_namespace)
        
        # Ensure the plot is fully rendered
        plt.tight_layout()
        
        # Generate unique filename
        graph_filename = f"{uuid.uuid4()}.png"
        graph_path = os.path.join("static", "graphs", graph_filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        
        # Save to local file system with high DPI
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        
        # Save to MongoDB
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        
        # Store in MongoDB
        mongodb_service.save_image(Binary(img_buffer.read()), content_type="image/png")
        
        # Clean up
        plt.close('all')
        
        return f"/static/graphs/{graph_filename}"
        
    except Exception as e:
        print(f"Graph generation error: {e}")
        plt.close('all')  # Ensure cleanup on error
        return None