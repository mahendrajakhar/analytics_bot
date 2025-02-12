import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import logging
import uuid
from typing import Optional, Dict, Any, Tuple
import json
from openai import AsyncOpenAI
from api.services.mongodb_service import MongoDBService
from api.config import settings

logger = logging.getLogger(__name__)

class GraphService:
    def __init__(self, mongodb_service: MongoDBService):
        self.mongodb_service = mongodb_service
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def generate_visualization_code(self, df: pd.DataFrame, question: str) -> Optional[str]:
        """Generate visualization code using AI"""
        try:
            # Create a data description
            data_info = {
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'sample_data': df.head(3).to_dict(orient='records'),
                'shape': df.shape
            }

            # System prompt to guide the AI
            system_prompt = """You are an expert in data visualization using Python matplotlib and seaborn.
            Given a pandas DataFrame description and a visualization request, generate Python code that creates an appropriate visualization.
            The code should:
            1. NOT include plt.figure() or plt.tight_layout() - these are handled separately
            2. Use matplotlib and seaborn only
            3. Include proper titles and labels
            4. Handle data cleaning if needed
            5. Use appropriate color schemes
            6. Use the exact column names provided
            7. Use 'df' as the variable name for the DataFrame
            8. NOT include any imports or plt.show()
            The code will be executed in a context where pandas (pd), matplotlib.pyplot (plt), and seaborn (sns) are already imported."""

            # Create the prompt
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""
                Create visualization code for the following:
                Question: {question}
                
                DataFrame Info:
                {json.dumps(data_info, indent=2)}
                
                Return ONLY the Python code without any explanation.
                """}
            ]

            # Get visualization code from AI
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )

            # Parse AI response
            visualization_code = response.choices[0].message.content.strip()
            logger.info(f"Generated visualization code:\n{visualization_code}")

            # Clean up the code
            visualization_code = visualization_code.replace('```python', '').replace('```', '').strip()
            
            return visualization_code

        except Exception as e:
            logger.error(f"Error generating visualization code: {e}")
            return None

    async def generate_and_save_graph(self, query_result: dict, user_id: str, question: str) -> Tuple[Optional[bytes], Optional[str]]:
        """Generate graph from query result and return image buffer and filename"""
        try:
            # Convert query result to DataFrame
            df = pd.DataFrame(query_result['rows'], columns=query_result['columns'])
            
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"DataFrame columns: {df.columns.tolist()}")
            logger.info(f"DataFrame head:\n{df.head()}")
            
            if df.empty:
                logger.error("Empty DataFrame")
                return None, None

            # Get visualization code
            viz_code = await self.generate_visualization_code(df, question)
            if not viz_code:
                logger.error("Failed to generate visualization code")
                return None, None

            try:
                # Clear any existing plots
                plt.clf()
                plt.close('all')

                # Create a new figure with a white background
                plt.figure(figsize=(10, 6), facecolor='white')
                
                # Create namespace for execution
                namespace = {
                    'plt': plt,
                    'sns': sns,
                    'pd': pd,
                    'df': df
                }
                
                # Execute the visualization code
                exec(viz_code, namespace)
                
                # Ensure tight layout and white background
                plt.tight_layout()
                ax = plt.gca()
                ax.set_facecolor('white')

                # Generate unique filename
                graph_filename = f"graph_{uuid.uuid4()}.png"
                
                # Save to memory buffer
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                img_buffer.seek(0)
                
                # Clean up
                plt.close('all')
                return img_buffer.getvalue(), graph_filename
                
            except Exception as e:
                logger.error(f"Error executing visualization code: {e}")
                plt.close('all')
                return None, None

        except Exception as e:
            logger.error(f"Error generating graph: {e}")
            plt.close('all')
            return None, None

    def save_graph_to_mongodb(self, img_buffer: bytes, graph_filename: str, user_id: str, 
                             question: str, viz_code: str) -> Optional[str]:
        """Save graph to MongoDB and return URL"""
        try:
            # Save to MongoDB
            graph_doc = {
                "user_id": user_id,
                "question": question,
                "image_data": img_buffer,
                "filename": graph_filename,
                "visualization_code": viz_code,
                "content_type": "image/png"
            }
            
            # Save graph to MongoDB
            graph_id = self.mongodb_service.save_graph(graph_doc)
            if not graph_id:
                logger.error("Failed to save graph to MongoDB")
                return None

            # Construct full URL
            base_url = settings.BASE_URL.rstrip('/')
            graph_url = f"{base_url}/api/images/{graph_id}"
            logger.info(f"Generated graph URL: {graph_url}")

            return graph_url

        except Exception as e:
            logger.error(f"Error saving graph to MongoDB: {e}")
            return None
