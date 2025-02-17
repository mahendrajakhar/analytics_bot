import os
from typing import Optional, List, Dict
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import json
import logging
import glob
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class LangChainService:
    def __init__(self, openai_api_key: str, use_gpu: bool = False, 
                 auto_refresh: bool = True, chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """Initialize LangChain service with configuration options"""
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.vectorstore = None
        self.use_gpu = use_gpu
        self.auto_refresh = auto_refresh
        self.last_refresh = None
        self.refresh_interval = timedelta(hours=1)  # Refresh index if older than 1 hour
        
        self.text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n"
        )
        self.tables_dir = "static/tables"
        self._setup_vector_store()

    def _get_schema_files(self) -> Dict[str, List[str]]:
        """Get all schema files from the tables directory"""
        try:
            if not os.path.exists(self.tables_dir):
                os.makedirs(self.tables_dir)
                logger.warning(f"Created tables directory: {self.tables_dir}")
                return {"text": [], "json": []}

            schema_files = {
                "text": glob.glob(os.path.join(self.tables_dir, "*.txt")),
                "json": glob.glob(os.path.join(self.tables_dir, "*.json"))
            }
            
            logger.info(f"Found {len(schema_files['text'])} text files and {len(schema_files['json'])} JSON files")
            return schema_files
            
        except Exception as e:
            logger.error(f"Error getting schema files: {e}")
            return {"text": [], "json": []}

    def _load_schema_text(self) -> str:
        """Load and combine schema from all files in tables directory"""
        try:
            schema_parts = []
            schema_files = self._get_schema_files()
            
            # Load all text files
            for text_file in schema_files["text"]:
                try:
                    with open(text_file, 'r') as f:
                        content = f.read().strip()
                        if content:
                            schema_parts.append(content)
                            logger.info(f"Loaded schema from text file: {text_file}")
                except Exception as e:
                    logger.error(f"Error loading text file {text_file}: {e}")

            # Load all JSON files
            for json_file in schema_files["json"]:
                try:
                    with open(json_file, 'r') as f:
                        db_structure = json.load(f)
                        
                        # Convert JSON structure to text
                        json_text = []
                        for table_name, table_info in db_structure.items():
                            json_text.append(f"\nTable: {table_name}")
                            if isinstance(table_info, dict) and 'columns' in table_info:
                                for col in table_info['columns']:
                                    if isinstance(col, dict):
                                        col_text = f"  - {col.get('name', 'unknown')}: {col.get('type', 'unknown')}"
                                        if 'description' in col:
                                            col_text += f" - {col['description']}"
                                        json_text.append(col_text)
                        
                        schema_parts.append("\n".join(json_text))
                        logger.info(f"Loaded and converted schema from JSON file: {json_file}")
                except Exception as e:
                    logger.error(f"Error loading JSON file {json_file}: {e}")

            # Combine all schema parts
            combined_schema = "\n\n".join(schema_parts)
            
            if not combined_schema.strip():
                raise ValueError("No schema content found in any file")
                
            return combined_schema

        except Exception as e:
            logger.error(f"Error loading schema text: {e}")
            raise

    def _should_refresh(self) -> bool:
        """Check if index should be refreshed"""
        if not self.auto_refresh:
            return False
            
        if not self.last_refresh:
            return True
            
        return datetime.now() - self.last_refresh > self.refresh_interval

    def _setup_vector_store(self):
        """Set up or load the vector store with GPU support"""
        try:
            index_path = "static/faiss_index"
            
            # Check if we should refresh existing index
            if os.path.exists(index_path) and not self._should_refresh():
                try:
                    # Load existing vector store
                    self.vectorstore = FAISS.load_local(
                        index_path,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    logger.info("Loaded existing vector store")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load existing index: {e}")

            # Create new vector store
            schema_text = self._load_schema_text()
            if not schema_text:
                raise ValueError("No schema text available")

            texts = self.text_splitter.split_text(schema_text)
            docs = [Document(page_content=t) for t in texts]
            
            # Create vector store
            self.vectorstore = FAISS.from_documents(
                docs, 
                self.embeddings,
                normalize_L2=True  # Important for GPU compatibility
            )
            
            # Save index
            os.makedirs("static/faiss_index", exist_ok=True)
            self.vectorstore.save_local(index_path)
            self.last_refresh = datetime.now()
            
            logger.info(
                f"Created new vector store (GPU: {self.use_gpu}, "
                f"Auto-refresh: {self.auto_refresh})"
            )

        except Exception as e:
            logger.error(f"Error setting up vector store: {e}")
            raise

    def get_schema_context(self, question: str, max_chunks: int = 3) -> str:
        """Get relevant schema context for a question with improved error handling"""
        try:
            if not self.vectorstore:
                logger.warning("Vector store not initialized")
                return "Schema information not available."

            # Get relevant documents
            docs = self.vectorstore.similarity_search(
                question, 
                k=max_chunks,
                fetch_k=max_chunks * 4  # Fetch more candidates for better matching
            )
            
            # Combine and format the context
            context_parts = []
            seen_content = set()  # Avoid duplicates
            
            for doc in docs:
                content = doc.page_content.strip()
                if content and content not in seen_content:
                    context_parts.append(content)
                    seen_content.add(content)
            
            if not context_parts:
                return "No relevant schema information found."
            
            return "\n\n".join(context_parts)

        except Exception as e:
            logger.error(f"Error getting schema context: {e}")
            return f"Error retrieving schema context: {str(e)}"

    def refresh_schema(self):
        """Refresh the vector store with latest schema"""
        try:
            schema_text = self._load_schema_text()
            if not schema_text:
                raise ValueError("No schema text available")

            texts = self.text_splitter.split_text(schema_text)
            docs = [Document(page_content=t) for t in texts]
            
            # Create new vector store
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
            self.vectorstore.save_local("static/faiss_index")
            logger.info("Successfully refreshed schema vector store")
            
        except Exception as e:
            logger.error(f"Error refreshing schema: {e}")
            raise
