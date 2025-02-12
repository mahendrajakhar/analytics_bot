from openai import AsyncOpenAI
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Tuple
import json
import logging
from api.config import settings
from api.services.sql_service import SQLService
from api.services.graph_service import GraphService
from api.utils.message_utils import (
    send_initial_response,
    update_final_message,
    update_with_processing,
    send_error,
    format_slack_message,
    format_query_result,
    send_excel_file,
    send_graph_status,
    send_graph
)
from api.models import ChatHistory
from api.handlers.history_handlers import save_chat_history, load_chat_history

logger = logging.getLogger(__name__)

class SlackChatService:
    def __init__(self, db: Session, sql_service: SQLService, mongodb_service):
        self.db = db
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.sql_service = sql_service
        self.mongodb_service = mongodb_service
        self.graph_service = GraphService(mongodb_service)
        
        # Define system prompts for different flows
        self.SYSTEM_PROMPTS = {
            "sql": """You are a SQL query assistant. Help users write SQL queries by converting their questions into SQL.
Format the SQL query with sql ``` ... ``` markers.""",

            "graph_step1": """You are a SQL expert. I need help writing a SQL query to answer the user question
according to Available tables and their schemas.

Requirements:
- Generate a SQL query that will return 2-5 sample rows
- Include relevant columns that would help answer the question
- Include any necessary joins between tables
- Keep the query simple - we just need to understand the data structure
- Add comments explaining key parts of the query
- Consider including important aggregations or calculations that might be needed

Please provide SQL query only, wrapped in sql ``` ... ``` markers.""",

            "graph_step2": """I have a user question and sample data from our database. Help me create a complete SQL query and visualization code.

Requirements:

SQL Query Requirements:
- Include all necessary joins
- Handle any required aggregations
- Apply appropriate filters
- Use window functions if needed for trending/comparison
- Optimize for performance with proper indexing hints

Visualization Requirements:
- Use matplotlib or seaborn for the visualization
- Create an appropriate chart type for the data
- Include proper labeling (title, axes, legend)
- Apply professional styling
- Handle data preprocessing if needed
- Consider color schemes for better readability
- Add date formatting if temporal data is involved
- Consider adding trend lines or statistical annotations if relevant

Please format your response with:
1. SQL query wrapped in sql ``` ... ``` markers
2. Python visualization code wrapped in python ``` ... ``` markers""",

            "explain": """You are a SQL query explainer. Break down SQL queries into clear, understandable parts.

Format your response in a clear, structured way:
1. Overall Purpose
2. FROM clause - which tables we're using
3. JOIN conditions - how tables are connected
4. WHERE conditions - what we're filtering
5. GROUP BY - how we're aggregating
6. ORDER BY - how we're sorting
7. Any special functions used""",
        }

    def _prepare_conversation_context(self, user_id: str, new_question: str, flow_type: str, additional_context: str = "") -> List[Dict[str, str]]:
        """Common method to prepare conversation context for all flows"""
        try:
            # Load schema information
            with open("static/db_structure.json", "r") as f:
                db_structure = json.load(f)

            # Get appropriate system prompt
            system_message = self.SYSTEM_PROMPTS.get(flow_type, self.SYSTEM_PROMPTS["sql"])
            
            # Add schema information
            system_message += f"\n\nAvailable tables and their schemas:\n{json.dumps(db_structure, indent=2)}\n"

            # Load user's chat history
            history = load_chat_history(user_id)

            # Prepare conversation messages
            messages = [{"role": "system", "content": system_message}]
            
            # Add history to context
            for chat in history[-5:]:  # Last 5 conversations
                messages.append({"role": "user", "content": chat['question']})
                if chat.get('sql_query'):
                    assistant_response = f"SQL query:\n```sql\n{chat['sql_query']}\n```"
                    messages.append({"role": "assistant", "content": assistant_response})
                elif chat.get('ai_response'):
                    messages.append({"role": "assistant", "content": chat['ai_response']})

            # Add current question
            messages.append({"role": "user", "content": new_question})

            # Add additional context if provided
            if additional_context:
                messages.append({"role": "system", "content": additional_context})

            return messages

        except Exception as e:
            logger.error(f"Error preparing conversation context: {e}")
            return [
                {"role": "system", "content": self.SYSTEM_PROMPTS.get(flow_type, self.SYSTEM_PROMPTS["sql"])},
                {"role": "user", "content": new_question}
            ]

    def _extract_code(self, response: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract both SQL query and visualization code from response"""
        import re
        
        # Extract SQL query
        sql_match = re.search(r"```sql\n(.*?)\n```", response, re.DOTALL)
        sql_query = sql_match.group(1).strip() if sql_match else None
        
        # Extract Python/visualization code
        viz_match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
        viz_code = viz_match.group(1).strip() if viz_match else None
        
        return sql_query, viz_code

    async def process_graph_command(self, request: Dict[str, str], channel_id: str, message_ts: str) -> Dict[str, str]:
        """Process graph command to generate visualization"""
        logger.info(f"Processing graph command for user: {request['user_id']} with question: {request['question']}")
        try:
            # Get AI response for query and visualization
            messages = self._prepare_conversation_context(
                user_id=request["user_id"],
                new_question=request["question"],
                flow_type="graph_step1"
            )

            # Get response from AI
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7
            )

            # Extract both SQL query and visualization code
            ai_response = response.choices[0].message.content
            sql_query, viz_code = self._extract_code(ai_response)
            
            if not sql_query:
                error_msg = "Could not generate SQL query"
                logger.error(error_msg)
                send_error(channel_id, message_ts, {}, error_msg)
                return {"error": error_msg}

            # Execute query
            query_result, error = await self.sql_service.execute_query(sql_query)
            if error:
                error_msg = f"Error executing query: {error}"
                send_error(channel_id, message_ts, {}, error_msg)
                return {"error": error_msg}

            # Update message to show graph generation
            update_with_processing(channel_id, message_ts, "Generating visualization... 📊")

            # Generate graph
            img_buffer, graph_filename = await self.graph_service.generate_and_save_graph(
                query_result,
                request["user_id"],
                request["question"]
            )

            if not img_buffer:
                error_msg = "Failed to generate visualization"
                send_error(channel_id, message_ts, {}, error_msg)
                return {"error": error_msg}

            # Send graph to Slack
            send_graph(channel_id, img_buffer, request["question"])

            # Save to MongoDB and get URL
            graph_url = self.graph_service.save_graph_to_mongodb(
                img_buffer,
                graph_filename,
                request["user_id"],
                request["question"],
                viz_code if viz_code else sql_query  # Use viz_code if available, else sql_query
            )

            if not graph_url:
                logger.error("Failed to save graph to MongoDB")
                # Continue anyway since the graph was already sent to Slack

            # Update final message with both SQL and visualization code
            formatted_response = {
                "text": f"SQL Query:\n```sql\n{sql_query}\n```\n" + 
                       (f"\nVisualization Code:\n```python\n{viz_code}\n```" if viz_code else ""),
                "graph_url": graph_url
            }
            update_final_message(channel_id, message_ts, formatted_response)

            # Save to chat history with complete AI response
            save_chat_history(
                user_id=request["user_id"],
                command="/graph",
                question=request["question"],
                sql_query=sql_query,
                response="Graph generated successfully",
                ai_response=ai_response,  # Store complete AI response
                graph_url=graph_url
            )

            return {
                "text": "Graph generated successfully! ✨",
                "graph_url": graph_url
            }

        except Exception as e:
            error_msg = f"Error processing graph command: {str(e)}"
            logger.error(error_msg)
            send_error(channel_id, message_ts, {}, error_msg)
            return {"error": error_msg}

    async def process_sql_command(self, request: Dict[str, str], channel_id: str, message_ts: str) -> Dict[str, str]:
        """Process SQL command - return query without executing it"""
        try:
            # Update message to show processing
            update_with_processing(channel_id, message_ts, "Generating SQL query... ")

            # Prepare conversation context with sql flow
            messages = self._prepare_conversation_context(
                user_id=request["user_id"],
                new_question=request["question"],
                flow_type="sql"
            )

            # Get response from OpenAI
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7
            )

            assistant_response = response.choices[0].message.content
            sql_query = self._extract_code(assistant_response)[0]

            if not sql_query:
                error_msg = "Could not generate SQL query from the response"
                send_error(channel_id, message_ts, {}, error_msg)
                return {"error": error_msg}

            # Format the response with the SQL query
            formatted_response = {
                "text": f"Here's the SQL query for your question:\n```{sql_query}```"
            }

            # Update the message with the SQL query
            update_final_message(channel_id, message_ts, formatted_response)
        
            # Save to chat history
            save_chat_history(
                user_id=request["user_id"],
                command="/sql",
                question=request["question"],
                sql_query=sql_query,
                ai_response=assistant_response
            )

            return {
                "text": formatted_response.get("text", ""),
                "sql_query": sql_query,
                "error": None
            }

        except Exception as e:
            error_msg = f"Error processing SQL command: {str(e)}"
            logger.error(error_msg)
            send_error(channel_id, message_ts, {}, error_msg)
            return {"error": error_msg}

    async def process_ask_command(self, request: Dict[str, str], channel_id: str, message_ts: str) -> Dict[str, str]:
        """Process ask command - execute query and return results"""
        try:
            # Update message to show processing
            update_with_processing(channel_id, message_ts, "Generating SQL query... ")

            # Prepare conversation context with sql flow
            messages = self._prepare_conversation_context(
                user_id=request["user_id"],
                new_question=request["question"],
                flow_type="sql"
            )

            # Get response from OpenAI
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7
            )

            sql_query = self._extract_code(response.choices[0].message.content)[0]
            if not sql_query:
                error_msg = "Could not generate SQL query from the response"
                send_error(channel_id, message_ts, {}, error_msg)
                return {"error": error_msg}

            # Update message to show query execution
            update_with_processing(channel_id, message_ts, "Executing query... ")

            # Execute query
            query_result, error = await self.sql_service.execute_query(sql_query)
            if error:
                error_msg = f"Error executing query: {error}"
                send_error(channel_id, message_ts, {}, error_msg)
                return {"error": error_msg}

            # Format the response
            formatted_response = format_query_result(query_result)

            if formatted_response["type"] == "excel":
                # Send Excel file for large results
                send_excel_file(channel_id, formatted_response["excel_file"], formatted_response["text"])
            else:
                # Update message with table for smaller results
                update_final_message(channel_id, message_ts, {"text": formatted_response["text"]})

            # Save to chat history
            save_chat_history(
                user_id=request["user_id"],
                command="/ask",
                question=request["question"],
                sql_query=sql_query,
                response="Query executed successfully",
                ai_response=response.choices[0].message.content
            )

            return {
                "text": formatted_response.get("text", ""),
                "error": None
            }

        except Exception as e:
            error_msg = f"Error processing ask command: {str(e)}"
            logger.error(error_msg)
            send_error(channel_id, message_ts, {}, error_msg)
            return {"error": error_msg}
