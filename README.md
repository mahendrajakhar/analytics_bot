# Natural Language to SQL Chatbot

A FastAPI-based chatbot that converts natural language questions into SQL queries, executes them, and can generate visualizations using matplotlib. The chatbot uses OpenAI's GPT model to understand queries and generate appropriate SQL and graph code.

## 🚀 Features

- Natural language to SQL query conversion
- Interactive chat interface
- Data visualization capabilities
- Chat history management
- Support for multiple database tables
- Command system (/ask, /help, /summarise, /improve)

## �� Project Structure

project_root/
├── main.py # FastAPI application entry point
├── config.py # Configuration settings
├── database.py # Database connection setup
├── models.py # SQLAlchemy models (currently unused)
├── schemas.py # Pydantic models for request/response
├── services/
│ ├── chat_service.py # Main chat processing logic
│ └── sql_service.py # SQL query execution handling
├── utils/
│ └── graph_utils.py # Graph generation utilities
├── static/
│ ├── chat_history.json # Chat history storage
│ ├── db_structure.json # Database schema information
│ └── styles.css # UI styling
└── templates/
├── index.html # Main chat interface
└── help.html # Help page



## 💡 How It Works

1. **User Input Processing**
   - User sends a question through the chat interface
   - The question is processed by the chat service
   - System adds context about available database tables and schema

2. **Query Generation**
   - OpenAI's GPT model generates:
     - SQL query based on the question
     - Optional visualization code if needed
   - Supports various commands (/ask, /summarise, /improve)

3. **SQL Execution**
   - Generated SQL is validated and executed
   - Results are formatted into a readable table
   - Error handling for invalid queries

4. **Visualization**
   - If graph code is generated, matplotlib creates visualizations
   - Graphs are saved as images and served statically
   - Supports various types of plots based on data

5. **History Management**
   - Chat history stored in JSON format
   - Maintains context for better responses
   - Limits history to last 50 conversations

## 🔧 Configuration

1. Create a `.env` file with:

# .env

OPENAI_API_KEY=key
DATABASE_URL=mysql+pymysql://root:mahensql@localhost:3306/datasets_db # for running localy else add your own
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_SIGNING_SECRET=your-signing-secret
SLACK_VERIFICATION_TOKEN=your-verification-token


2. Database structure is defined in `static/db_structure.json`

## 🚀 Getting Started

1. Install dependencies:
bash
pip install -r requirements.txt

2. Run the application:
bash
uvicorn main:app --reload

3. Access the chat interface at `http://localhost:8000`

## 📝 Available Commands

- `/help` - Show available commands
- `/ask` - Generate SQL query and optional graph
- `/summarise` - Summarize data or graph
- `/improve` - Improve existing graph code

## 🔍 Example Usage

1. Natural Language Query: Show me the top 10 most popular Netflix shows

2. Generated SQL:
```SELECT title, type, rating, duration
FROM netflix
ORDER BY date_added DESC
LIMIT 10;```

3. Visualization Request: Create a bar chart of Olympic medals by country



## 🛠️ Technical Details

- **Framework**: FastAPI
- **Database**: MySQL
- **ORM**: SQLAlchemy
- **AI Model**: OpenAI GPT
- **Visualization**: Matplotlib
- **Frontend**: HTML/CSS with Tailwind CSS

## 🔒 Security Features

- SQL injection prevention
- Graph code validation
- Input sanitization
- Error handling and logging

## 📊 Supported Databases

Currently supports tables in datasets_db:
- nba_games
- netflix
- olympics_athletes
- spotify


for running it on openweb url 
1. Install ngrok
brew install ngrok/ngrok/ngrok
2. Sign Up and Get Auth Token (Optional but Recommended)
ngrok config add-authtoken YOUR_AUTH_TOKEN
3. Run Your FastAPI App Locally
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
4. Expose Your Localhost Using ngrok
ngrok http 8000

```Bonus Tips
Keep the ngrok terminal open while testing.
If you want to keep the same URL every time:
only for paid users
ngrok http --region=us --subdomain=customname 8000```







clear && uvicorn main:app --host 0.0.0.0 --port 8000 --reload



TODO fiass indexing you have to add text files in static/tables folder all the .txt and .json files will be added in indexing 