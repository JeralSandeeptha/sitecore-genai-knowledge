import os;
from dotenv import load_dotenv;

load_dotenv();

envConfig = {
    "QDRANT_URL": os.getenv("QDRANT_URL"),
    "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY"),
    "ACCESS_TOKEN": os.getenv("ACCESS_TOKEN"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
};