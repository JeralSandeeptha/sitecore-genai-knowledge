import os;
from dotenv import load_dotenv;

load_dotenv();

envConfig = {
    "QDRANT_URL": os.getenv("QDRANT_URL"),
    "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY"),
    "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
};