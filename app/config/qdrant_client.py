from qdrant_client import QdrantClient;
from .envConfig import envConfig;

def get_qdrant_client():
    return QdrantClient(
        host="host.docker.internal",
        port=6333
    );