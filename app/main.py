from fastapi import FastAPI, Query;
from .config.qdrant_client import get_qdrant_client;
from langchain_community.document_loaders import GithubFileLoader;
from dotenv import load_dotenv;
from .config.envConfig import envConfig;
from langchain_community.document_loaders import JSONLoader;
from pathlib import Path;
import json;
from langchain_text_splitters import RecursiveCharacterTextSplitter;
from qdrant_client.http.models import PointStruct, VectorParams, Distance;
from langchain_openai import OpenAIEmbeddings;
from langchain_huggingface import HuggingFaceEmbeddings;
from qdrant_client.models import ScoredPoint;
from fastapi.middleware.cors import CORSMiddleware;

# Load variables from .env file
load_dotenv();

origins = [
    "http://localhost:5000",
    "http://localhost:5173",
];

app = FastAPI(title="Vector API");

# Setup Cors Permissions
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
);

qdrant = get_qdrant_client();

# Health Checking
@app.get("/api/v1/health")
def health():
    return {"statusCode": "200", "message": "Knowledge Service is running", "data": "Knowledge Service is running" };

# This is for OpenAI emmbeddings
@app.post("/update_vector")
def upsert_vector():
    # Dummy Data File Path
    file_path = Path(__file__).parent / "data" / "components.json";

    print(file_path);

    if not file_path.exists():
        return {"message": "File not found", "path": str(file_path)};

    # Load JSON directly
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f);

    components = data.get("components", [])

    if not components:
        return {"message": "No components found", "data": []}

    print(data);

    # Prepare text chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    );

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector = embeddings.embed_query(text_splitter);

    # if not data:
    #     return {"message": "No content found in JSON", "data": [] };

    # all_points = [];
    # for idx, comp in enumerate(components):
    #     content = comp.get("content", "")
    #     metadata = comp.get("metadata", {})

    #     chunks = text_splitter.split_text(content)

    #     # Convert each chunk to a vector and prepare for Qdrant
    #     for i, chunk in enumerate(chunks):
    #         # Use your embedding model (OpenAI in this example)
    #         embedding = OpenAIEmbeddings(
    #             model="text-embedding-3-large",
    #             api_key=envConfig["OPENAI_API_KEY"]
    #         ).embed_query(chunk);

    #         point = PointStruct(
    #             id=f"{idx}_{i}",
    #             vector=embedding,
    #             payload={
    #                 "metadata": metadata,
    #                 "text": chunk
    #             }
    #         );
    #         all_points.append(point);

    # if "components_collection" not in [c.name for c in qdrant.get_collections().collections]:
    #     qdrant.recreate_collection(
    #         collection_name="components_collection",
    #         vector_size=len(all_points[0].vector),
    #         distance="Cosine"
    #     );

    # Upsert points into Qdrant
    # qdrant.upsert(
    #     collection_name="components_collection",
    #     points=all_points
    # );

    qdrant.upsert(
        collection_name="components_collection",
        points=vector
    );

    return {"message": "Vector stored", "stored_points": len(vector) };
    # return {"message": "Vector stored", "stored_points": len(all_points) };

# Create components
@app.post("/update_components")
def updatedb():
    file_path = Path(__file__).parent / "data" / "components.json"

    if not file_path.exists():
        return {"message": "File not found", "path": str(file_path)}

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    components = data.get("data", [])
    if not components:
        return {"message": "No data found", "data": []}

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    all_points = []

    for idx, comp in enumerate(components):
        content = comp.get("content", "")
        metadata = comp.get("metadata", {})

        chunks = text_splitter.split_text(content)

        for i, chunk in enumerate(chunks):
            vector = embeddings.embed_query(chunk)
            point_id = idx * 1_000_000 + i;
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": chunk,
                    "metadata": metadata
                }
            );
            all_points.append(point)

    # Create collection if it doesn't exist
    if "components_collection" not in [c.name for c in qdrant.get_collections().collections]:
        qdrant.recreate_collection(
            collection_name="components_collection",
            vectors_config=VectorParams(
                size=len(all_points[0].vector),
                distance=Distance.COSINE
            )
        );

    print(envConfig["QDRANT_URL"]);

    qdrant.upsert(collection_name="components_collection", points=all_points)

    return {"message": "Vector components stored", "stored_points": len(all_points)}

# Create stories
@app.post("/update_stories")
def updatedb():
    file_path = Path(__file__).parent / "data" / "stories.json"

    if not file_path.exists():
        return {"message": "File not found", "path": str(file_path)}

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    components = data.get("data", [])
    if not components:
        return {"message": "No data found", "data": []}

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    all_points = []

    for idx, comp in enumerate(components):
        content = comp.get("content", "")
        metadata = comp.get("metadata", {})

        chunks = text_splitter.split_text(content)

        for i, chunk in enumerate(chunks):
            vector = embeddings.embed_query(chunk)
            point_id = idx * 1_000_000 + i;
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": chunk,
                    "metadata": metadata
                }
            );
            all_points.append(point)

    # Create collection if it doesn't exist
    if "stories_collection" not in [c.name for c in qdrant.get_collections().collections]:
        qdrant.recreate_collection(
            collection_name="stories_collection",
            vectors_config=VectorParams(
                size=len(all_points[0].vector),
                distance=Distance.COSINE
            )
        );

    print(envConfig["QDRANT_URL"]);

    qdrant.upsert(collection_name="stories_collection", points=all_points)

    return {"message": "Vector stories stored", "stored_points": len(all_points)}

# Create ts
@app.post("/update_ts")
def updatedb():
    file_path = Path(__file__).parent / "data" / "ts.json"

    if not file_path.exists():
        return {"message": "File not found", "path": str(file_path)}

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    components = data.get("data", [])
    if not components:
        return {"message": "No data found", "data": []}

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    all_points = []

    for idx, comp in enumerate(components):
        content = comp.get("content", "")
        metadata = comp.get("metadata", {})

        chunks = text_splitter.split_text(content)

        for i, chunk in enumerate(chunks):
            vector = embeddings.embed_query(chunk)
            point_id = idx * 1_000_000 + i;
            point = PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": chunk,
                    "metadata": metadata
                }
            );
            all_points.append(point)

    # Create collection if it doesn't exist
    if "ts_collection" not in [c.name for c in qdrant.get_collections().collections]:
        qdrant.recreate_collection(
            collection_name="ts_collection",
            vectors_config=VectorParams(
                size=len(all_points[0].vector),
                distance=Distance.COSINE
            )
        );

    print(envConfig["QDRANT_URL"]);

    qdrant.upsert(collection_name="ts_collection", points=all_points)

    return {"message": "Vector ts stored", "stored_points": len(all_points)}

# Load components
@app.post("/load_components")
def upsert_vector():
    loader = GithubFileLoader(
        repo="JeralSandeeptha/Sitecore.Demo.SitecoreAI.IndustryVerticals.SiteTemplates-GenAI",
        branch="main",
        access_token=envConfig["ACCESS_TOKEN"],
        github_api_url="https://api.github.com",
        file_filter=lambda path: (
            path.startswith("industry-verticals/")
            and path.endswith(".tsx")
            and not path.endswith((".stories.tsx", ".ts"))
        )
    );
    docs = loader.load();
    print(loader);
    print(docs);
    return {"message": "Vectors Loaded", "data": [
        {
            "content": d.page_content,
            "metadata": d.metadata,
        }
        for d in docs
    ] };

# Load stories
@app.post("/load_stories")
def upsert_vector():
    loader = GithubFileLoader(
        repo="JeralSandeeptha/Sitecore.Demo.SitecoreAI.IndustryVerticals.SiteTemplates-GenAI",
        branch="main",
        access_token=envConfig["ACCESS_TOKEN"],
        github_api_url="https://api.github.com",
        file_filter=lambda path: path.startswith("industry-verticals/") and path.endswith((".stories.tsx"))
    );
    docs = loader.load();
    print(loader);
    print(docs);
    return {"message": "Vectors Loaded", "data": [
        {
            "content": d.page_content,
            "metadata": d.metadata,
        }
        for d in docs
    ] };

# Load ts files
@app.post("/load_ts")
def upsert_vector():
    loader = GithubFileLoader(
        repo="JeralSandeeptha/Sitecore.Demo.SitecoreAI.IndustryVerticals.SiteTemplates-GenAI",
        branch="main",
        access_token=envConfig["ACCESS_TOKEN"],
        github_api_url="https://api.github.com",
        file_filter=lambda path: path.startswith("industry-verticals/") and path.endswith((".ts"))
    );
    docs = loader.load();
    print(loader);
    print(docs);
    return {"message": "Vectors ts Loaded", "data": [
        {
            "content": d.page_content,
            "metadata": d.metadata,
        }
        for d in docs
    ] };

# Get data
@app.get("/api/v1/search")
def search_vector(query: str = Query(..., description="Search query")):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    );

    # 1. Embed the query
    query_vector = embeddings.embed_query(query);

    # 2. Search in Qdrant
    points = qdrant.query_points(
        collection_name="components_collection",
        query=query_vector,
        with_payload=True
    );

    print(points);

    # 3. Format results
    # Normalize safely
    normalized_points = [];
    for item in points:
        if isinstance(item, ScoredPoint):
            normalized_points.append(item)
        elif isinstance(item, tuple):
            for elem in item:
                if isinstance(elem, ScoredPoint):
                    normalized_points.append(elem)
                    break

    simplifiedPoints = [
        {
            "id": point.id,
            "score": point.score,
            "path": point.payload.get("metadata", {}).get("path") if point.payload else None,
            "text": point.payload.get("text") if point.payload else None
        }
        for point in normalized_points
    ];

    return { "statusCode": 200, "message": "Vector retrieved", "data": points };

