import os
import chromadb
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

load_dotenv()

Settings.llm = Anthropic(model="claude-haiku-4-5-20251001")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

# Set up persistent ChromaDB storage
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("party_planning")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

if chroma_collection.count() == 0:
    print("No existing data found. Running ingestion pipeline...")
    documents = [
        Document(text="A superhero-themed masquerade ball with luxury decor, including gold accents and velvet curtains."),
        Document(text="Hire a professional DJ who can play themed music for superheroes like Batman and Wonder Woman."),
        Document(text="For catering, serve dishes named after superheroes, like 'The Hulk's Green Smoothie' and 'Iron Man's Power Steak.'"),
        Document(text="Decorate with iconic superhero logos and projections of Gotham and other superhero cities around the venue."),
        Document(text="Interactive experiences with VR where guests can engage in superhero simulations or compete in themed games.")
    ]

    # Explicit pipeline: split into chunks -> embed each chunk -> store in ChromaDB
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=100, chunk_overlap=10),
            embed_model,
        ],
        vector_store=vector_store,
    )

    nodes = pipeline.run(documents=documents)
    print(f"Pipeline created {len(nodes)} nodes and stored them in ChromaDB.")

    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
else:
    print(f"Found existing index with {chroma_collection.count()} documents. Loading from disk...")
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

query_engine = index.as_query_engine(response_mode="tree_summarize")
response = query_engine.query("What entertainment ideas are there for a superhero party?")
print("\n--- Response ---")
print(response)