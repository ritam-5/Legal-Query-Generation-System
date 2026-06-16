from sentence_transformers import SentenceTransformer
import chromadb
from data_preprocess import df

# Load a lightweight embedding model
model = SentenceTransformer("all-mpnet-base-v2")

# Generate embeddings for legal text
embeddings = model.encode(df["combined_text"].tolist(), show_progress_bar=True)

print(f"✅ Embeddings generated! Shape: {embeddings.shape}")

# Initialize ChromaDB client (PersistentClient for saving to disk)
chroma_client = chromadb.PersistentClient(path="./chroma_legal_db")

# Delete the collection if it exists to reset the embedding dimension
#try:
    #chroma_client.delete_collection(name="ipc_legal_sections")
#except:
    #pass # Collection doesn't exist, no need to delete

# Create / get a collection
collection = chroma_client.get_or_create_collection(name="ipc_legal_sections")

# Add data
collection.add(
    ids=[f"id_{i}" for i in range(len(df))],
    embeddings=embeddings.tolist(),
    documents=df["combined_text"].tolist(),
    metadatas=[{"Section": sec} for sec in df["Section"].tolist()]
)

print("✅ Data successfully inserted into ChromaDB!")