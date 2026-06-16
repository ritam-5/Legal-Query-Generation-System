from sentence_transformers import SentenceTransformer
import chromadb

# Initialize ChromaDB client (PersistentClient for loading from disk)
chroma_client = chromadb.PersistentClient(path="./chroma_legal_db")

# Get the collection
collection = chroma_client.get_collection(name="ipc_legal_sections")

# Load the model
print("Loading model...")
model = SentenceTransformer("all-mpnet-base-v2")
print("✅ Model loaded successfully!\n")

# Interactive query loop
while True:
    query = input("🔍 Enter your legal query (or type 'exit' to quit): ").strip()
    
    if query.lower() == 'exit':
        print("Exiting the query interface.")
        break
    
    if not query:
        print("⚠️ Please enter a valid query.\n")
        continue
    
    # Generate embedding for the query
    query_embedding = model.encode(query)
    
    # Query the database
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5
    )
    
    print("\n🔍 Top matching IPC sections:")
    print("=" * 80)
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
        print(f"{i}. Section: {meta['Section']}")
        print(f"   {doc[:200]}...")
        print()
    print("=" * 80 + "\n")