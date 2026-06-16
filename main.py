#FastAPI Backend for Legal Query System

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import time
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path


# MODELS & SCHEMAS

class QueryRequest(BaseModel):
    """Schema for incoming search queries"""
    query: str
    n_results: int = 5


class SearchResult(BaseModel):
    """Schema for individual search result"""
    section: str
    text: str
    relevance: str
    distance: float


class QueryResponse(BaseModel):
    """Schema for query response"""
    results: List[SearchResult]
    search_time: float
    total_results: int


class HealthResponse(BaseModel):
    """Schema for health check"""
    status: str
    message: str

# GLOBAL VARIABLES (Initialized on startup)


MODEL = None
CHROMA_CLIENT = None
COLLECTION = None


# LIFESPAN CONTEXT MANAGER (Modern FastAPI approach)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app startup and shutdown"""
    global MODEL, CHROMA_CLIENT, COLLECTION
    
    # STARTUP
    try:
        print("🚀 Initializing FastAPI application...")
        
        # Load embedding model
        print("📦 Loading SentenceTransformer model...")
        MODEL = SentenceTransformer("all-mpnet-base-v2")
        print("✅ Model loaded successfully!")
        
        # Initialize ChromaDB
        print("🗄️ Connecting to ChromaDB...")
        CHROMA_CLIENT = chromadb.PersistentClient(path="./chroma_legal_db")
        
        # Get or create collection
        try:
            COLLECTION = CHROMA_CLIENT.get_collection(name="ipc_legal_sections")
            print("✅ ChromaDB collection loaded!")
        except Exception as e:
            print(f"⚠️ Collection not found. Please run upload_chroma.py first.")
            raise Exception("ChromaDB collection 'ipc_legal_sections' not found. Run upload_chroma.py to populate the database.")
        
        print("✅ FastAPI initialization complete!")
        
    except Exception as e:
        print(f"❌ Startup error: {str(e)}")
        raise
    
    # Yield control to the app
    yield
    
    # SHUTDOWN (cleanup if needed)
    print("🛑 Shutting down FastAPI application...")

#fastapi initialization
app = FastAPI(
    title="Legal Query System API",
    description="Search Indian Penal Code (IPC) Sections using semantic search",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ROUTES

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """Serve the HTML frontend"""
    html_path = Path("./legal_query_interface.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Frontend file not found")
    return html_path


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="FastAPI backend is running successfully"
    )


@app.post("/api/search", response_model=QueryResponse)
async def search_legal_database(request: QueryRequest):
    """
    Search the legal database using semantic similarity
    
    Args:
        request: QueryRequest object containing the search query and number of results
        
    Returns:
        QueryResponse with matching IPC sections
    """
    
    # Validation
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if len(request.query.split()) < 3:
        raise HTTPException(status_code=400, detail="Query must contain at least 3 words")
    
    if request.n_results < 1 or request.n_results > 20:
        raise HTTPException(status_code=400, detail="n_results must be between 1 and 20")
    
    try:
        # Start timer
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = MODEL.encode(request.query)
        
        # Query ChromaDB
        results = COLLECTION.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=request.n_results
        )
        
        # Process results
        search_results = []
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]  # ChromaDB returns distances, lower = more similar
        )):
            # Determine relevance based on distance
            if distance < 0.3:
                relevance = "high"
            elif distance < 0.6:
                relevance = "medium"
            else:
                relevance = "low"
            
            search_results.append(SearchResult(
                section=metadata.get("Section", "Unknown"),
                text=doc[:500],  # Limit text to 500 chars for frontend
                relevance=relevance,
                distance=float(distance)
            ))
        
        # Calculate search time
        search_time = time.time() - start_time
        
        return QueryResponse(
            results=search_results,
            search_time=round(search_time, 3),
            total_results=len(search_results)
        )
        
    except Exception as e:
        print(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/api/sections")
async def get_all_sections():
    """
    Get all available IPC sections (for reference)
    
    Returns:
        List of all sections in the database
    """
    try:
        # Get all documents from collection
        all_results = COLLECTION.get()
        
        sections = list(set([meta.get("Section", "Unknown") for meta in all_results["metadatas"]]))
        sections.sort()
        
        return {
            "total_sections": len(sections),
            "sections": sections
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch sections: {str(e)}")


@app.get("/api/section/{section_id}")
async def get_section_details(section_id: str):
    """
    Get detailed information about a specific IPC section
    
    Args:
        section_id: The IPC section number (e.g., "378")
        
    Returns:
        Full text and details of the section
    """
    try:
        # Query for specific section
        results = COLLECTION.get(
            where={"Section": section_id}
        )
        
        if not results["documents"]:
            raise HTTPException(status_code=404, detail=f"Section {section_id} not found")
        
        return {
            "section": section_id,
            "text": results["documents"][0],
            "metadata": results["metadatas"][0]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch section: {str(e)}")


@app.post("/api/feedback")
async def submit_feedback(feedback: dict):
    """
    Submit user feedback (can be extended to store in database)
    
    Args:
        feedback: Dictionary containing user feedback
        
    Returns:
        Confirmation message
    """
    try:
        # In a production system, save this to a database
        print(f"Feedback received: {feedback}")
        
        return {
            "status": "success",
            "message": "Thank you for your feedback!"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to submit feedback")

#error  handling
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return {
        "status": "error",
        "detail": exc.detail,
        "status_code": exc.status_code
    }

# RUN SERVER

if __name__ == "__main__":
    import uvicorn
    
    print("""Legal Query System - FastAPI Backend                                     
       Access at: http://localhost:8000              
       API Docs: http://localhost:8000/docs  """)        
                                                     
    
    # Use import string to enable reload and workers
    uvicorn.run(
        "main:app",  # Import string format
        host="0.0.0.0",
        port=8000,
        reload=True  # Set to False in production
    )
