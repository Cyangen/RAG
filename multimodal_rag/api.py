from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import uvicorn
from rag import get_rag_pipeline  # Import RAG function




# Initialize FastAPI
app = FastAPI()

# Load RAG model
rag_chain, _ = get_rag_pipeline()  # Unpack the tuple

# Define input model
class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_model(request: QueryRequest):
    """API endpoint to handle user queries."""
    try:
        response = ""
        for chunk in rag_chain.stream(request.question):
            response += chunk.content
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)