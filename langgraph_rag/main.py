import logging
from typing import Literal
from fastapi.responses import StreamingResponse
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import config
import graph

logger = logging.getLogger(__name__)

app = FastAPI(
    title="GADIO",
    version="0.1.0"
)

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]

class ChatResponse(BaseModel):
    message: ChatMessage
    done: bool

@app.get("/")
async def root():
    return "This is GADIO!"

@app.post("/api/chat")
async def chat(req: ChatRequest):
    logger.info(f"Received request: {req}")
    last_msg = req.messages[-1] # Assume the last message is the user's message

    async def response_stream():
        async for (msg_chunk, meta) in graph.workflow.astream({
            "question": last_msg
        }, stream_mode="messages"):
            msg_content = msg_chunk.content # type: ignore
            message = ChatMessage(role="assistant", content=msg_content)
            yield ChatResponse(message=message, done=False).model_dump_json()
        yield ChatResponse(message=message, done=True).model_dump_json()

    return StreamingResponse(response_stream(), media_type="application/json")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=config.PORT)
