import json
import httpx
from typing import Iterator, Literal, Type, TypeVar, Union
from pydantic import BaseModel, Field

T = TypeVar('T', bound=BaseModel)

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatResponse(BaseModel):
    message: ChatMessage
    done: bool

class Pipe:
    class Valves(BaseModel):
        GADIO_URL: str = Field(title="GADIO URL", description="The URL of the GADIO backend", default="http://localhost:8000")

    def __init__(self):
        self.valves = self.Valves()


    def pipe(self, body: dict) -> Union[str, Iterator[str]]:
        base_url = self.valves.GADIO_URL
        api_url = f"{base_url}/api/chat"

        messages = body["messages"]
        request_json = { "messages": messages }
        
        try:
            with httpx.Client(timeout=60.0) as client:
                with client.stream("POST", api_url, json=request_json) as resp:
                    resp.raise_for_status()
                    for chat_resp in self._process_streaming_json_response(resp, ChatResponse):
                        yield chat_resp.message.content

        except Exception as e:
            yield f"Error connecting to GADIO: {str(e)}"

    def _process_streaming_json_response(self, resp: httpx.Response, obj_type: Type[T]) -> Iterator[T]:
        """
        Processes a streaming HTTP JSON response and parses its chunks to objects of the given type.
        """
        buffer = ""
        for chunk in resp.iter_text():
            buffer += chunk
            try:
                # Try to parse a complete JSON object
                json_data = json.loads(buffer)
                obj = obj_type.model_validate(json_data)
                buffer = ""  # Clear the buffer after successful parsing
                yield obj
            except json.JSONDecodeError:
                # Incomplete JSON, wait for more data
                pass