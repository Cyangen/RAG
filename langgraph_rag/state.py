from typing import TypedDict


class AgentGraphState(TypedDict):
    question: str
    router_need_retriever: bool
    router_need_retriever_reason: str
    router_need_system_2: bool
    router_need_system_2_reason: str
    search_terms: str
    retrieved_info: str
    responder_reply: str
