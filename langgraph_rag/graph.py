from langgraph.graph import END, StateGraph

from agents.responder import ResponderAgent
from agents.retriever import RetrieverAgent
from agents.router import RouterAgent
from state import AgentGraphState

router_agent = RouterAgent()
retriever_agent = RetrieverAgent()
responder_agent = ResponderAgent()


def check_require_RAG(state: AgentGraphState):
    if state["router_need_retriever"]:
        return "retriever"
    else:
        return "responder"


# structure
graph = StateGraph(AgentGraphState)

graph.add_node("router", lambda state: router_agent.invoke(state))
graph.add_node("retriever", lambda state: retriever_agent.invoke(state))
graph.add_node("responder", lambda state: responder_agent.invoke(state))

# flow
graph.set_entry_point("router")
graph.add_conditional_edges(
    "router",
    check_require_RAG,
    {"retriever": "retriever", "responder": "responder"},
)
graph.add_edge("retriever", "responder")
graph.add_edge("responder", END)

workflow = graph.compile()

if __name__ == "__main__":
    question = "who is the ceo of kredivo?"
    for msg, metadata in workflow.stream(
        {"question": question,}, stream_mode="messages",
    ):
        if "retriever" in metadata["langgraph_triggers"]:
            print(msg.content, flush=True)
