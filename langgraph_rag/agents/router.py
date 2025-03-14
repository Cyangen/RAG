from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel


class RequireRetrieval(BaseModel):
    requires_retrieval: bool
    reason: str


class RequireThinking(BaseModel):
    requires_thinking: bool
    reason: str


class RouterAgent:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.1", temperature=0)

        self.req_ret_output_parser = PydanticOutputParser(
            pydantic_object=RequireRetrieval
        )
        self.req_thi_output_parser = PydanticOutputParser(
            pydantic_object=RequireThinking
        )

        self.req_ret_prompt_template = """
# Task
You are an intelligent routing agent. Given a user question determine if the question requires:
- "knowledge retrieval on Specialised Knoweldge". If the question is asking about Specialised Knoweldge, return True

Strictly Return your response in this JSON format:
{{"requires_retrieval": "<bool>", "reason": "<explanation>"}}

Question: {question}
        """

        self.req_thi_prompt_template = """
# Task
You are an intelligent routing agent. Given a user question determine if the question requires:
- "thinking". If the question is complicated and need some thinking before you can reply, return True

Strictly Return your response in this JSON format:
{{"requires_thinking": "<bool>", "reason": "<explanation>"}}

Question: {question}
        """

        self.req_ret_prompt = ChatPromptTemplate.from_template(
            self.req_ret_prompt_template
        )
        self.req_ret_chain = (
            self.req_ret_prompt | self.llm | self.req_ret_output_parser
        )

        self.req_thi_prompt = ChatPromptTemplate.from_template(
            self.req_thi_prompt_template
        )
        self.req_thi_chain = (
            self.req_thi_prompt | self.llm | self.req_thi_output_parser
        )

    def run(self, question):
        ret_result = self.req_ret_chain.invoke({"question": question,})

        thi_result = self.req_thi_chain.invoke({"question": question,})
        return ret_result, thi_result

    def invoke(self, state):
        question = state.get("question")
        try:  # sometimes it will fail due to guardrails
            ret_result, thi_result = self.run(question)
            print("requires retriever: ", ret_result)
            print("requires thinking: ", thi_result)
            state["router_need_retriever"] = ret_result.requires_retrieval
            state["router_need_system_2"] = thi_result.requires_thinking
            state["router_need_retriever_reason"] = ret_result.reason
            state["router_need_system_2_reason"] = thi_result.reason
        except Exception as e:
            print(e)
            state["router_need_retriever"] = False
            state["router_need_system_2"] = False

        return state


if __name__ == "__main__":
    question = 'how many "y" is in strawberry?'
    router_agent = RouterAgent()
    print(router_agent.run(question))
