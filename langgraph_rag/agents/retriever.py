from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings


class RetrieverAgent:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.1", temperature=0)
        self.search_term_prompt_template = """
# Task
You are an intelligent search term suggestion agent. Given a user question, suggest search english terms, up to 3 words, which will optimize the vector search.
Strictly Return your response with just the search term.

Question: {question}
"""
        self.search_term_prompt = ChatPromptTemplate.from_template(
            self.search_term_prompt_template
        )
        self.search_term_chain = self.search_term_prompt | self.llm
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3_000, chunk_overlap=50,
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250, chunk_overlap=25,
        )

        self.fs = LocalFileStore(f"knowledge_base/local_file_store")
        self.store = create_kv_docstore(self.fs)

        self.vectorstore = Chroma(
            collection_name="parent_docs",
            embedding_function=OllamaEmbeddings(
                model="nomic-embed-text:latest"
            ),
            persist_directory=f"knowledge_base/vectorstore",
        )

        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
            search_kwargs={"k": 3},
        )

    def run_search_term(self, question):
        result = self.search_term_chain.invoke({"question": question})
        search_terms = result.content
        search_text = f"{search_terms}, {question}"
        return search_text

    def invoke(self, state):
        question = state.get("question")
        search_text = self.run_search_term(question)
        relevant_docs = self.retriever.invoke(search_text)
        retrieved_info = ""
        # print('retrieved docs')
        for i, d in enumerate(relevant_docs):

            retrieved_info += d.page_content
            retrieved_info += "\n"

            # print(f'--------------------------------doc_{i}--------------------------------')
            # print(retrieved_info)
        state["search_terms"] = search_text
        state["retrieved_info"] = retrieved_info

        return state


if __name__ == "__main__":
    question = "who founded kredivo?"
    retriever_agent = RetrieverAgent()
