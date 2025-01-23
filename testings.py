# from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama.llms import OllamaLLM
import warnings

import torch
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

# ignore warnings
warnings.filterwarnings("ignore")

# # Create embeddingsclear
embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=False)

db = Chroma(persist_directory="./database/db-keef",
            embedding_function=embeddings)

# # Create retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs= {"k": 20} # was 3
)

# # Create Ollama language model - Gemma 2
local_llm = 'phi3.5'

llm = ChatOllama(model=local_llm,
                 keep_alive=0 , # was "3h"
                 max_tokens=512,  
                 temperature=0)

# Create prompt template
template = """<|user|>Answer the question based only on the following context and extract out a meaningful answer. \
Please write in full sentences with correct spelling and punctuation. if it makes sense use lists. \
If the context doen't contain the answer, just respond that you are unable to find an answer. \

CONTEXT: {context}

QUESTION: {question}<|end|>
<|assistant|>AI:"""
prompt = ChatPromptTemplate.from_template(template)

# Create the RAG chain using LCEL with prompt printing and streaming output
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Function to ask questions
def ask_question(question):
    print("Answer:\n\n", end=" ", flush=True)
    for chunk in rag_chain.stream(question):
        print(chunk.content, end="", flush=True)
    print("\n")
    
def test_out_shit(question):
    print("Answer:\n\n", end=" ", flush=True)
    
    compressor = FlashrankRerank(top_n=3)
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever,
    )
    
    output = compression_retriever.invoke(question)
    
    print(
        f"\n{'-' * 100}\n".join(
            [
                f"Document {i+1}:\n\n{d.page_content}\nMetadata: {d.metadata}"
                for i, d in enumerate(output)
            ]
        )
    )
    
    print([doc.metadata["id"] for doc in output])
    
    print("\n")

# Example usage
if __name__ == "__main__":
    while True:
        user_question = input("Ask a question (or type 'quit' to exit): ")
        if user_question.lower() == 'quit':
            break
        
        # answer = ask_question(user_question)
        answer = test_out_shit(user_question)
        # print("\nFull answer received.\n")