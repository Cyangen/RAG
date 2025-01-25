# Other modules
from IPython.display import Image, display
from PIL import Image as PIL_Image
from base64 import b64decode
from io import BytesIO
import pickle, base64

# Langshit
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain.vectorstores import Chroma
from langchain.storage import LocalFileStore
from langchain_community.embeddings import OllamaEmbeddings
# from langchain.retrievers.multi_vector import MultiVectorRetriever
# from langchain_openai import ChatOpenAI
from langchain.retrievers.document_compressors import FlashrankRerank

# Editted Source Codes
from Rerank_MVR import MultiVectorRetriever

def parse_docs(docs):
    """Split base64-encoded images and texts"""
    b64 = []
    text = []
    for doc_pickle in docs:
        doc = pickle.loads(doc_pickle)
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(doc)
    return {"images": b64, "texts": text}


def build_prompt(kwargs):

    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.text

    # construct prompt with context (including images)
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and the below image.
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            # print(image)
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )


def display_base64_image(base64_code):
    # Decode the base64 string to binary
    image_data = base64.b64decode(base64_code)

    # Display the image
    display(Image(data=image_data))

def embedding_chains():
    # Define embedding model
    embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=False)

    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(collection_name="multi_modal_rag", 
                        embedding_function=embeddings,
                        persist_directory="./db-test"
                        )

    # The storage layer for the parent documents
    store = LocalFileStore("./db-localfiles")
    id_key = "doc_id"

    # The retriever (empty to start) (Rerank)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
        reranking_model=FlashrankRerank(top_n=5),
        search_kwargs= {"k": 20} # set to 5 if not doing reranking
    )
    
    # The retriever (empty to start) (No Rerank)
    # retriever = MultiVectorRetriever(
    #     vectorstore=vectorstore,
    #     docstore=store,
    #     id_key=id_key,
    #     search_kwargs= {"k": 5}
    # )

    chain = (
        {
            "context": retriever | RunnableLambda(parse_docs),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(build_prompt)
        | ChatOllama(model=local_model, keep_alive=0, temperature=0, max_tokens=512)
    )

    chain_with_sources = {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    } | RunnablePassthrough().assign(
        response=(
            RunnableLambda(build_prompt)
            | ChatOllama(model=local_model, keep_alive=0, temperature=0, max_tokens=512)
        )
    )
    return chain, chain_with_sources

chain, chain_with_sources = embedding_chains()
def response_no_sources(user_input):
    print("\n\nResponse:\n")
    for chunk in chain.stream(user_input):
        print(chunk.content, end="", flush=True)
    print()


def response_with_sources(user_input):
    response = chain_with_sources.invoke(
        user_input
    )
    print("\n\nResponse:", response['response'])

    print("\n\nContext:")
    # print(len(response['context']['texts']), len(response['context']['images']))
    for text in response['context']['texts']:
        print(text.text)
        print("Page number: ", text.metadata.page_number)
        print("\n" + "-"*50 + "\n")
    for image in response['context']['images']:
        # display_base64_image(image)
        PIL_Image.open(BytesIO(base64.b64decode(image))).show()


if __name__ == "__main__":
    local_model = "llama3.2-vision"
    response_with_sources("How does multi-head attention mechanism works within the Transformer model?") #Replace testing prompt with user input
    # response_with_sources("The image presents a diagram illustrating the flow of information from sensors to the central processing unit (CPU) and then back out to various components. The diagram is in black and white, with arrows indicating the direction of data transfer.\n\nAt the top left,")