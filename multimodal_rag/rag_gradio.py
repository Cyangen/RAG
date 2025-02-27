from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain.vectorstores import Chroma
from langchain.storage import LocalFileStore
from langchain_community.embeddings import OllamaEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
# from langchain_openai import ChatOpenAI
from IPython.display import Image, display
from PIL import Image as PIL_Image
from base64 import b64decode
from io import BytesIO
import pickle, base64
import gradio as gr

local_model = "llama3.2-vision"

# Split base64-encoded images and texts
def parse_docs(docs):
    b64 = []
    text = []
    for doc_pickle in docs:
        doc = pickle.loads(doc_pickle)
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(doc)
    global data 
    data = {"images": b64, "texts": text}
    print(len(b64), len(text))
    return {"images": b64, "texts": text}

# Using Dict output of parse_docs and Dict output of chain, makes a ChatPromptTemplate for model input.
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

# Returns constructed chain and retriever functions
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

    # The retriever (empty to start)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    chain = (
        {
            "context": retriever | RunnableLambda(parse_docs),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(build_prompt)
        | ChatOllama(model=local_model, keep_alive=0, temperature=0, max_tokens=512)
    )

    return chain, retriever

chain, retriever = embedding_chains()


def gr_func(user_input):
    print("-"*50 + "\nRUNNING\n" + "-"*50)
    

    result = ""
    text_context = ""
    img = []
    
    for chunk in chain.stream(user_input):
        print(chunk.content, end="", flush=True)
        result += chunk.content
        yield result, text_context, img

    for index, i in enumerate(data["texts"]):
        text_context += f"\n{'-'*50} [{index}] {'-'*50} \n" + i.text

    for i in data["images"]: #please change to adapt for more images
        img.append(PIL_Image.open(BytesIO(base64.b64decode(i))))

    print()
    yield result, text_context, img

if __name__ == "__main__":
    interface = gr.Interface(
        title= "Multimodal RAG",
        fn=gr_func,
        inputs=[gr.Textbox(label="Question", placeholder="Enter your question here", lines=1, max_lines=10)],
        outputs=[gr.Textbox(label="Answer", lines=1, max_lines=10), gr.Textbox(label="Context", lines=1, max_lines=10), gr.Gallery(label="Image Context")],
    )

    interface.launch(share=False)