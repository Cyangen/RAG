from unstructured.partition.pdf import partition_pdf
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import Chroma
from langchain.storage import LocalFileStore
from langchain.schema.document import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
# from langchain_openai import ChatOpenAI
from PIL import Image
from io import BytesIO
import uuid, pickle, base64
import os

def pdf_embedder(folder_path):
    # loop through folder
    for dir, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith('.pdf'):
                file_path = os.path.join(dir, filename)
                print(file_path)

                # Reference: https://docs.unstructured.io/open-source/core-functionality/chunking
                chunks = partition_pdf(
                    filename=file_path,
                    infer_table_structure=True,            # extract tables
                    strategy="hi_res",                     # mandatory to infer tables

                    extract_image_block_types=["Image"],   # Add 'Table' to list to extract image of tables
                    # image_output_dir_path=output_path,   # if None, images and tables will saved in base64

                    extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

                    chunking_strategy="by_title",          # or 'basic'
                    max_characters=10000,                  # defaults to 500
                    combine_text_under_n_chars=2000,       # defaults to 0
                    new_after_n_chars=6000,

                    # extract_images_in_pdf=True,          # deprecated
                )
                print("DOCUMENT CHUNKED")

                #Getting text and table elements from chunks
                tables = []
                texts = []
                for chunk in chunks:
                    if "Table" in str(type(chunk)):
                        tables.append(chunk)

                    for elem in chunk.metadata.orig_elements:
                        if "Table" in str(type(elem)): tables.append(elem)

                    if "CompositeElement" in str(type((chunk))):
                        texts.append(chunk)
                    
                #Getting image data from chunks
                images = get_images_base64(chunks)

                # Prompt for text/table summary
                prompt_text = """
                You are an assistant tasked with summarizing tables and text.
                Give a concise summary of the table or text.

                Respond only with the summary, no additionnal comment.
                Do not start your message by saying "Here is a summary" or anything like that.
                Just give the summary as it is.

                Table or text chunk: {element}

                """
                prompt = ChatPromptTemplate.from_template(prompt_text)

                # Text/table summary chain
                model = ChatOllama(temperature=0.5, 
                                model='phi3.5',
                                keep_alive=0,
                                max_tokens=512)

                summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

                # Summarize text
                text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})

                # Summarize tables
                tables_html = [table.metadata.text_as_html for table in tables]
                table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})
                print("TEXT & TABLES SUMMARISED")

                # Prompt for image summary(change prompt as you see fit)
                prompt_template = """Describe the image in detail. For context,
                                the image is part of a military documents explaining the assets, as well as any situation reports. 
                                Be specific about graphs, such as bar plots."""
                # prompt_template = """Describe the image in detail."""
                messages = [
                    (
                        "user",
                        [
                            {"type": "text", "text": prompt_template},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/jpeg;base64,{image}"},
                            },
                        ],
                    )
                ]

                prompt = ChatPromptTemplate.from_messages(messages)

                chain = prompt | ChatOllama(model="llava-phi3", keep_alive=0) | StrOutputParser()

                image_summaries = chain.batch(images)
                print("IMAGES SUMMARISED")

                #Embeddings
                #Define embedding model
                embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

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


                #convert objects to bytes-like objects
                texts_pickled = [pickle.dumps(i) for i in texts]
                tables_pickled = [pickle.dumps(i) for i in tables]
                images_pickled = [pickle.dumps(i) for i in images]

                # Add texts
                if texts:
                    print("EMBEDDING TEXTS...")
                    doc_ids = [str(uuid.uuid4()) for _ in texts_pickled]
                    summary_texts = [
                        Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
                    ]
                    retriever.vectorstore.add_documents(summary_texts)
                    retriever.docstore.mset(list(zip(doc_ids, texts_pickled)))

                # Add tables
                if tables:
                    print("EMBEDDING TABLES...")
                    table_ids = [str(uuid.uuid4()) for _ in tables_pickled]
                    summary_tables = [
                        Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
                    ]
                    retriever.vectorstore.add_documents(summary_tables)
                    retriever.docstore.mset(list(zip(table_ids, tables_pickled)))

                # Add image summaries
                if images:
                    print("EMBEDDING IMAGES...")
                    img_ids = [str(uuid.uuid4()) for _ in images_pickled]
                    summary_img = [
                        Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)
                    ]
                    retriever.vectorstore.add_documents(summary_img)
                    retriever.docstore.mset(list(zip(img_ids, images_pickled)))
                print("SUMMARIES EMBEDDED")



def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

def image_embedder(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode("utf-8")

    prompt_template = """Describe the image in detail."""
    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                },
            ],
        )
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    chain = prompt | ChatOllama(model="llava-phi3", keep_alive=0) | StrOutputParser()

    image_summaries = [chain.invoke(encoded_string)]
    #Embeddings
    #Define embedding model
    embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

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

    #convert objects to bytes-like objects
    images_pickled = [pickle.dumps(i) for i in [encoded_string]]

    # Add image summaries
    img_ids = [str(uuid.uuid4()) for _ in images_pickled]
    summary_img = [
        Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)
    ]

    retriever.vectorstore.add_documents(summary_img)
    retriever.docstore.mset(list(zip(img_ids, images_pickled)))

if __name__ == "__main__":
    # pdf_embedder(r"..\data\pdf_test\1706.03762v7.pdf")
    # image_embedder(r"S:\LLM\RAG\multimodal_rag\image2.jpg")
    
    pdf_embedder("../data/Open_Source_Dataset/Intel reports")

    print("DONE")

