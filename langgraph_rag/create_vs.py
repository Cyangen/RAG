from pathlib import Path

import fitz
from langchain.docstore.document import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm

doc_dir = "knowledge_base/"

pdf_paths = sorted(list(Path(doc_dir).glob("*.pdf")))

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3_000, chunk_overlap=50,
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250, chunk_overlap=25,
)

fs = LocalFileStore("knowledge_base/local_file_store")
store = create_kv_docstore(fs)

vectorstore = Chroma(
    collection_name="parent_docs",
    embedding_function=OllamaEmbeddings(model="nomic-embed-text:latest"),
    persist_directory="knowledge_base/vectorstore",
)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 1},
)

docs = []

for p in tqdm(pdf_paths):
    pdf_doc = fitz.open(str(p))
    pdf_str = ""
    for page in pdf_doc:
        pdf_str += page.get_text()

    # create langchian doc
    doc = Document(page_content=pdf_str)
    docs.append(doc)

retriever.add_documents(docs)
