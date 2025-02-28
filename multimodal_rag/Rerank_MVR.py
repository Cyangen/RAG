from enum import Enum
from typing import Dict, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.stores import BaseStore, ByteStore
from langchain_core.vectorstores import VectorStore
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from langchain.storage._lc_store import create_kv_docstore

# Meine Additions
from langchain.retrievers import ContextualCompressionRetriever
# from ragatouille import RAGPretrainedModel
from langchain.retrievers.document_compressors import FlashrankRerank


class SearchType(str, Enum):
    """Enumerator of the types of search to perform."""

    similarity = "similarity"
    """Similarity search."""
    mmr = "mmr"
    """Maximal Marginal Relevance reranking of similarity search."""


class MultiVectorRetriever(BaseRetriever):
    """Retrieve from a set of multiple embeddings for the same document."""

    vectorstore: VectorStore
    """The underlying vectorstore to use to store small chunks
    and their embedding vectors"""
    byte_store: Optional[ByteStore] = None
    """The lower-level backing storage layer for the parent documents"""
    docstore: BaseStore[str, Document]
    """The storage interface for the parent documents"""
    id_key: str = "doc_id"
    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""
    search_type: SearchType = SearchType.similarity
    """Type of search to perform (similarity / mmr)"""
    reranking_model: Optional[FlashrankRerank] = None
    """Type of Reranking Model loaded with RAGPretrainedModel"""
    reranking_top_k: Optional[int] = None
    """Type of how many (k) documents should be returned after reranking"""

    @root_validator(pre=True)
    def shim_docstore(cls, values: Dict) -> Dict:
        byte_store = values.get("byte_store")
        docstore = values.get("docstore")
        if byte_store is not None:
            docstore = create_kv_docstore(byte_store)
        elif docstore is None:
            raise Exception("You must pass a `byte_store` parameter.")
        values["docstore"] = docstore
        return values
    
    def get_matching_reranked_docs(self, child_chunk_results: List[Document], reranking_results: List[Dict]) -> List[Document]:
        """
        Return a list of strings that are present in both child_chunk_results and list2.

        Parameters:
        child_chunk_results (list of LangchainDoc): The first list of Langchain documents.
        reranking_results(list of dict): The second list of dictionaries with content strings.

        Returns:
        list of str: A list containing strings that are present in both input lists.
        """
        # Extract strings and scores from list2 and create an order map
        content_score_map = {d.page_content: d.metadata["relevance_score"] for d in reranking_results}
        order_map = {content: idx for idx, content in enumerate(content_score_map.keys())}

        # Filter list1 based on matching content in list2
        filtered_list1 = [doc for doc in child_chunk_results if doc.page_content in content_score_map]

        # Update metadata with scores and sort filtered list1 based on the order in list2
        for doc in filtered_list1:
            doc.metadata['score'] = content_score_map[doc.page_content]

        filtered_list1.sort(key=lambda doc: order_map[doc.page_content])
        
        return filtered_list1

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """

        # Query Rewriting for text
        prompt_text ="""You are an expert at converting user questions into database queries. \

Perform query expansion. If there are multiple common ways of phrasing a user question \
or common synonyms for key words in the question, make sure to return multiple versions \
of the query with the different phrasings.

If there are acronyms or words you are not familiar with, do not try to rephrase them.

Return at least 3 versions of the question.

QUESTION: {question}
"""
        prompt = ChatPromptTemplate.from_template(prompt_text)

        model = ChatOllama(temperature=0.5, 
                        model='phi3.5',
                        keep_alive=0,
                        max_tokens=512)

        rewrite_chain = prompt | model | StrOutputParser()

        query = str(rewrite_chain.invoke(query))
        print(query)



        if self.search_type == SearchType.mmr:
            sub_docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            if self.reranking_model:
                sub_docs = self.vectorstore.similarity_search(query, **self.search_kwargs)
                page_contents = []
                # print(len(sub_docs))
                for i in sub_docs:
                    ct = i.page_content
                    ctd = Document(
                        page_content=ct,
                        metadata={"source":"local"}
                    )
                    page_contents.append(ctd)

                reranking_docs = self.reranking_model.compress_documents(page_contents, query)
                # print(len(reranking_docs))
                sub_docs = self.get_matching_reranked_docs(child_chunk_results=sub_docs, reranking_results=reranking_docs)

                # print(sub_docs)
                print(f"AMOUNT OF SUB_DOCS AFTER RERANKING: {len(sub_docs)}")
            else:
                sub_docs = self.vectorstore.similarity_search(query, **self.search_kwargs)

        # We do this to maintain the order of the ids that are returned
        ids = []
        for d in sub_docs:
            print(d.metadata[self.id_key])
            if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])
        docs = self.docstore.mget(ids)
        # debugging code
        # for i in docs:
        #     if i is None: 
        #         print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        #     else:
        #         print("###############################################")
        # print(self.docstore.mget("248df390-defc-461b-840c-cdaa4a773137"))
        return [d for d in docs if d is not None]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Asynchronously get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        if self.search_type == SearchType.mmr:
            sub_docs = await self.vectorstore.amax_marginal_relevance_search(
                query, **self.search_kwargs
            )
        else:
            sub_docs = await self.vectorstore.asimilarity_search(
                query, **self.search_kwargs
            )

        # We do this to maintain the order of the ids that are returned
        ids = []
        for d in sub_docs:
            if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])
        docs = await self.docstore.amget(ids)
        return [d for d in docs if d is not None]



