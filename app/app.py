from qdrant_client import QdrantClient
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.document_loaders.pdf import BasePDFLoader
from typing import List
from langchain.embeddings import CohereEmbeddings
import json
import logging
import os
import re
import sys

import langchain
from flask import Flask, render_template, request, send_from_directory
from flask_cors import CORS, cross_origin

from langchain import OpenAI, PromptTemplate
from langchain.cache import InMemoryCache
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.example_selector import \
    SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain import VectorDBQA, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import WebBaseLoader

from langchain.docstore.document import Document
from secure import require_apikey

import random
import json
import os
import urllib.request
import mimetypes
# Loading environment variables
import os

import urllib.request


# from secure import require_apikey

langchain.llm_cache = InMemoryCache()

logger = logging.getLogger()

app = Flask(__name__)

CORS(app)

app.config.from_object(__name__)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))


@app.route('/health')
def health():
    return 'It is alive!\n'


# Embedding of a document


class PyPDFLoader(BasePDFLoader):
    """Loads a PDF with pypdf and chunks at character level.

    Loader also stores page numbers, from_line, and to_line in metadatas.
    """

    def __init__(self, file_path: str):
        """Initialize with file path."""
        try:
            import pypdf  # noqa:F401
        except ImportError:
            raise ValueError(
                "pypdf package not found, please install it with " "`pip install pypdf`"
            )
        super().__init__(file_path)

    def load(self) -> List[Document]:
        """Load given path as pages."""
        import pypdf

        with open(self.file_path, "rb") as pdf_file_obj:
            pdf_reader = pypdf.PdfReader(pdf_file_obj)
            total_lines = 0
            documents = []
            for i, page in enumerate(pdf_reader.pages):
                from_line = total_lines + 1
                page_lines = page.extract_text().count("\n") + 1
                to_line = from_line + page_lines - 1
                total_lines += page_lines

                documents.append(
                    Document(
                        page_content=page.extract_text(),
                        metadata={
                            "source": self.file_path,
                            "page": i,
                            "from_line": from_line,
                            "to_line": to_line,
                        },
                    )
                )
            return documents


@app.route('/embed', methods=['POST'])
@require_apikey
def embed():
    try:
        # Define the file path variable
        file_path = None

        body = request.get_json(force=True)
        if body is None or 'document_id' not in body or 'datastore_id' not in body or 'loader_type' not in body or 'openai_api_key' not in body or 'qdrant_url' not in body or 'qdrant_api_key' not in body:
            raise ValueError('Invalid request body')

        document_id = body.get('document_id')
        datastore_id = body.get('datastore_id')

        loader_type = body.get('loader_type')

        openai_api_key = body.get('openai_api_key')
        collection_name = "text-embedding-ada-002"

        qdrant_url = body.get('qdrant_url')
        qdrant_api_key = body.get('qdrant_api_key')

        docs = []
        if loader_type == "webpage":
            urls = body.get("urls")

            # loader = WebBaseLoader(["https://www.espn.com/", "https://google.com"])
            loader = WebBaseLoader(urls)
            docs = loader.load()
        elif loader_type == "file":
            file_url = body.get("file_url")

            # Download the file from the url provided
            folder_path = f'./'
            # Create the folder if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)
            # Filename for the downloaded file
            filename = file_url.split('/')[-1]
            # Full path to the downloaded file
            file_path = os.path.join(folder_path, filename)

            import ssl  # not the best for production use to not verify ssl, but fine for testing
            ssl._create_default_https_context = ssl._create_unverified_context

            print(file_url)
            # Download the file and save it to the local folder
            urllib.request.urlretrieve(file_url, file_path)

            # Checking filetype for document parsing, PyPDF is a lot faster than Unstructured for pdfs.
            import mimetypes
            mime_type = mimetypes.guess_type(file_path)[0]

            print(file_path, mime_type)
            if mime_type == 'application/pdf':
                loader = PyPDFLoader(file_path)
                docs = loader.load_and_split()
            else:
                loader = UnstructuredFileLoader(file_path)
                docs = loader.load()

        # add metadata to the document
        for doc in docs:
            doc.metadata["datastore_id"] = datastore_id
            doc.metadata["document_id"] = document_id

        # Generate embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        try:
            print("Adding to existing collection")
            client = QdrantClient(
                url=qdrant_url, prefer_grpc=True, api_key=qdrant_api_key)
            qdrant = Qdrant(client, collection_name,
                            embedding_function=embeddings.embed_query)
            qdrant.add_documents(docs)
        except:
            print("Creating new collection")
            qdrant = Qdrant.from_documents(
                docs,
                embeddings,
                url=qdrant_url,
                collection_name=collection_name,
                prefer_grpc=True,
                api_key=qdrant_api_key,
            )

        if file_path is not None:
            os.remove(file_path)  # Delete downloaded file
        # add metadata to the document

        return {
            "collection_name": qdrant.collection_name,
            "datastore_id": datastore_id,
            "document_id": document_id,
        }
    except Exception as e:
        # Remove the file if it exists
        try:
            if file_path is not None:
                os.remove(file_path)
        except:
            pass

        print(e)
        return {
            "error": str(e),
        }


# Retrieve information from a collection
# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI

# def get_chat_history(inputs) -> str:
#     res = []
#     for human, ai in inputs:
#         res.append(f"Human:{human}\nAI:{ai}")
#     return "\n".join(res)


@app.route('/search', methods=['POST'])
@require_apikey
def search():
    body = request.get_json(force=True)
    # messages_array = body.get('messages') or []

    openai_api_key = body.get('openai_api_key')
    qdrant_url = body.get('qdrant_url')
    qdrant_api_key = body.get('qdrant_api_key')

    collection_name = "text-embedding-ada-002"

    # document_id = body.get("document_id")
    datastore_id = body.get("datastore_id")

    query = body.get("query")

    client = QdrantClient(url=qdrant_url, prefer_grpc=True,
                          api_key=qdrant_api_key)
    # chat_history = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    qdrant = Qdrant(client=client, collection_name=collection_name,
                    embedding_function=embeddings.embed_query)
    search_results = qdrant.similarity_search(
        query,
        k=5,
        filter={"datastore_id": datastore_id},
    )

    # chain = load_qa_chain(
    #     ChatOpenAI(openai_api_key=openai_api_key,temperature=0.2),
    #     chain_type="stuff",
    #     memory=memory,
    # )
    # llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.2)

    # retriever = qdrant.as_retriever()
    # retriever.search_kwargs = {
    #     'k': 5,
    #     'filters': [{ "datastore_id": datastore_id }],
    # }

    # qa = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=retriever,
    #     return_source_documents=True,
    #     get_chat_history=get_chat_history,
    # )

    # result = qa({"question": query, "chat_history": chat_history})

    results = []
    for document in search_results:
        results.append({
            "page_content": document.page_content,
            "metadata": document.metadata,
        })

    # print(result)

    return {
        "results": results,
    }


@ app.route('/dummy')
def dummy():
    return {
        'text': 'Hello World!',
    }, 200
