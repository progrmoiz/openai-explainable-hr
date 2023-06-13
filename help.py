import os
import openai

from qdrant_client import QdrantClient
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

qdrant_url = "https://ead8e7d9-2410-4f0b-a586-d0942c942471.us-east-1-0.aws.cloud.qdrant.io:6333"
qdrant_api_key = "nvVZW1lghCm-DB05P_PetBQ0aLtSYG9zPICnXBEuDb0hNPHOPOeqPg"
openai_api_key = "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"
collection_name = "text-embedding-ada-002"

system_prompt_format = """
Assistant is a large language model trained by OpenAI. Follow the user's instructions carefully. Respond using markdown.
Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question. 
Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brakets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].

Sources:
{sources}
"""

# ChatGPT uses a particular set of tokens to indicate turns in conversations
# Assistant helps the company employees with their healthcare plan questions and employee handbook questions. 
prompt_prefix = """<|im_start|>system
Assistant is a large language model trained by OpenAI. Follow the user's instructions carefully. Respond using markdown.
Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question. 
Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brakets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].

Sources:
{sources}

<|im_end|>"""

turn_prefix = """
<|im_start|>user
"""

turn_suffix = """
<|im_end|>
<|im_start|>assistant
"""

prompt_history = turn_prefix

history = []
messages_history = []

summary_prompt_template = """Below is a summary of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base. Generate a search query based on the conversation and the new question. Source names are not good search terms to include in the search query.

Summary:
{summary}

Question:
{question}

Search query:
"""

# Execute this cell multiple times updating user_input to accumulate chat history
user_input = "what are differences in industry profitablity?"

if len(history) > 0:
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
          {"role": "user", "content": summary_prompt_template.format(summary="\n".join(history), question=user_input)}
        ],
        temperature=0.7,
        max_tokens=32,
        stop=["\n"])
    search = completion.choices[0].message.content
else:
    search = user_input

# Alternatively simply use search_client.search(q, top=3) if not using semantic search
print("Searching:", search)
print("-------------------")


client = QdrantClient(url=qdrant_url, prefer_grpc=True, api_key=qdrant_api_key)

# embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
qdrant = Qdrant(client=client, collection_name=collection_name, embedding_function=embeddings.embed_query)

datastore_id = "clhzxoo0o0008i608x2zgicm5" # HBR Review
r = qdrant.similarity_search(
    search,
    k=5,
    filter={"datastore_id": datastore_id},
)

results = [str(doc.metadata['page']) + ": " + doc.page_content.replace("\n", "").replace("\r", "") for doc in r]
content = "\n".join(results)

# prompt = prompt_prefix.format(sources=content) + prompt_history + user_input + turn_suffix
system_prompt = system_prompt_format.format(sources=content)

messages = [
    {"role": "system", "content": system_prompt},
]

messages += messages_history

messages.append({"role": "user", "content": user_input})

print("Messages:")
print(messages)

completion = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages,
    temperature=0.7, 
    max_tokens=3500,
    stop=["<|im_end|>", "<|im_start|>"])

# prompt_history += user_input + turn_suffix + completion.choices[0].message.content + "\n<|im_end|>" + turn_prefix
history.append("user: " + user_input)
history.append("assistant: " + completion.choices[0].message.content)
messages_history.append({"role": "user", "content": user_input})
messages_history.append({"role": "assistant", "content": completion.choices[0].message.content})

print("\n-------------------\n".join(history))
print("\n-------------------\nMessage History:\n" + messages_history)
