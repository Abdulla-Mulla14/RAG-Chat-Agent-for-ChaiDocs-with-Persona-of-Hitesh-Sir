from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

client = OpenAI()


SYSTEM_PROMPT = """
    You are an assistant that breaks down a complex technical query into 3 simple, focused sub-questions suitable for information retrieval. 

    examples:
    1. what is html
    subquestions: * What does HTML stand for and what is its purpose in web development?
                  * What are some common HTML tags and their functions?
                  * How is a basic HTML document structured?
    
    2. what is git
    subquestions: * What is Git and what problem does it solve in software development?
                  * How does Git track changes in files and manage versions?
                  * What are the basic Git commands and their purposes (init, add, commit, push, pull)?

    3. why erd is used
    subquestions: * How does an ERD help in designing a database structure?
                  * What are the key components of an ERD (entities, relationships, attributes)?
                  * What is an Entity Relationship Diagram (ERD)?
"""

def generate_subqueries(user_query: str) -> list[str]:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", 
            "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]
    )

    raw = response.choices[0].message.content.strip()

    subqueries = [
        line.strip(" -1234567890.").strip()
        for line in raw.split()
        if line.strip()
    ]

    return subqueries



# query = input("> ")
# subs = generate_subqueries(query)
# print(subs)

url = "https://chaidocs.vercel.app/youtube/getting-started"
loader = WebBaseLoader(web_path=url)
raw_docs = loader.scrape()
docs = [doc for doc in raw_docs if isinstance(doc, Document) and hasattr(doc, "page_content")]


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Embed and store into Qdrant
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

vector_db = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding,
    url="http://localhost:6333",  # This matches your docker-compose setup
    collection_name="chai-docs-index"
)

print("✅ Indexing complete. Collection created.")