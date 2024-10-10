import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


load_dotenv()
# os.environ['LANGCHAIN_TRACING_V2'] = 'true'
# os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
# os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['OPENAI_API_KEY'] =os.getenv('OPEN_API_KEY') 



# load the data 

def load_data():
    loader = WebBaseLoader(
        web_path=("https://medium.com/@zlodeibaal/cookbook-for-edge-ai-boards-2024-2025-b9d7dcad73d6",)
    )
    return loader.load()



def get_chunks(blog_doc):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350, 
    chunk_overlap=50
)
    chunks = text_splitter.split_documents(blog_doc)
    print(f"Number of chunks: {len(chunks)}")
    return chunks
        

def store_in_vector_db(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore


def build_model():
    load_dotenv()  # Load environment variables
    blog_doc = load_data()  # Load data from the web
    chunks = get_chunks(blog_doc)  # Split the data into chunks
    vectorstore = store_in_vector_db(chunks)  # Store in vector DB
    print("Data stored in vector database successfully.")
    return vectorstore


vector_store_result = build_model()
vector_store_result






