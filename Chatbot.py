import os
import datetime
import time
import streamlit as st
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

st.set_page_config(
    page_title="Chatbot",
    layout="wide"
)

load_dotenv()
def greet_user():
    current_hour = datetime.datetime.now().hour
    if 5 <= current_hour < 12:
        greeting = "Good Morning!"
    elif 12 <= current_hour < 16:
        greeting = "Good Afternoon!"
    elif 16 <= current_hour < 21:
        greeting = "Good Evening!"
    else:
        greeting = "Good Night!"
    return greeting


st.title("Schneider Electric Chatbot")
with st.sidebar:
    st.image("./Schneider-Electric-logo-.png", use_container_width=True)
    st.write("")
    st.write("")
    st.markdown(":blue[About:]")
    st.write("Welcome to Schneider Electric's AI-powered chatbot! Get instant support and expert guidance for all "
             "your Schneider Electric devices and solutions.")
    st.write("")
st.sidebar.write("")

# ------------------------------------- STREAMLIT UI -------------------------------------------------------------------
st.header("" + greet_user(), anchor=False, divider="violet")
st.sidebar.markdown(":orange[Disclaimer:]")
st.sidebar.write("Using a preloaded document for querying.")

# Select LLM
chosen_model = "Llama3-70b-8192"

# ------------------------------------- DOCUMENT SETUP ---------------------------------------------------------------
DOCUMENT_PATH = "ALL_MERGED.pdf"  # Replace with the path to your document
INDEX_PATH = "retriever_index"  # Path to save/load the FAISS index

# Ensure retriever is initialized
if "retriever" not in st.session_state:
    embedding_model = OllamaEmbeddings(model="nomic-embed-text:latest")
    try:
        if os.path.exists(INDEX_PATH):
            with st.spinner("Loading retriever from saved index..."):
                # Load FAISS index with safe deserialization
                vectors = FAISS.load_local(
                    INDEX_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True
                )
                st.session_state.retriever = vectors.as_retriever()
        else:
            raise FileNotFoundError("FAISS index not found. Recreating retriever.")
    except Exception as e:
        st.warning(f"Failed to load index: {e}. Recreating retriever.")
        if os.path.exists(DOCUMENT_PATH):
            # Handle retriever recreation
            if DOCUMENT_PATH.endswith(".pdf"):
                loader = PyPDFLoader(DOCUMENT_PATH)
            elif DOCUMENT_PATH.endswith(".txt"):
                loader = TextLoader(file_path=DOCUMENT_PATH)
            elif DOCUMENT_PATH.endswith(".csv"):
                loader = CSVLoader(file_path=DOCUMENT_PATH)
            else:
                st.error("Unsupported document type. Please use PDF, TXT, or CSV.")
                st.stop()

            document = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splitted_documents = text_splitter.split_documents(document)

            # Create and save FAISS index
            vectors = FAISS.from_documents(documents=splitted_documents, embedding=embedding_model)
            vectors.save_local(INDEX_PATH)
            st.session_state.retriever = vectors.as_retriever()
        else:
            st.error(f"Document not found at path: {DOCUMENT_PATH}")
            st.stop()

# Ensure retrieval chain is initialized
if "retrieval_chain" not in st.session_state:
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name=chosen_model
    )
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent assistant specializing in answering user queries related to Schneider 
        Electric's products and technical documents.  If the user asks for information about Schneider Electric's 
        products or any other topic, respond clearly and concisely in a well-structured format."""),
        ("user", "The Document is as follows: {context}. User Question: {input}")
    ])
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
    st.session_state.retrieval_chain = create_retrieval_chain(st.session_state.retriever, document_chain)

# ------------------------------------- USER QUERY --------------------------------------------------------------------
if "retrieval_chain" in st.session_state:
    user_prompt = st.chat_input("Enter your query: ")
    if user_prompt:
        with st.container():
            st.markdown(":orange[User Prompt:]")
            st.write(user_prompt)

            start_time = time.time()

            # Run retrieval chain using 'invoke' instead of 'run'
            response = st.session_state.retrieval_chain.invoke({'input': user_prompt})

            # Extract and display the "answer" field
            if isinstance(response, dict) and "answer" in response:
                st.markdown(":blue[Response:]")
                st.write(response["answer"])
            else:
                st.markdown(":blue[Response:]")
                st.write("Could not extract a valid answer. Please try again.")

            response_time = round((time.time() - start_time), 2)
            st.sidebar.markdown(f"\n\n\n:green[Response Time:] {response_time} sec.")
