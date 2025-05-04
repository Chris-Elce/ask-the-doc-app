import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 🔐 Set OpenAI key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

def generate_response(uploaded_file, query_text):
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode("utf-8")]

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents(documents)

    # Use environment-configured OpenAI embeddings and LLM
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    # Retrieval-based QA chain
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)
    return qa.run(query_text)

# Streamlit UI setup
st.set_page_config(page_title="🦜🔗 Ask the Doc App")
st.title("🦜🔗 Ask the Doc App")

# File upload
uploaded_file = st.file_uploader("Upload a .txt file", type="txt")

# Query input
query_text = st.text_input("Enter your question:", placeholder="Please provide a short summary.", disabled=not uploaded_file)

# Form submission
result = []
with st.form("ask_doc_form", clear_on_submit=True):
    submitted = st.form_submit_button("Submit", disabled=not (uploaded_file and query_text))
    if submitted:
        with st.spinner("Searching..."):
            try:
                response = generate_response(uploaded_file, query_text)
                result.append(response)
            except Exception as e:
                st.error(f"⚠️ An error occurred: {e}")

# Display result
if result:
    st.info(result[0])
