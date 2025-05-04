import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Get OpenAI API key from env or Streamlit Secrets
openai_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

def generate_response(uploaded_file, query_text):
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode("utf-8")]

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_key), chain_type="stuff", retriever=retriever)
    return qa.run(query_text)

# Streamlit UI
st.set_page_config(page_title="🦜🔗 Ask the Doc App")
st.title("🦜🔗 Ask the Doc App")

uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
query_text = st.text_input("Enter your question:", placeholder="Please provide a short summary.", disabled=not uploaded_file)

result = []
with st.form("ask_doc_form", clear_on_submit=True):
    submitted = st.form_submit_button("Submit", disabled=not (uploaded_file and query_text))
    if submitted:
        if not openai_key:
            st.error("❌ OpenAI API key not found. Please set it via environment variable or Streamlit secrets.")
        else:
            with st.spinner("Searching..."):
                response = generate_response(uploaded_file, query_text)
                result.append(response)

if result:
    st.info(result[0])
