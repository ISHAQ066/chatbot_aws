import json
import os
import io
import tempfile
import sys
import boto3
import streamlit as st
from fpdf2 import FPDF 
## We will be using Titan Embeddings Model To generate Embedding

from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data Ingestion

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Vector Embedding And Vector Store

from langchain.vectorstores import FAISS

## LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)


## Data ingestion
def data_ingestion(uploaded_pdf):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        # Write the PDF data to the temporary file
        temp_file.write(uploaded_pdf.read())
        temp_file_path = temp_file.name

    # Use PyPDFLoader to load the PDF from the file path
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()

    # Use text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)

    # Clean up the temporary file
    temp_file.close()
    
    return docs
## Vector Embedding and vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def get_claude_llm():
    ##create the Claude Model
    llm = Bedrock(model_id="ai21.j2-mid-v1", client=bedrock, model_kwargs={'maxTokens':512})
    return llm

def get_llama2_llm():
    ##create the Llama2 Model
    llm = Bedrock(model_id="meta.llama2-70b-chat-v1", client=bedrock, model_kwargs={'max_gen_len':512})
    return llm

prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def save_output_to_pdf(output_text, file_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, output_text)
    pdf.output(file_path)

def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with PDF using AWS Bedrock💁")

    # User uploads PDF file
    uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_pdf is not None:
        with st.spinner("Processing PDF..."):
            docs = data_ingestion(uploaded_pdf)
            get_vector_store(docs)
            st.success("PDF Processed and Vector Store Updated!")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if 'output_text' not in st.session_state:
        st.session_state.output_text = ""

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_claude_llm()
            st.session_state.output_text = get_response_llm(llm, faiss_index, user_question)
            st.write(st.session_state.output_text)
            st.success("Done")


    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llama2_llm()
            st.session_state.output_text = get_response_llm(llm, faiss_index, user_question)
            st.write(st.session_state.output_text)
            st.success("Done")

    if st.session_state.output_text:
        if st.button("Save to PDF"):
            file_path = "output.pdf"
            save_output_to_pdf(st.session_state.output_text, file_path)
            with open(file_path, "rb") as f:
                st.download_button("Download PDF", f, file_name="output.pdf")

if __name__ == "__main__":
    main()
