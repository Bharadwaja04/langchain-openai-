import streamlit as st
import os
import numpy as np
from typing import Optional
from langchain_community.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import pandas as pd

st.set_page_config(page_title='Airline Reviews Analysis', page_icon=":airplane:", layout="wide", initial_sidebar_state="expanded")




llm = ChatOpenAI(openai_api_key="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")
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

## Embedding Technique Of OPENAI
embeddings=OpenAIEmbeddings(api_key='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')

class Document:
    def __init__(self, page_content, metadata: Optional[dict] = {}):
        self.page_content = page_content
        self.metadata = metadata


# Check if the FAISS index file exists
index_file_path = "faiss_index"
if os.path.exists(index_file_path):
    # Load the existing FAISS index
    new_db = FAISS.load_local(index_file_path, embeddings, allow_dangerous_deserialization=True)
else:
    # Load the data and create a new FAISS index
    df = pd.read_csv('Airline_processed.csv')
    documents = [Document(text) for text in df['text']]
    vector = FAISS.from_documents(documents, embeddings)
    vector.save_local(index_file_path)
    new_db = FAISS.load_local(index_file_path, embeddings, allow_dangerous_deserialization=True)


document_chain = create_stuff_documents_chain(llm, prompt)

retriever = new_db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)



# Add a styled box around the form
with st.container():
    st.title('Airline Reviews Analysis')
    st.write("Enter your question below:")
    
    # Create a form for input submission
    with st.form('input_form'):
        # Add a text input field for the question
        input_prompt = st.text_input('Question:')
        
        # Add a submit button
        submit_button = st.form_submit_button(label='Get Answer')
    
    # Process the form submission
        if submit_button:
        # Display the user input
         with st.container():
            st.write("You:", input_prompt)
        
        # Invoke retrieval chain to get the answer
            response = retrieval_chain.invoke({"input": input_prompt})
        
        # Display the answer
            with st.container():
                st.write("Airline Bot: ", response["answer"])

