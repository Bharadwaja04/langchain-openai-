from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import pandas as pd
llm = ChatOpenAI(openai_api_key="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")
embeddings = OpenAIEmbeddings(openai_api_key = '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')

df2 = pd.read_csv('preprocessed_dataset.csv')
df2['text'] = df2['Airline Name'] + ' ' + df2['Overall_Rating'].astype(str) + ' ' + df2['Review_Title'] + ' ' + df2['Review Date'].astype(str) + ' ' + df2['Recommended'].astype(str) + ' ' + df2['Date Flown'].astype(str) + ' ' + df2['Seat Comfort'].astype(str) + ' ' + df2['Value For Money'].astype(str) + ' ' + df2['Seat Type'] + ' ' + df2['Wifi & Connectivity'].astype(str) + ' ' + df2['Type Of Traveller']


from typing import Optional
from langchain_community.vectorstores.faiss import FAISS

class Document:
    def __init__(self, page_content, metadata: Optional[dict] = {}):
        self.page_content = page_content
        self.metadata = metadata

# Assuming df2['text'] contains strings that represent documents
# You need to convert these strings into Document objects
documents = [Document(text) for text in df2['text']]

# Now you can proceed with the vectorization process
vector = FAISS.from_documents(documents, embeddings)

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)


from langchain_core.documents import Document

document_chain.invoke({
    "input": "how can langsmith help with testing?",
    "context": [Document(page_content="langsmith can let you visualize test results")]
})

from langchain.chains import create_retrieval_chain

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)


response = retrieval_chain.invoke({"input": "What are the major challenges customers are facing? Can you tell me this based on the type of traveler? "})
print(response["answer"])

