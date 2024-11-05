import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

templete='''
다음 내용을 바탕으로 질문에 답변하세요:
<context>
{context}
</context>

Question:{input}
'''

prompt=ChatPromptTemplate.from_template(templete)

llm=ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model_name="gpt-3.5-turbo-1106")

output_parser=StrOutputParser()

loader=PyPDFLoader("VanilaRAG\VanilaRAG\data\schoolscholar.pdf")
pages=loader.load()

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False
)
splits=text_splitter.split_documents(pages)

embedings_model=OpenAIEmbeddings(api_key=OPENAI_API_KEY)
db=Chroma.from_documents(splits,embedings_model)

retriever=db.as_retriever()

document_chain=create_stuff_documents_chain(llm, prompt)
retriever_chain = create_retrieval_chain(retriever, document_chain)

user_input=input("질문을 입력하세요.")
response=retriever_chain.invoke({"input":user_input})
print(response["answer"])