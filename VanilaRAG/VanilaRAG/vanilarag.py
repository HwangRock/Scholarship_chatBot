import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

template = '''
다음 정보를 바탕으로 학점, 학년, 학과(전공), 소득분위에 모두 맞는 학생에게 장학금을 추천하세요:
<context>
{context}
</context>

학생 정보:
- 학점: {grade}
- 학년: {year}
- 학과: {department}
- 소득분위: {income_level}

Question: 장학금 추천
'''

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model_name="gpt-3.5-turbo-1106")
output_parser = StrOutputParser()

loader = PyPDFLoader("VanilaRAG/VanilaRAG/data/scholarship_doc.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False
)

splits = text_splitter.split_documents(pages)

embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
db = Chroma.from_documents(splits, embeddings_model)
retriever = db.as_retriever()

# 체인 설정
document_chain = create_stuff_documents_chain(llm, prompt)
retriever_chain = create_retrieval_chain(retriever, document_chain)

user_data = {
    "grade": None,
    "year": None,
    "department": None,
    "income_level": None
}

# 유저 데이터 확인
def display_user_data(data, seq):
    print("\n입력된 정보를 확인합니다:")
    if seq==1:
        print(f"- 학점: {data['grade']}")
    if seq==2:
        print(f"- 학점: {data['grade']}")
        print(f"- 학년: {data['year']}")
    if seq==3:
        print(f"- 학점: {data['grade']}")
        print(f"- 학년: {data['year']}")
        print(f"- 학과: {data['department']}")
    if seq==4:
        print(f"다음의 정보에 해당하는 장학금을 추천드리겠습니다. 학점: {data['grade']}, 학년: {data['year']}, 학과: {data['department']}, 소득분위: {data['income_level']}")

def validata_input(seq):
    if seq==1:
        while True:
            result=float(input("현재 학점 (예: 4.0): "))
            if result>=0 and result<=4.5:
                return result
            else:
                print("올바르지 못한 입력입니다. 다시 입력해주세요.")

    if seq==2:
         while True:
            result=int(input("현재 학년 (예: 2): "))
            if result>=1 and result<=4:
                return result
            else:
                print("올바르지 못한 입력입니다. 다시 입력해주세요.")

    if seq==3:
         result=input("현재 학과 (예: 컴퓨터공학과): ")
         return result

    if seq==4:
         while True:
            result=int(input("현재 소득분위 (예: 4): "))
            if result>=1 and result<=10:
                return result
            else:
                print("올바르지 못한 입력입니다. 다시 입력해주세요.")

print("장학금 추천 챗봇입니다. 정보를 입력해주세요.\n")
user_data["grade"] = validata_input(1)
display_user_data(user_data,1)
user_data["year"] = validata_input(2)
display_user_data(user_data,2)
user_data["department"] = validata_input(3)
display_user_data(user_data,3)
user_data["income_level"] = validata_input(4)
display_user_data(user_data,4)

response = retriever_chain.invoke({
    "input": "장학금 추천",
    "grade": user_data["grade"],
    "year": user_data["year"],
    "department": user_data["department"],
    "income_level": user_data["income_level"]
})

print("\n추천 장학금:")
print(response["answer"])
