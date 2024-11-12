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

# Prompt Template
template = '''
다음 정보를 바탕으로 장학금을 추천하세요:
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

# Define prompt template
prompt = ChatPromptTemplate.from_template(template)

# Initialize LLM
llm = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model_name="gpt-3.5-turbo-1106")
output_parser = StrOutputParser()

# Load PDF and split text
loader = PyPDFLoader("VanilaRAG/VanilaRAG/data/scholarship_doc.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 청크당 500자
    chunk_overlap=100,  # 청크 간 100자 중첩
    length_function=len,
    is_separator_regex=False
)

splits = text_splitter.split_documents(pages)

# Create embeddings and retriever
embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
db = Chroma.from_documents(splits, embeddings_model)
retriever = db.as_retriever()

# Combine chains
document_chain = create_stuff_documents_chain(llm, prompt)
retriever_chain = create_retrieval_chain(retriever, document_chain)

# User data
user_data = {
    "grade": None,
    "year": None,
    "department": None,
    "income_level": None
}

# Function to display current user data
def display_user_data(data):
    print("\n지금까지 입력된 정보를 정리해볼게요:")
    if data["grade"]:
        print(f"현재 학점은 {data['grade']}점이에요.")
    if data["year"]:
        print(f"학년은 {data['year']}학년이고요.")
    if data["department"]:
        print(f"학과는 {data['department']}에 다니고 있네요.")
    if data["income_level"]:
        print(f"소득분위는 {data['income_level']}로 확인했어요.")

# Chat loop
print("장학금 추천 챗봇입니다. 대화를 종료하려면 '종료'라고 입력해주세요.\n")

while True:
    if not user_data["grade"]:
        user_data["grade"] = input("현재 학점 (예: 4.0): ")
        if user_data["grade"].lower() == "종료":
            print("대화를 종료합니다. 안녕히 가세요!")
            break
        display_user_data(user_data)

    if not user_data["year"]:
        user_data["year"] = input("현재 학년 (예: 2): ")
        if user_data["year"].lower() == "종료":
            print("대화를 종료합니다. 안녕히 가세요!")
            break
        display_user_data(user_data)

    if not user_data["department"]:
        user_data["department"] = input("현재 학과 (예: 컴퓨터공학과): ")
        if user_data["department"].lower() == "종료":
            print("대화를 종료합니다. 안녕히 가세요!")
            break
        display_user_data(user_data)

    if not user_data["income_level"]:
        user_data["income_level"] = input("현재 소득분위 (예: 4): ")
        if user_data["income_level"].lower() == "종료":
            print("대화를 종료합니다. 안녕히 가세요!")
            break
        display_user_data(user_data)

    # All data collected, generate response
    response = retriever_chain.invoke({
        "input": "장학금 추천",
        "grade": user_data["grade"],
        "year": user_data["year"],
        "department": user_data["department"],
        "income_level": user_data["income_level"]
    })

    print("\n추천 장학금:")
    print(response["answer"])

    # Reset for next conversation or continue asking
    proceed = input("\n다른 질문이 있나요? (예/아니요): ").strip().lower()
    if proceed == "아니요" or proceed == "종료":
        print("대화를 종료합니다. 안녕히 가세요!")
        break
    else:
        print("\n새로운 대화를 시작합니다. 필요하면 기존 정보를 입력하지 않아도 됩니다.\n")
