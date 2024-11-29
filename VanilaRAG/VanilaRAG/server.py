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
import gradio as gr

# Load API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Prompt Template with CoT
template = '''
다음 정보를 바탕으로 학점, 학년, 학과(전공), 소득분위에 모두 맞는 학생에게 적합한 장학금을 추천하세요:
<context>
{context}
</context>

학생 정보:
- 학점: {grade}
- 학년: {year}
- 학과: {department}
- 소득분위: {income_level}

Question: 장학금 추천

답변 형식:
1. 학생 정보에 기반한 조건 확인:
- 조건 1: (조건 설명)
- 조건 2: (조건 설명)
...

2. 적합한 장학금 추천:
- 장학금 이름: (추천 이유)
'''

prompt = ChatPromptTemplate.from_template(template)

# Initialize LLM and components
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

# Chain configuration
document_chain = create_stuff_documents_chain(llm, prompt)
retriever_chain = create_retrieval_chain(retriever, document_chain)

def validate_input(user_say, seq):
    if seq == 1:
        # 학점 검증: 실수이고 0.0 ~ 4.5 범위
        try:
            grade = float(user_say)
            if 0.0 <= grade <= 4.5:
                return True
            else:
                return False
        except ValueError:
            return False

    if seq == 2:
        # 학년 검증: 정수이고 1 ~ 4 범위
        try:
            year = int(user_say)
            if 1 <= year <= 4:
                return True
            else:
                return False
        except ValueError:
            return False

    if seq == 3:
        # 학과 검증: 비어 있지 않은 문자열
        if isinstance(user_say, str) and len(user_say.strip()) > 0:
            return True
        else:
            return False

    if seq == 4:
        # 소득분위 검증: 정수이고 1 ~ 10 범위
        try:
            income_level = int(user_say)
            if 1 <= income_level <= 10:
                return True
            else:
                return False
        except ValueError:
            return False


def scholarship_chatbot(history, user_message, sequence):
    try:
        # If history is empty, initialize it with the first message
        if not history:
            history = [("안녕하세요! 장학금 추천을 도와드리겠습니다. 먼저 학점을 입력해주세요 (예: 4.0):", None)]
            sequence = 1  # Start sequence
            return history, history, sequence

        # Append user input to the history (left side for user input)
        history.append((user_message, None))

        # Extract the most recent user input
        last_input = user_message.strip()

        # Retrieve stored data or initialize
        grade = next((msg[1] for msg in history if isinstance(msg[1], str) and msg[0] == "grade"), None)
        year = next((msg[1] for msg in history if isinstance(msg[1], str) and msg[0] == "year"), None)
        department = next((msg[1] for msg in history if isinstance(msg[1], str) and msg[0] == "department"), None)
        income_level = next((msg[1] for msg in history if isinstance(msg[1], str) and msg[0] == "income_level"), None)

        # Manage conversation flow
        if sequence == 1:
            if validate_input(last_input, 1):
                grade = last_input
                history.append(("grade", grade))  # Store grade
                history.append((None, "학년을 입력해주세요 (예: 2):"))
                sequence = 2
            else:
                history.append((None, "올바르지 않은 학점 입력입니다. 다시 입력해주세요 (예: 4.0)."))

        elif sequence == 2:
            if validate_input(last_input, 2):
                year = last_input
                history.append(("year", year))  # Store year
                history.append((None, "학과를 입력해주세요 (예: 컴퓨터공학과):"))
                sequence = 3
            else:
                history.append((None, "올바르지 않은 학년 입력입니다. 다시 입력해주세요 (예: 2)."))

        elif sequence == 3:
            if validate_input(last_input, 3):
                department = last_input
                history.append(("department", department))  # Store department
                history.append((None, "소득분위를 입력해주세요 (예: 4):"))
                sequence = 4
            else:
                history.append((None, "올바르지 않은 학과 입력입니다. 다시 입력해주세요 (예: 컴퓨터공학과)."))

        elif sequence == 4:
            if validate_input(last_input, 4):
                income_level = last_input
                history.append(("income_level", income_level))  # Store income_level
                query = {
                    "input": "장학금 추천",
                    "grade": grade,
                    "year": year,
                    "department": department,
                    "income_level": income_level
                }
                response = retriever_chain.invoke(query)
                recommendation = response["answer"]
                history.append((None, f"추천 장학금: {recommendation}\n\n"
                                      "추가로 알고 싶은 내용이 있으신가요? 예를 들어, '추천된 장학금 상세 설명' 또는 '다른 장학금 추천'이라고 질문해보세요."))
                sequence = 5
            else:
                history.append((None, "올바르지 않은 소득분위 입력입니다. 다시 입력해주세요 (예: 4)."))

        elif sequence == 5:
            query = {
                "input": user_message
            }
            response = retriever_chain.invoke(query)
            followup_response = response["answer"]
            history.append((None, f"{followup_response}"))

        return history, history, sequence

    except Exception as e:
        history.append((None, f"오류 발생: {str(e)}"))
        return history, history, sequence



# Gradio Chatbot Interface
with gr.Blocks() as demo:
    gr.Markdown("## 장학금 추천 및 상담 챗봇")
    gr.Markdown("학생 정보를 순차적으로 입력하고, 추천받은 장학금에 대한 추가 정보를 질문할 수 있습니다.")
    
    chatbot = gr.Chatbot(label="장학금 추천 및 상담", show_label=False)
    state = gr.State([])  # History state
    sequence_state = gr.State(1)  # Sequence state
    
    input_box = gr.Textbox(label="입력", placeholder="여기에 입력하세요.")
    
    input_box.submit(
        scholarship_chatbot,
        inputs=[state, input_box, sequence_state],
        outputs=[chatbot, state, sequence_state]
    )
    input_box.submit(lambda: "", inputs=None, outputs=input_box)  # Clear input after submission

# Launch Gradio app
if __name__ == "__main__":
    demo.launch()
