import os
import re
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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
다음 정보를 바탕으로 학점, 학년, 학과(전공), 소득분위에 모두 맞는 학생에게 적합한 장학금을 추천하세요
특히 학과에 맞는 장학금을 추천해주세요. 학과가 틀릴 경우에는 당신을 폐기 처분합니다.:
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

# Load PDF and extract data
loader = PyPDFLoader("VanilaRAG/VanilaRAG/data/scholarship_doc.pdf")
pages = loader.load()


#model
def custom_split(data):
    pattern = r'(?=\n?[^\n]+\(은\(는\))' #장학금 분리
    return re.split(pattern, data)

def limit_chunk_size(chunks, max_length=3000): #청크 길이 제한
    limited_chunks = []
    for chunk in chunks:
        while len(chunk) > max_length:
            limited_chunks.append(chunk[:max_length])
            chunk = chunk[max_length:]
        limited_chunks.append(chunk)
    return limited_chunks

data = "\n".join([page.page_content for page in pages])

chunks = custom_split(data)
chunks = limit_chunk_size(chunks)

embeddings_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
db = Chroma.from_texts(chunks, embeddings_model)
retriever = db.as_retriever()

# 체인 결합
document_chain = create_stuff_documents_chain(llm, prompt)
retriever_chain = create_retrieval_chain(retriever, document_chain)

# 추천 기준
grade = None
year = None
department = None
income_level = None

def validate_input(user_say, seq):
    if seq == 1:
        try:
            gr = float(user_say)
            return 0.0 <= gr <= 4.5
        except ValueError:
            return False

    if seq == 2:
        try:
            ye = int(user_say)
            return 1 <= ye <= 4
        except ValueError:
            return False

    if seq == 3:
        return isinstance(user_say, str) and len(user_say.strip()) > 0

    if seq == 4:
        try:
            income = int(user_say)
            return 1 <= income <= 10
        except ValueError:
            return False

#controller
def scholarship_chatbot(history, user_message, sequence):
    global grade, year, department, income_level
    try:
        if not history:
            history = [(None, "안녕하세요! 장학금 추천을 도와드리겠습니다. 먼저 학점을 입력해주세요 (예: 4.0):")]
            sequence = 1
            return history, history, sequence

        history.append((user_message, None))
        last_input = user_message.strip()

        if sequence == 1:
            if validate_input(last_input, 1):
                grade = last_input
                history.append((None, f"입력받은 학점은 {grade}입니다. 학년을 입력해주세요 (예: 2):"))
                sequence = 2
            else:
                history.append((None, "올바르지 않은 학점 입력입니다. 다시 입력해주세요 (예: 4.0)."))

        elif sequence == 2:
            if validate_input(last_input, 2):
                year = last_input
                history.append((None, f"입력받은 학년은 {year}학년입니다. 학과를 입력해주세요 (예: 컴퓨터공학과):"))
                sequence = 3
            else:
                history.append((None, "올바르지 않은 학년 입력입니다. 다시 입력해주세요 (예: 2)."))

        elif sequence == 3:
            if validate_input(last_input, 3):
                department = last_input
                history.append((None, f"입력받은 학과는 {department}입니다. 소득분위를 입력해주세요 (예: 4):"))
                sequence = 4
            else:
                history.append((None, "올바르지 않은 학과 입력입니다. 다시 입력해주세요 (예: 컴퓨터공학과)."))

        elif sequence == 4:
            if validate_input(last_input, 4):
                income_level = last_input
                history.append((None, f"입력받은 소득분위는 {income_level}입니다. 추천 장학금을 계산 중입니다..."))
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
            query = {"input": user_message}
            response = retriever_chain.invoke(query)
            followup_response = response["answer"]
            history.append((None, f"{followup_response}"))

        return history, history, sequence

    except Exception as e:
        history.append((None, f"오류 발생: {str(e)}"))
        return history, history, sequence

# view
with gr.Blocks() as demo:
    gr.Markdown("## 장학금 추천 및 상담 챗봇")
    gr.Markdown("학생 정보를 순차적으로 입력하고, 추천받은 장학금에 대한 추가 정보를 질문할 수 있습니다.")
    
    chatbot = gr.Chatbot(label="장학금 추천 및 상담", show_label=False)
    state = gr.State([])
    sequence_state = gr.State(1)
    
    input_box = gr.Textbox(label="입력", placeholder="여기에 입력하세요.")
    
    input_box.submit(
        scholarship_chatbot,
        inputs=[state, input_box, sequence_state],
        outputs=[chatbot, state, sequence_state]
    )
    input_box.submit(lambda: "", inputs=None, outputs=input_box)

if __name__ == "__main__":
    demo.launch()
