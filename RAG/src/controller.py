import os
from PDFmodel import ScholarshipModel
from view import ScholarshipView
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_openai import ChatOpenAI

class ScholarshipController:
    def __init__(self, pdf_path):
        self.model = ScholarshipModel(pdf_path)
        self.grade = None
        self.year = None
        self.department = None
        self.income_level = None
        self.sequence = 1 
        
        self.llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0, model_name="gpt-3.5-turbo-1106")
        self.prompt = ChatPromptTemplate.from_template('''
           다음 정보를 바탕으로 학점, 학년, 학과(전공), 소득분위에 모두 맞는 학생에게 적합한 장학금을 추천하세요
            특히 해당 학생의 학과 위주에 맞는 장학금을 더욱 추천해주세요. 학과가 틀릴 경우에는 당신을 폐기 처분합니다.:
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
        ''')
        self.document_chain = None
        self.retriever_chain = None

    def prepare_model(self):
        self.model.load_and_prepare_data()
        retriever = self.model.get_retriever()
        self.document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.retriever_chain = create_retrieval_chain(retriever, self.document_chain)

    def validate_input(self, user_input, seq):
        if seq == 1: 
            try:
                grade = float(user_input)
                return 0.0 <= grade <= 4.5
            except ValueError:
                return False
        if seq == 2: 
            try:
                year = int(user_input)
                return 1 <= year <= 4
            except ValueError:
                return False
        if seq == 3: 
            return isinstance(user_input, str) and len(user_input.strip()) > 0
        if seq == 4:
            try:
                income = int(user_input)
                return 1 <= income <= 10
            except ValueError:
                return False
        return False

    def handle_chat(self, history, user_message, sequence):
        try:
            if not history:
                history = [(None, "안녕하세요! 장학금 추천을 도와드리겠습니다. 먼저 학점을 입력해주세요 (예: 4.0):")]
                sequence = 1
                return history, history, sequence

            history.append((user_message, None))
            last_input = user_message.strip()

            if sequence == 1 and self.validate_input(last_input, sequence):
                self.grade = last_input
                history.append((None, f"학점 {self.grade} 입력받았습니다. 학년을 입력해주세요 (예: 2):"))
                return history, history, 2
            elif sequence == 2 and self.validate_input(last_input, sequence):
                self.year = last_input
                history.append((None, f"학년 {self.year} 입력받았습니다. 학과를 입력해주세요 (예: 컴퓨터공학과):"))
                return history, history, 3
            elif sequence == 3 and self.validate_input(last_input, sequence):
                self.department = last_input
                history.append((None, f"학과 {self.department} 입력받았습니다. 소득분위를 입력해주세요 (예: 4):"))
                return history, history, 4
            elif sequence == 4 and self.validate_input(last_input, sequence):
                self.income_level = last_input
                history.append((None, "장학금 추천 결과를 계산 중입니다..."))
                query = {
                    "input": "장학금 추천",
                    "grade": self.grade,
                    "year": self.year,
                    "department": self.department,
                    "income_level": self.income_level
                }
                response = self.retriever_chain.invoke(query)
                recommendation = response.get("answer", "추천 결과를 찾을 수 없습니다.")
                history.append((None, f"추천 장학금: {recommendation}"))
                return history, history, 5
            else:
                history.append((None, "입력값이 잘못되었습니다. 다시 시도해주세요."))
                return history, history, sequence
        except Exception as e:
            history.append((None, f"오류 발생: {e}"))
            return history, history, sequence

    def run(self):
        view = ScholarshipView(self.handle_chat)
        demo = view.create_ui()
        demo.launch()

if __name__ == "__main__":
    controller = ScholarshipController("RAG/src/data/scholarship_doc.pdf")
    controller.prepare_model()
    controller.run() 
