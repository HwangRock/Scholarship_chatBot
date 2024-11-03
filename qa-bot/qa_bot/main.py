import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

prompt=ChatPromptTemplate.from_messages(
    [("system", "당신은 장학금 추천 전문가입니다."),
     ("user","{input}")
    ]
)

llm=ChatOpenAI(api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo-1106")

output_parser=StrOutputParser()

#LCEL
chain=prompt|llm|output_parser

user_input=input("질문을 입력하세요.")
response=chain.invoke({"input":user_input})
print(response)
