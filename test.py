from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

model = ChatOpenAI(model='gpt-4o-mini')

from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content='Translate the following from English into Italian'),
    HumanMessage(content='hi!'),
]

x = model.invoke(messages)
print(type(x))
print(x)
