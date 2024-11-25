from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

prompt_template = ChatPromptTemplate.from_messages([
    ('system', 'Translate the following message into {language}'),
    ('user', '{text}')
])

message = input('enter a message: ')
language = input('enter a language to translate the message into: ')

prompt = prompt_template.invoke({ 'language': language, 'text': message })

llm = ChatOpenAI(model='gpt-4o-mini')
response = llm.invoke(prompt)

print('translation:', response.content)

