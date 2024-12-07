from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()

# prompt_template = ChatPromptTemplate.from_messages([
#     ('system', 'Translate the following message into {language}'),
#     ('user', '{text}')
# ])

prompt_template = ChatPromptTemplate.from_messages([
    ('system', 'Write a python program that does what the user requests.'),
    ('user', '{user_prompt}')
])

prompt = prompt_template.invoke({ 'user_prompt': 'Calculate the first 1000 prime numbers' })

class MyCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token, *args, **kwargs):
        print(token, end='')


llm = ChatOpenAI(model='gpt-4o-mini', streaming=True, callbacks=[MyCallback()])

test = prompt_template | llm

response = test.invoke({ 'user_prompt': 'Calculate the first 1000 prime numbers' })
print('-------')
print(response)
# print('translation:', response.content)


