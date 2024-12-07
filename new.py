import re
import shutil
import pathlib

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()


def extract_code_blocks_with_filenames(llm_output):
    # note: this fails if there are three backticks in the code block itself
    matches = re.findall('(.+)\n```(.*)\n([\\s\\S]*?)```', llm_output)  # re.findall returns a list of tuples of the capturing groups in each match

    return tuple({
        'filename': match[0],
        'language': match[1],
        'content': match[2]
    } for match in matches)


def write_to_file(file_path, content):
    path = pathlib.Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as file:
        file.write(content)


class PrintLinkStreamsCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token, *args, **kwargs):
        print(token, end='')

    def on_llm_end(self, response, *args, **kwargs):
        print('\n\n --- link complete --- \n')


def get_prompts():
    '''
    IMPORTANT NOTE: if you include --- anywhere in the file that is not a system prompt header, it has a good chance of messing up the regex!
    If there is stuff going wrong, make sure to check this.
    If you remove this comment you run the risk of introducing a really hard to diagnose problem and wasting a lot of time solving it.
    Do not remove this comment. There is simply no good reason to do so. (I promise you it's true)
    '''
    with open('prompts.txt', 'r') as file:
        return re.findall(r'\n\n--- [\s\S]+? ---\n\n([\s\S]+?)(?=\n\n---|$)', '\n\n' + file.read())


llm = ChatOpenAI(model='gpt-4o', streaming=True, callbacks=[PrintLinkStreamsCallback()], top_p=0)
# llm = ChatOpenAI(model='gpt-4o', streaming=True, top_p=0)

chain = RunnableSequence(*[runnable for prompt in get_prompts() for runnable in (ChatPromptTemplate.from_messages([('system', prompt), ('user', '{x}')]), llm)])

x = input('enter something: ')

code = chain.invoke(x)

print(code.content)
#
# code_blocks = extract_code_blocks_with_filenames(code.content)
#
# shutil.rmtree('application')
#
# for code_block in code_blocks:
#     write_to_file(code_block['filename'], code_block['content'])

