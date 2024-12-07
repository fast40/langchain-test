import re
import pathlib

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()


def extract_code_blocks_with_filenames(llm_output):
    # note: this fails if there are three backticks in the code block itself
    matches = re.findall('(.+)\n```(.*)\n([\\s\\S]*?)```', llm_output)  # re.findall returns a list of tuples of the capturing groups in each match

    return tuple({
        'filename': match[0],
        'language': match[1],
        'code': match[2]
    } for match in matches)


def write_to_file(file_path, content):
    path = pathlib.Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as file:
        file.write(content)


class MyCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token, *args, **kwargs):
        print(token, end='')

    def on_llm_end(self, response, *args, **kwargs):
        print('\n\n --- link complete --- \n')


requirements_prompt = ChatPromptTemplate.from_messages([
    ('system', 'The user will describe a web application that they would like to build. Respond by breaking down their description into requirements. Interpret the description literally, meaning that you should only include requirements that are somehow part of the description, even if you think you are leaving out important requirements. Always create as few requirements as possible. Format the requirements as a bullet point list of requirements each starting with "The app shall."'),
    ('user', '{prompt}')
])

subrequirements_prompt = ChatPromptTemplate.from_messages([
    ('system', 'Create a new list that expands on the original by adding sub-requirements to each requirement. The sub-requirements should be the minimal set of requirements that if fulfilled will also fulfill the requirement that they are under. These sub-requirements should be direct logical conclusions of the top requirements. This is not a place for interpretation or inferring what the user might want. This is a logical and deterministic step that can and will leave out "desirable" requirements. There is a massive emphasis on minimal and logical here. Never include HOW the requirement should be fulfilled, only WHAT it needs to accomplish. Format your output as a bullet point list of requirements starting with \'The app shall\' with no text other than this.'),
    ('user', '{requirements}')
])

code_prompt = ChatPromptTemplate.from_messages([
    ('system', 'Create a flask app based on the given requirements. Right before every code block, list the full file path including the filename. The format should look like \n\ntemplates/index.html\n```html\n<html>\n</html>\n```\n\nDo not list any empty directories as code blocks. Simply create them from the flask app. After the app is started, the user should not have to interact with the app in any way except through the web. For example, they should not have to create nonexistent directories through the command line. Do these things automatically if possible, but if that is not possible you must provide a way to do it from within the app.'),  # may want to add the prompt back in here
    ('user', '{requirements}')
])


'''
What should the system look like?
- I need to create the prompt which will result in the best output for the user.
- This has a high degree of overlap with creating the best prompt for ME (or another human).

- I need to create the most descriptive prompt possible.
- ok so I need to address ambiguity.

- What makes a good prompt?
    - says exactly what to do.
    - complete
    - unambiguous
    - does not leave room for interpretation
    - obvious what it's asking

- What makes a bad prompt?
    - unclear what to do.
    - very ambiguous
    - asks to do other things that you don't want (red herrings/tangents/unnecessary things)
    - Gives permission to be loosey goosey with following it

Ultimately what am I looking for?
- low variability in the output that the prompt produces.
- a good output.
- output is clearly what the user asked for (regardless of if that's what they wanted or not)

The behavior I'm looking for is consistent, rule-following form.
'''


# def on_finish(obj):
#     print(obj.outputs)
#
# RunnableLambda(subrequirements_prompt).with_listeners()
#
# print(subrequirements_prompt.invoke('test hellow there').with())


# llm = ChatOpenAI(model='gpt-4o', streaming=True, callbacks=[MyCallback()], n=2)
llm = ChatOpenAI(model='gpt-4o', n=1, top_p=0)
# llm = ChatOpenAI(model='gpt-4o', n=1, temperature=1.5)

chain = requirements_prompt | llm | subrequirements_prompt | llm
 # | subrequirements_prompt | llm | code_prompt | llm

# code = chain.invoke({ 'prompt': 'Create an image gallery that I can upload images to and then view.' })

x = input('enter something: ')

# code = llm.generate([requirements_prompt.invoke({ 'prompt': 'Create an image gallery that I can upload images to and then view.' })])
# code = llm.generate([requirements_prompt.invoke({ 'prompt': x })])
code = chain.invoke(x)

# for generation in code.generations[0]:
#     if generation.text.count('\n') != 1:
#         print(generation.text)
#         print('--')
#
#     print(generation.text)
#     print('--')

print(code.content)


# code_blocks = extract_code_blocks_with_filenames(code)

# for code_block in code_blocks:
#     write_to_file('test/' + code_block['filename'], code_block['code'])
