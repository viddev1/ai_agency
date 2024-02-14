import os
import openai
import webbrowser
import time

from flask import Flask, request, render_template
from openai import AzureOpenAI
from dotenv import load_dotenv, find_dotenv
from typing import List, Optional
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai.chat_models.azure import AzureChatOpenAI

from langchain.prompts import ChatPromptTemplate

_ = load_dotenv(find_dotenv())  # read local .env file
client = AzureOpenAI(
    api_key=os.environ['AZURE_OPENAI_API_KEY'],
    api_version="2024-02-15-preview",
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT']
)
print(client)
app = Flask(__name__)

ASSISTANT_ID = "asst_h1tCGjxfWM7DqmkrFOl4xWP1"  # Replace with your Assistant ID

# open web browser
webbrowser.open('http://localhost:5000', new=2)


class Question(BaseModel):
    section: str = Field(description="section: The section of the questionnaire. \
                                    This could be empty but could also be Screener, \
                                    Main (with or without subsections), Demographic section.\
                                    If not found just ignore.")
    question_number: int = Field(description="Question number in the questionnaire.\
                                            If not found just ignore.")
    question_wording: str = Field(description="Question wording. If not found just ignore.")
    question_type: str = Field(description="Type of question.\
                                        Examples include:  Single Option, Multiple, etc.\
                                        If not found just ignore.")
    answer_options: Optional[List[str]] = Field(description="This is a list of the options the \
                                                respondent can select when \
                                                answering the the question.\
                                                If not found just ignore.")


class Questions(BaseModel):
    questions: List[Question] = Field(description="All the questions in the \
                                                questionnaire.")


# def pretty_print(messages):
#     result = []
#     for m in messages:
#         line = f"{m.role}: {m.content[0].text.value}"
#         print(line)
#         result.append(line)
#     return result

def pretty_print(messages):
    result = []
    for m in messages:
        line = f"{m.content[0].text.value}"
        print(line)
        result.append(line)
    return result


thread = client.beta.threads.create()


@app.route("/", methods=["GET", "POST"])
def chat():
    user_input = ""
    messages = []
    html_output = ""
    if request.method == "POST":
        user_input = request.form["user_input"]
        message = client.beta.threads.messages.create(thread_id=thread.id, role="user",
                                                      content=user_input)
        run = client.beta.threads.runs.create(thread_id=thread.id,
                                              assistant_id=ASSISTANT_ID, )
        run = wait_on_run(run, thread)
        if run.status == 'requires_action':
            function_name = run.required_action.submit_tool_outputs.tool_calls[0].function.name
            print(function_name)
            if function_name == 'format_section':
                arguments = run.required_action.submit_tool_outputs.tool_calls[0].function.arguments
                formatted_response = format_response(arguments)
                print(formatted_response)
                run = client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread.id,
                    run_id=run.id,
                    tool_outputs=[
                        {
                            "tool_call_id": run.required_action.submit_tool_outputs.tool_calls[0].id,
                            "output": formatted_response,
                        }
                    ],
                )
                html_output = formatted_response
                run = wait_on_run(run, thread)
        messages = client.beta.threads.messages.list(thread_id=thread.id, order="asc",
                                                     after=message.id)
        messages = pretty_print(messages)
    return render_template("chat.html", user_input=user_input, bot_response_lines=messages,
                           questionnaire=html_output)


def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run


def get_function_names_from_tools(tools):
    function_names = []
    for tool in tools:
        if tool.type == 'function':
            function_names.append(tool.function.name)
    return function_names


def display_questions(questionnaire: Questions) -> str:
    response = ''
    for question in questionnaire.questions:
        response += f"{question.section}\n"
        response += f"{question.question_number}. {question.question_wording}\n"
        response += f"{question.question_type}\n"
        if question.answer_options is not None:  # add this line
            for option in question.answer_options:
                response += f"{option}\n"
        response += '\n'  # Separate questions with a blank line
    return response


def format_response(arguments_call_parser):
    text = arguments_call_parser
    template = """Use {text} and apply the following \
                format instructions.  \
                \
                {format_instructions}"""
    prompt = ChatPromptTemplate.from_template(template=template)
    parser = PydanticOutputParser(pydantic_object=Questions)
    format_instructions = parser.get_format_instructions()
    model = AzureChatOpenAI(
        azure_deployment="beta-deployment",
        api_version="2024-02-15-preview",
        temperature=0.0
    )
    chain = prompt | model | parser
    out = chain.invoke({"text": text,
                        "format_instructions": format_instructions})
    formatted_out = display_questions(out)
    return formatted_out


if __name__ == "__main__":
    app.run(debug=False)
