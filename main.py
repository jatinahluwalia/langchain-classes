from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

client = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", api_key=os.environ["GOOGLE_GENAI_KEY"]
)

parser = CommaSeparatedListOutputParser()

prompt = ChatPromptTemplate.from_messages(
    messages=[
        {
            "role": "system",
            "content": "Get all the names from the user input and format them using these instructions: {formatting_instructions}",
        },
        {"role": "user", "content": "{input}"},
    ]
)

chain = prompt | client | parser

res = chain.invoke(
    {
        "formatting_instructions": parser.get_format_instructions(),
        "input": "I have banana, apple, orange and grapes",
    }
)

print(res)
