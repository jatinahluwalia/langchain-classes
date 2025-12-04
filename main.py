from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from pydantic import BaseModel
from dotenv import load_dotenv


class User(BaseModel):
    name: str
    age: int
    address: str


load_dotenv()

client = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", api_key=os.environ["GOOGLE_GENAI_KEY"]
)

parser = PydanticOutputParser(pydantic_object=User)

prompt = ChatPromptTemplate.from_messages(
    messages=[
        {
            "role": "system",
            "content": "Get the data from this format: {formatting_instructions}",
        },
        {"role": "user", "content": "{input}"},
    ]
)

chain = prompt | client | parser

res = chain.invoke(
    {
        "formatting_instructions": parser.get_format_instructions(),
        "input": "I am Jatin Ahluwalia, 23 years old male from Delhi, I work in TCS and now I am switching to Bayer and moving to Bangalore",
    }
)

print(res)
