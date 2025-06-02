import json
import os
import openai
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import wiki_tool, save_tool, recipe_tool

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Error: OPENAI_API_KEY is missing. Please check your .env file.")

openai.api_key = OPENAI_API_KEY

def generate_image(query: str):
    try:
        response = openai.images.generate(
            model="dall-e-3",
            prompt=query, 
            n=1, 
            size="1024x1024"
        )
        return response.data[0].url
    except Exception as e:
        return f"Error generating image: {str(e)}"

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    ingredients: list[str] = []
    instructions: str = "No instructions available"
    image_url: str = ""

llm = ChatOpenAI(model="gpt-4o-mini")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
        """You are a research assistant that will help generate a research paper.
        Answer the user query and use necessary tools.
        Wrap the output in this format and provide no other text\n{format_instructions}"""),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [wiki_tool, save_tool, recipe_tool]
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=25, max_execution_time=20)

def run_ai_research(query: str):
    try:
        raw_response = agent_executor.invoke({"query": query})

        if "output" not in raw_response or not raw_response["output"]:
            return {"error": "Agent did not produce a valid output"}

        output_data = raw_response["output"].strip()
        if output_data.startswith("```json"):
            output_data = output_data.replace("```json", "").replace("```", "").strip()

        parsed_data = json.loads(output_data)

        # Ensure all required fields exist
        parsed_data.setdefault("ingredients", [])
        parsed_data.setdefault("instructions", "No instructions available")
        parsed_data.setdefault("image_url", "")

        structured_response = parser.parse(json.dumps(parsed_data))
        return structured_response.dict()

    except json.JSONDecodeError:
        return {"error": "Failed to decode JSON output"}
    except Exception as e:
        return {"error": f"Error parsing response: {str(e)}"}
