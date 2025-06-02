import requests
import os
import openai
from dotenv import load_dotenv
from langchain.tools import Tool
from datetime import datetime
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Load API Keys from .env file
load_dotenv()
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SPOONACULAR_API_KEY:
    raise ValueError("Error: SPOONACULAR_API_KEY is missing. Please check your .env file.")
if not OPENAI_API_KEY:
    raise ValueError("Error: OPENAI_API_KEY is missing. Please check your .env file.")

openai.api_key = OPENAI_API_KEY

def generate_image(query: str):
    """Generate an AI-powered image using OpenAI's latest API."""
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

def get_recipe(query: str):
    """Fetch detailed recipe info including ingredients, instructions, source, image."""
    url = f"https://api.spoonacular.com/recipes/complexSearch?query={query}&apiKey={SPOONACULAR_API_KEY}&number=1"
    response = requests.get(url)

    if response.status_code != 200:
        return {"error": f"Unable to fetch recipe (Status Code: {response.status_code})"}

    data = response.json()
    if not data.get("results"):
        return {"error": "No recipe found for the given query."}

    recipe_id = data["results"][0]["id"]

    details_url = f"https://api.spoonacular.com/recipes/{recipe_id}/information?apiKey={SPOONACULAR_API_KEY}"
    details_response = requests.get(details_url)

    if details_response.status_code != 200:
        return {"error": f"Error fetching detailed recipe info (Status Code: {details_response.status_code})"}

    details_data = details_response.json()

    instructions = details_data.get("instructions", "No instructions available")
    if isinstance(instructions, list):
        instructions = " ".join([step["step"] for step in instructions])

    ingredients = [ingredient["name"] for ingredient in details_data.get("extendedIngredients", [])]
    image_url = details_data.get("image") or generate_image(query)
    source_url = details_data.get("sourceUrl", "")

    return {
        "recipe": details_data.get("title", "Unknown"),
        "preparation_time": details_data.get("readyInMinutes", "Unknown"),
        "ingredients": ingredients,
        "instructions": instructions,
        "image_url": image_url,
        "source": source_url,
    }

def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    return f"Data successfully saved to {filename}"

# Define tools
recipe_tool = Tool(
    name="fetch_recipe",
    func=get_recipe,
    description="Fetches detailed food recipes using Spoonacular API.",
)

image_tool = Tool(
    name="generate_image",
    func=generate_image,
    description="Generates AI-powered images using OpenAI's DALL·E model.",
)

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
