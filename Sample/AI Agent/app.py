import json
import os
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from main import run_ai_research, llm, prompt, tools
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Error: OPENAI_API_KEY is missing. Please check your .env file.")

openai.api_key = OPENAI_API_KEY

app = Flask(__name__)
CORS(app)

agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=25, max_execution_time=60)

def generate_image_fallback(prompt: str):
    try:
        response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        return response.data[0].url
    except Exception as e:
        return f"Error generating image: {str(e)}"

def is_food_related(query: str) -> bool:
    """Simple heuristic to detect food-related queries."""
    keywords = ["recipe", "cook", "dish", "food", "ingredient", "meal", "cuisine", "bake", "prepare", "kitchen"]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in keywords)

@app.route("/query", methods=["POST"])
def handle_query():
    data = request.get_json(force=True)
    user_query = data.get("query", "").strip()

    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400

    try:
        research_result = run_ai_research(user_query)
        if "error" in research_result:
            return jsonify(research_result), 500

        image_url = research_result.get("image_url", "")
        if not image_url or image_url.startswith("Error"):
            image_url = generate_image_fallback(user_query)

        food_related = is_food_related(user_query)

        response = {
            "topic": research_result.get("topic", user_query),
            "summary": research_result.get("summary", ""),
            "sources": research_result.get("sources", []),
            "image_url": image_url,
            "image_source": "OpenAI DALL·E" if "openai" in image_url.lower() else "Agent Generated / Other"
        }

        # Add ingredients and instructions only if food-related and present
        if food_related:
            response["ingredients"] = research_result.get("ingredients", [])
            response["instructions"] = research_result.get("instructions", "No instructions available.")
        else:
            response["ingredients"] = []
            response["instructions"] = ""

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"Server error processing request: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
