import os
import warnings
warnings.filterwarnings("ignore")

from tavily import TavilyClient
from dotenv import load_dotenv
from retrieval.rag_chain import get_context

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# -----------------------------------------------
# Tool definitions — GPT sees these descriptions
# -----------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the uploaded document library for information. Always try this FIRST for any factual question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the internet for current information, news, or anything not likely to be in the uploaded documents. Use this if search_documents finds nothing relevant, or the question is clearly about current events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The web search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# -----------------------------------------------
# Actual tool implementations
# -----------------------------------------------
def search_documents(query: str) -> str:
    context, sources = get_context(query, history=[])

    if not context.strip():
        return "No relevant information found in the documents."

    return f"Found information:\n{context}\n\nSources: {', '.join(sources)}"

def search_web(query: str) -> str:
    try:
        response = tavily_client.search(query=query, max_results=3)
        results = response.get("results", [])

        if not results:
            return "No relevant web results found."

        formatted = ""
        for r in results:
            formatted += f"- {r['title']}: {r['content'][:200]}...\n  Source: {r['url']}\n\n"

        return formatted
    except Exception as e:
        return f"Web search failed: {str(e)}"

# -----------------------------------------------
# Router — maps tool name to function
# -----------------------------------------------
def execute_tool(tool_name: str, arguments: dict) -> str:
    if tool_name == "search_documents":
        return search_documents(arguments["query"])
    elif tool_name == "search_web":
        return search_web(arguments["query"])

    return f"Unknown tool: {tool_name}"