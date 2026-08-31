import warnings
warnings.filterwarnings("ignore")

from retrieval.rag_chain import get_context

# -----------------------------------------------
# Tool definitions — this is what GPT "sees"
# Same shape as Laravel route definitions,
# just describing what's callable and its params
# -----------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the uploaded documents to find relevant information. Use this whenever the user asks a factual question that might be answered by the document library.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query — should capture the key topic of the user's question"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# -----------------------------------------------
# Actual Python functions that execute when
# GPT decides to call a tool
# -----------------------------------------------
def search_documents(query: str) -> str:

    context, sources = get_context(query, history=[])

    if not context.strip():
        return "No relevant information found in the documents."

    result = f"Found information:\n{context}\n\nSources: {', '.join(sources)}"
    return result

# -----------------------------------------------
# Router — maps tool name to actual function
# -----------------------------------------------
def execute_tool(tool_name: str, arguments: dict) -> str:
    if tool_name == "search_documents":
        return search_documents(arguments['query'])

    return f"Unknown tool: {tool_name}"