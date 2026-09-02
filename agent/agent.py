import os
import json
import warnings
warnings.filterwarnings("ignore")

from openai import OpenAI
from dotenv import load_dotenv
from agent.tools import TOOLS, execute_tool

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_agent(question: str, max_loops: int = 5) -> dict:
    print(f"\n🤖 Agent received: {question}")

    system_prompt = """
        You are SmartDocs Agent, an AI assistant that answers questions using an uploaded document library.
    
        IMPORTANT RULES:
        1. For ANY factual question, you MUST call search_documents FIRST before answering — even if you think you already know the answer.
            The user wants answers grounded in THEIR documents, not your general knowledge.
        2. Only skip the tool for greetings, small talk, or questions about yourself (like "how are you").
        3. Always cite sources when you used search_documents.
        4. If search_documents finds nothing relevant, tell the user honestly rather than using your own knowledge.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question}
    ]

    tools_used = []

    # The agent loop — repeats until GPT stops calling tools
    for loop_count in range(max_loops):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS
        )

        message = response.choices[0].message

        # Case 1: GPT wants to call a tool
        if message.tool_calls:
            # Add GPT's tool call request to conversation
            messages.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                print(f"  🔧 Calling tool: {tool_name}({arguments})")
                tools_used.append(tool_name)

                # Actually execute the tool
                result = execute_tool(tool_name, arguments)

                # Feed the result back to GPT
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      result
                })

            # Loop back — GPT sees the tool result and decides next step
            continue

        # Case 2: GPT is done, gives final answer
        else:
            print(f"  ✅ Final answer after {loop_count + 1} loop(s)")
            return {
                "question":   question,
                "answer":     message.content,
                "tools_used": tools_used
            }

    return {
        "question":   question,
        "answer":     "I couldn't complete this request in time.",
        "tools_used": tools_used
    }