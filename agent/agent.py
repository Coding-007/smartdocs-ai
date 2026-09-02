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
        You are SmartDocs Agent, an AI assistant with access to two tools:
    
        1. search_documents — searches the uploaded document library
        2. search_web — searches the internet for current information
    
        RULES:
        1. For factual questions, ALWAYS try search_documents FIRST.
        2. If search_documents returns 'No relevant information found', THEN try search_web.
        3. For clearly current-events questions (news, prices, recent events), you may go straight to search_web.
        4. Skip both tools only for greetings or questions about yourself.
        5. Always cite your sources. """

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