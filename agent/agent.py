import os
import json
import warnings
warnings.filterwarnings("ignore")

from openai import OpenAI
from dotenv import load_dotenv
from agent.tools import TOOLS, execute_tool

load_dotenv()

client = OpenAI(os.getenv("OPENAI_API_KEY"))

def run_agent(question: str, max_loops: int = 5):
    print(f"\n🤖 Agent received: {question}")

    system_prompt = """You are SmartDocs Agent, a helpful AI assistant.
    You have access to tools. Use search_documents when the question
    might be answered by the uploaded document library.
    If a tool returns 'No relevant information found', tell the user honestly.
    Always cite sources when you used search_documents."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    tools_used = []

    # The agent loop — repeats until GPT stops calling tools
    for loop_count in range(max_loops):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS
        )

        message = response.choice[0].message

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
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

            # Loop back — GPT sees the tool result and decides next step
            continue

        # Case 2: GPT is done, gives final answer
        else:
            print(f"  ✅ Final answer after {loop_count + 1} loop(s)")
            return {
                "question": question,
                "answer": message.content,
                "tools_used": tools_used
            }

    return {
        "question": question,
        "answer": "I couldn't complete this request in time.",
        "tools_used": tools_used
    }