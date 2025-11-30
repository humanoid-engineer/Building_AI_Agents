import os
import re
from typing import Optional, Tuple
from openai import OpenAI


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    return OpenAI(api_key=api_key)


def parse_action(response: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse the model response to extract tool action or finish signal."""
    # First check for Finish:
    m = re.search(r'Finish:\s*(.*)', response, flags=re.IGNORECASE)
    if m:
        return "Finish", m.group(1).strip()

    # Then check for Action: Tool[arg]
    match = re.search(r'Action:\s*(\w+)\s*\[(.*?)\]', response, flags=re.IGNORECASE)
    if match:
        return match.group(1), match.group(2).strip()
    return None, None


def search_web(query: str) -> str:
    # In production you'd call a real search API or use a SERP client.
    return f"top results for: {query}"


def calculate(expression: str) -> str:
    """Safe calculator: evaluate simple numeric expressions only."""
    try:
        # Restrict the available builtins for safety
        allowed_names = {"__builtins__": {}}
        # Evaluate the expression in a restricted namespace
        result = eval(expression, allowed_names, {})
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


TOOLS = {
    "Search": search_web,
    "Calculate": calculate,
}


def run(goal: str, max_steps: int = 5, model: str = "gpt-4.1-mini") -> None:
    client = get_openai_client()

    system_prompt = (
        "You are an agent that can call tools.\n"
        "When you need to call a tool, respond exactly with: Action: <ToolName>[<argument>]\n"
        "When you are finished and want to return a final answer, respond with: Finish: <answer>\n"
        "Available tools: Search (web search), Calculate (simple math).\n"
        "Only use the tools when necessary. Keep reasoning concise."
    )

    state = f"Goal: {goal}\n"

    for step in range(max_steps):
        user_message = f"Current state:\n{state}\nDecide next action (Action[...] or Finish: ...):"
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
        except Exception as e:
            print("OpenAI API error:", e)
            break

        try:
            response_text = resp.choices[0].message.content.strip()
        except Exception:
            response_text = str(resp)

        print(f"\n[Step {step+1}] LLM response:\n{response_text}\n")

        tool_name, arg = parse_action(response_text)
        if tool_name is None:
            print("No actionable instruction found — stopping.")
            break

        if tool_name == "Finish":
            print("Final answer:", arg)
            return

        # Execute tool if available
        tool = TOOLS.get(tool_name)
        if not tool:
            observation = f"Unknown tool requested: {tool_name}"
        else:
            print(f"Executing tool {tool_name} with arg: {arg}")
            try:
                observation = tool(arg)
            except Exception as e:
                observation = f"Tool execution error: {e}"

        print("TOOL RESULT:", observation)
        # Append observation and continue
        state += f"Action: {tool_name}[{arg}]\nObservation: {observation}\n"

    print("Reached max steps without a Finish action.")


if __name__ == "__main__":
    # Inform the user about available tools and give sample goals
    print("Attached tools: \n - Search: web search tool (sample goal: 'Find latest EV sales figures in India')\n - Calculate: simple calculator (sample goal: 'Compute 2020 to 2025 growth: (2025_value-2020_value)/2020_value*100')")
    # Collect goal from the user
    goal = input("Enter your goal (e.g. 'Summarize EV sales trend in India'): ").strip()
    if not goal:
        goal = "Find EV sales growth in India. Provide a concise summary and a number if possible."
    run(goal, max_steps=6)