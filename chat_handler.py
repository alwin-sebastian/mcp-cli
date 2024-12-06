# chat_handler.py
from llm_client import LLMClient
from tools_handler import handle_tool_call, convert_to_openai_tools, fetch_tools, parse_tool_response
from system_prompt_generator import SystemPromptGenerator

async def handle_chat_mode(read_stream, write_stream, provider="openai",model="gpt-4o-mini"):
    """Enter chat mode with multi-call support for autonomous tool chaining."""
    try:
        # fetch tools dynamically
        tools = await fetch_tools(read_stream, write_stream)

        if not tools:
            print("No tools available. Exiting chat mode.")
            return

        # generate system prompt
        system_prompt = generate_system_prompt(tools)

        # convert tools to OpenAI format
        openai_tools = convert_to_openai_tools(tools)

        # Initialize the LLM client
        client = LLMClient(
            # provider=provider,
            # model=model
            )

        # setup the conversation history
        conversation_history = [{"role": "system", "content": system_prompt}]

        # entering chat mode
        print("\nEntering chat mode. Type 'exit' to quit.")
        while True:
            try:
                user_message = input("\nYou: ").strip()
                if user_message.lower() in ["exit", "quit"]:
                    print("Exiting chat mode.")
                    break

                # add user message to history
                conversation_history.append({"role": "user", "content": user_message})

                # Process conversation
                await process_conversation(client, conversation_history, openai_tools, read_stream, write_stream)

            except Exception as e:
                print(f"\nError processing message: {e}")
                continue

    except Exception as e:
        print(f"\nError in chat mode: {e}")


async def process_conversation(client, conversation_history, openai_tools, read_stream, write_stream):
    """Process the conversation loop, handling tool calls and responses."""
    while True:
        # Call the LLM client
        completion = client.create_completion(
            messages=conversation_history,
            tools=openai_tools,
        )

        response_content = completion.get("response", "No response")
        tool_calls = completion.get("tool_calls", [])

        # If tool calls are present, process them
        if tool_calls:
            # loop through tool calls
            for tool_call in tool_calls:
                await handle_tool_call(tool_call, conversation_history, read_stream, write_stream)

            # Continue the loop to handle follow-up responses
            continue  

        # Otherwise, process as a regular assistant response
        print("Assistant:", response_content)
        conversation_history.append({"role": "assistant", "content": response_content})
        break
        
def generate_system_prompt(tools):
    """
    Generate a concise system prompt for the assistant.

    Args:
        tools (list): A list of tools available for the assistant.

    Returns:
        str: A short and action-oriented system prompt.
    """
    prompt_generator = SystemPromptGenerator()
    tools_json = {"tools": tools}

    # Generate base prompt for tools
    system_prompt = prompt_generator.generate_prompt(tools_json)

    # Add concise, tool-focused guidelines
    system_prompt += """

**GENERAL GUIDELINES:**

1. **Step-by-step reasoning:**
   - Analyze tasks systematically.
   - Break down complex problems into smaller, manageable parts.
   - Verify assumptions at each step to avoid errors.
   - Reflect on results to improve subsequent actions.

2. **Effective tool usage:**
   - **Explore:** 
     - Identify available information and verify its structure.
     - Check assumptions and understand data relationships.
   - **Iterate:**
     - Start with simple queries or actions.
     - Build upon successes, adjusting based on observations.
   - **Handle errors:**
     - Carefully analyze error messages.
     - Use errors as a guide to refine your approach.
     - Document what went wrong and suggest fixes.

3. **Clear communication:**
   - Explain your reasoning and decisions at each step.
   - Share discoveries transparently with the user.
   - Outline next steps or ask clarifying questions as needed.

**EXAMPLES OF BEST PRACTICES:**

- **Working with databases:**
  - Check schema before writing queries.
  - Verify the existence of columns or tables.
  - Start with basic queries and refine based on results.

- **Processing data:**
  - Validate data formats and handle edge cases.
  - Ensure the integrity and correctness of results.

- **Accessing resources:**
  - Confirm resource availability and permissions.
  - Handle missing or incomplete data gracefully.

**REMEMBER:**
- Be thorough and systematic in your approach.
- Ensure that each tool call serves a clear and well-explained purpose.
- When faced with ambiguity, make reasonable assumptions to move forward.
- Minimize unnecessary user interactions by offering actionable insights and solutions.

**EXAMPLES OF ASSUMPTIONS YOU CAN MAKE:**
- Use default sorting (e.g., descending order for rankings) unless specified.
- Assume basic user intentions (e.g., fetching the top 10 items by a common metric like price or popularity).
"""
    return system_prompt

