# taken and expanded from https://openai.github.io/openai-agents-python/quickstart/

from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner, function_tool, WebSearchTool
from agents.exceptions import InputGuardrailTripwireTriggered
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel
from dotenv import load_dotenv
import gradio as gr

import asyncio

load_dotenv(override=True)

### web search tool run by openai###
web_search_tool = WebSearchTool(
    search_context_size='low',
)

### custom (local) calculator tool ###
@function_tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression in python and return the answer"""
    return eval(expression)

### Guardrail Agent ###
class ValidQueryOutput(BaseModel):
    is_valid_query: bool
    reasoning: str

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking a valid question about maths, history or current affairs. \
        If so, return True, otherwise return False.",
    model="gpt-4o-mini",
    output_type=ValidQueryOutput,
)
### math tutor agent ###


math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples. \
        Use the calculator tool to evaluate mathematical expressions. \
        After evaluating the final answer return a step by step explanation of how you arrived at the answer.",
    model="gpt-4o-mini",
    tools=[calculator],
)

### history tutor agent ###
history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
)

current_affairs_agent = Agent(
    name="Current Affairs Agent",
    handoff_description="Specialist agent for current affairs questions",
    instructions="You provide assistance with current affairs queries. Explain important events and context clearly. \
        Use the web search tool to find the answer to the user's question.",
    model="gpt-4o-mini",
    tools=[web_search_tool],
)

### triage agent with guardrail ###

async def homework_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(ValidQueryOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_valid_query, # trigger if a valid query
    )

triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's question",
    handoffs=[history_tutor_agent, math_tutor_agent, current_affairs_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail),
    ],
)

async def answer_question_stream(question: str):
    """Stream updates for a single question through the triage agent."""
    question = (question or "").strip()
    if not question:
        yield "Please enter a question."
        return

    # quick initial response so the UI updates immediately
    yield "Working on it..."

    try:
        # Most agents are not token-streaming yet; we still expose a generator
        # so the Gradio UI can stream incremental messages.
        result = await Runner.run(triage_agent, question)
        yield str(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        yield f"Guardrail blocked this input: {e}"
    except Exception as e:  # keep app resilient
        yield f"Something went wrong: {e}"

### Gradio App
def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Agents Homework Helper") as demo:
        gr.Markdown("# Homework Helper\nAsk about maths, history, or current affairs.")
        question_box = gr.Textbox(label="Your question", placeholder="e.g. What is the answer to 3 * 4 + e^2?")
        ask_btn = gr.Button("Ask")
        output_box = gr.Markdown()

        ask_btn.click(fn=answer_question_stream, inputs=question_box, outputs=output_box)
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.queue()  # allow async handler
    demo.launch()