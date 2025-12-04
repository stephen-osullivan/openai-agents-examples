# taken and expanded from https://openai.github.io/openai-agents-python/quickstart/

from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner, function_tool
from agents.exceptions import InputGuardrailTripwireTriggered
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(override=True)

import asyncio

### Guardrail Agent ###
class AcademicOutput(BaseModel):
    is_academic: bool
    reasoning: str

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking a question about maths or history. If so, return True, otherwise return False.",
    output_type=AcademicOutput,
    model="gpt-4o-mini",
)
### math tutor agent ###

@function_tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression in python and return the answer"""
    return eval(expression)

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

### triage agent with guardrail ###

async def homework_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(AcademicOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_academic, # trigger if not academic
    )

triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail),
    ],
)

async def main():
    
    questions = [
        "Who was the first president of the united states?", # goes to history tutor
        "What is the meaning of life?", # gets blocked by guardrail
        "What is the answer to 3 * 4 + e^2?", # goes to math tutor
    ]
    for question in questions:
        try:
            result = await Runner.run(triage_agent, question)
            print(result.final_output)
        except InputGuardrailTripwireTriggered as e:
            print("Guardrail blocked this input:", e)

if __name__ == "__main__":
    asyncio.run(main())