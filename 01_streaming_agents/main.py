import asyncio
import gradio as gr
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, Runner
from dotenv import load_dotenv

load_dotenv(override=True)

mandarin_agent = Agent(
        name="Mandarin Agent",
        instructions="You are a Mandarin speaking chatbot.",
        model = "gpt-4o-mini",
    )

english_agent = Agent(
        name="English Agent",
        instructions="You are a English speaking chatbot.",
        model = "gpt-4o-mini",
    )

triage_agent = Agent(
        name="Triage Agent",
        instructions="Please pass Mandarin related queries to the Mandarin Agent and English related queries to the English Chatbot.",
        model = "gpt-4o-mini",
        handoffs=[mandarin_agent, english_agent]
    )

async def chat(message, history):
    # append chat hsitory to message
    new_input = [{"role": m["role"], "content": m["content"][0]['text']} for m in history] + [{"role": "user", "content":message}]
    
    # stream response
    result = Runner.run_streamed(triage_agent, input=new_input)
    response = ""
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            response += event.data.delta
            yield response


if __name__ == "__main__":
    gr.ChatInterface(
        fn=chat, 
    ).launch()