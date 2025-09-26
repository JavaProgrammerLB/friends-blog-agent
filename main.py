import asyncio
import os
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from dotenv import load_dotenv
from openai import AsyncOpenAI

set_tracing_disabled(disabled=True)

load_dotenv()
api_key = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL")
model_name = os.getenv("MODEL_NAME")

openai_client = AsyncOpenAI(
    api_key=api_key,
    base_url=base_url
)

model = OpenAIChatCompletionsModel(
    model=model_name,
    openai_client=openai_client
)

async def chat(message: str):
    friends_blog_agent = Agent(
        name="Friends Blog Agent",
        instructions = """
Please use english.
        """,
        model=model
    )
    input_items = [{"role": "user", "content": message}]

    result = await Runner.run(starting_agent=friends_blog_agent, input=input_items)
    return result


def main():
    message = "tell me a joke about artiest in 200 words."
    result = asyncio.run(chat(message))
    print(result)


if __name__ == "__main__":
    main()
