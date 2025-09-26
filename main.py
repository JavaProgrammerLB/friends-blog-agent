import asyncio
import os
import sys
from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    Runner,
    set_tracing_disabled,
)
from agents.mcp import MCPServerStdio
from dotenv import load_dotenv
from openai import AsyncOpenAI

set_tracing_disabled(disabled=True)

load_dotenv()
api_key = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL")
model_name = os.getenv("MODEL_NAME")

openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

model = OpenAIChatCompletionsModel(model=model_name, openai_client=openai_client)


async def chat(message: str):
    friends_blog_agent = Agent(
        name="Friends Blog Agent",
        instructions="""
Please use english.
        """,
        model=model,
    )
    input_items = [{"role": "user", "content": message}]

    result = await Runner.run(starting_agent=friends_blog_agent, input=input_items)
    return result


async def find_rss_feed(message: str):
    async with MCPServerStdio(
        name="playwright mcp server",
        params={
            "command": "npx",
            "args": ["-y", "@playwright/mcp@latest"],
        },
    ) as server:
        find_rss_feed_agent = Agent(
            name="find rss feed agent",
            instructions="""
    Please use chinese.
            """,
            model=model,
            mcp_servers=[server],
            model_settings=ModelSettings(tool_choice="auto"),
        )
        input_items = [{"role": "user", "content": message}]

        result = await Runner.run(starting_agent=find_rss_feed_agent, input=input_items)
        return result


def main():
    url = sys.argv[1]
    print(f"Processing URL: {url}")

    # You can now use the URL in your message or processing
    message = f"使用playwright mcp server的帮助找到这个url对应的网站提供的rss订阅地址: {url}"
    result = asyncio.run(find_rss_feed(message))
    print(result)


if __name__ == "__main__":
    main()
