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


async def find_rss_feed(message: str, original_url: str):
    async with MCPServerStdio(
        name="playwright mcp server",
        params={
            "command": "npx",
            "args": ["-y", "@playwright/mcp@latest"],
        },
        client_session_timeout_seconds=30,
    ) as server:
        find_rss_feed_agent = Agent(
            name="find rss feed agent",
            instructions="""
你是一个可以使用 Playwright MCP 工具的浏览智能体，你的任务：给定一个站点 URL，找到其 RSS/Atom 订阅地址。

严格执行下面步骤（不要跳过，也不要在没尝试工具前说“无法”）：
1. 使用浏览器导航工具 (例如 browser_navigate / navigate) 打开用户提供的原始 URL。
2. 获取页面内容：优先调用 browser_snapshot 或 page_source / browser_page_source / browser_content 之类的工具（根据可用列表选择）。
3. 在 HTML 中查找 <link> 标签： rel 包含 alternate 且 type 包含 rss+xml / atom+xml / xml；提取 href。
4. 在正文和 footer / header 中查找包含 rss / feed / 订阅 / atom 关键词的 <a> 链接。
5. 若仍为空：按以下顺序尝试常见候选路径（用浏览器工具逐一访问，若 404 则继续）：
     /feed  /rss  /rss.xml  /feed.xml  /atom.xml  /index.xml  /posts/index.xml  /blog/feed  /feeds/posts/default?alt=rss
6. 对每个候选或发现的链接，访问一次并判定是否为 RSS/Atom：
     判定标准：
         - Content-Type 含 application/rss+xml / application/atom+xml / application/xml / text/xml
         - 或文本前几行包含 <rss  或 <feed  或 <rdf:RDF
7. 汇总结果，选出最“主”订阅（通常是包含最多 <item> 或 <entry> 的那个，或命名为 feed / atom.xml 的）。
8. 输出 JSON（中文说明）格式：
{
    "original_url": "...",
    "feeds": [
        {"url": "...", "valid": true/false, "type": "rss|atom|unknown", "discovered_by": "link_tag|anchor_text|guess_pattern", "item_count": 数字或 null}
    ],
    "primary_feed": "...或null",
    "notes": "如果失败，说明尝试了哪些步骤"
}

重要要求：
- 一定要真正调用工具，除非工具不可用；
- 如果某个工具名称不在你可用的列表里，尝试列出你拥有的工具名称并选择最接近的；
- 不要编造结果；发现不了就列出尝试步骤和失败原因；
- 语言使用中文；
- 在执行前先简要计划（一步式文字），然后开始调用工具。
                        """,
            model=model,
            mcp_servers=[server],
            model_settings=ModelSettings(tool_choice="auto"),
        )
        input_items = [{"role": "user", "content": message}]
        try:
            result = await Runner.run(
                starting_agent=find_rss_feed_agent, input=input_items
            )
            if isinstance(result, dict):
                if result.get("primary_feed") is None:
                    result["primary_feed"] = ""
                if result.get("notes") is None:
                    result["notes"] = ""
            return result
        except Exception as e:
            return {
                "original_url": original_url,
                "feeds": [],
                "primary_feed": "",
                "notes": f"执行查找过程中出现异常: {type(e).__name__}: {e}. 已返回空的结果（表示未找到或页面不支持 RSS/Atom）。"
            }


def main():
    url = sys.argv[1]
    print(f"Processing URL: {url}")

    # You can now use the URL in your message or processing
    message = f"找到这个url对应的网站提供的rss订阅地址: {url}"
    result = asyncio.run(find_rss_feed(message=message, original_url=url))
    print(result)


if __name__ == "__main__":
    main()
