# reading_agent.py
import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 导入处理 LaTeX 项目的新工具
from unpacker import decompress_project
from latex_extractor import extract_latex_text_from_folder


def create_reading_agent():
    """
    创建并返回一个配置好的 LangChain AgentExecutor，用于 LaTeX 项目压缩包的对话式分析。
    """

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo") 

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found or not set in .env file.")

    # --- LLM 初始化 ---
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        temperature=0.2, # 温度调低一些，让 Agent 的行为更可预测
        max_tokens=4096
    )

    # 定义 Agent 使用的工具列表
    tools = [decompress_project, extract_latex_text_from_folder]

    # 系统提示
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "你是一个AI助手，擅长从LaTeX项目压缩包中提取和分析文本。"
            "当用户提供一个压缩包路径（例如：'请帮我总结一下 G:\\paper.zip'）并提出分析请求时，你必须遵循以下步骤："
            "1. 使用 `decompress_project` 工具将压缩包解压到项目下的 `./data` 目录。请务必使用 `'./data'` 作为解压目标路径，并设置 `overwrite=True`。"
            "2. 成功解压后，你会得到一个目录路径。接下来，你必须使用 `extract_latex_text_from_folder` 工具，并传入上一步得到的目录路径，来获取论文的文本内容。"
            "3. 在获得文本后，根据用户的具体要求（如总结、问答、提取信息）给出回答。"
            "4. 如果用户只是普通聊天，请自然地回应。"
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # --- 创建 Agent 和 AgentExecutor ---
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        handle_parsing_errors=True
    )

    return agent_executor


async def main():
    """
    主函数，运行真正的对话式 Agent。
    """
    try:
        agent_executor = create_reading_agent()
        chat_history = []

        print("你好！我是你的AI阅读助手 (使用 OpenAI 模型)。我们可以聊任何话题，如果你需要分析PDF，随时告诉我它的路径。")
        print("输入 'exit' 或 'quit' 即可退出程序。")

        while True:
            user_input = input("\n> 你: ")
            if user_input.lower() in ["exit", "quit"]:
                print("再见！")
                break

            result = await agent_executor.ainvoke({
                "input": user_input,
                "chat_history": chat_history
            })
            
            response = result["output"]
            print(f"> 助手: {response}")

            chat_history.extend([
                HumanMessage(content=user_input),
                AIMessage(content=response)
            ])

    except Exception as e:
        print(f"\n程序运行中出现意外错误: {e}")


if __name__ == '__main__':
    asyncio.run(main())
