import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from openai import OpenAI

# --- 配置路径与环境 ---
TOOLS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TOOLS_DIR.parent

# 确保能导入同目录下的模块
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

# 尝试导入 MinerU 工具
try:
    from mineru_vlm_tool import mineru_vlm_pdf_to_markdown  # noqa: E402
except ImportError:
    print("[Error] Could not import 'mineru_vlm_tool'. Please ensure dependencies are installed.")
    sys.exit(1)
try:
    from latex_tool import latex_project_to_text  # noqa: E402
except ImportError:
    print("[Error] Could not import 'latex_tool'. Please ensure dependencies are installed.")
    sys.exit(1)

# ================= 显卡与路径配置区域 =================

# 1. 模型路径
MODEL_PATH = "/inspire/hdd/project/exploration-topic/public/downloaded_ckpts/Qwen2.5-7B-Instruct"
SERVED_MODEL_NAME = "Qwen2.5-7B-Instruct"

# 2. 显卡分配策略
VLLM_GPU_ID = "0"
VLLM_TP_SIZE = 1 
MINERU_GPU_ID = "1"

# 3. 服务端口
HOST = "127.0.0.1"
PORT = 8000
SERVER_URL = f"http://{HOST}:{PORT}/v1"

# ====================================================

READY_TIMEOUT_SEC = 300
READY_POLL_SEC = 2


def _start_vllm_server() -> subprocess.Popen:
    """启动 vLLM 服务器"""
    print(f"[System] Starting vLLM server on port {PORT}...")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = VLLM_GPU_ID

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_PATH,
        "--served-model-name", SERVED_MODEL_NAME,
        "--tensor-parallel-size", str(VLLM_TP_SIZE),
        "--host", HOST,
        "--port", str(PORT),
        "--trust-remote-code",
        "--max-model-len", "32768", 
        "--gpu-memory-utilization", "0.9",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",  
    ]

    return subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


def _wait_for_ready(timeout_sec: int = READY_TIMEOUT_SEC) -> None:
    print("[System] Waiting for vLLM server to be ready...")
    deadline = time.time() + timeout_sec
    url = f"{SERVER_URL}/models"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    print("[System] Server is ready!")
                    return
        except Exception: 
            pass
        time.sleep(READY_POLL_SEC)
    raise TimeoutError(f"vLLM server not ready after {timeout_sec}s.")


def _build_tools_schema() -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": "mineru_vlm_pdf_to_markdown",
                "description": "Used to convert PDF files to Markdown content. Use this when user asks to read, parse, summarize or explain a PDF file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdf_path": {
                            "type": "string", 
                            "description": "The absolute file path of the PDF."
                        },
                        "backend": {"type": "string", "enum": ["vlm-transformers"], "default": "vlm-transformers"},
                    },
                    "required": ["pdf_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "latex_project_to_text",
                "description": "Convert a LaTeX project directory or archive to plain text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Directory or archive path (zip/tar.*) containing a LaTeX project."
                        },
                        "engine": {
                            "type": "string",
                            "enum": ["pandoc", "latexml", "plastex", "pylatexenc"],
                            "default": "pandoc"
                        },
                        "fallback_mode": {
                            "type": "string",
                            "enum": ["rule_then_llm", "llm_only", "rule_only"],
                            "description": "Fallback strategy when Pandoc fails."
                        }
                    },
                    "required": ["project_path"]
                },
            },
        },
    ]


def _execute_tool_logic(tool_name: str, args: dict) -> str:
    """执行 MinerU 工具的封装函数 (含修复逻辑)"""
    if tool_name == "latex_project_to_text":
        project_path = args.get("project_path") if isinstance(args, dict) else None
        if not project_path:
            return "Error: project_path is required."
        engine = args.get("engine", "pandoc") if isinstance(args, dict) else "pandoc"
        fallback_mode = args.get("fallback_mode") if isinstance(args, dict) else None
        return latex_project_to_text(project_path, engine=engine, fallback_mode=fallback_mode)

    if tool_name != "mineru_vlm_pdf_to_markdown":
        return f"Error: Unknown tool {tool_name}"

    print(f"\n[System] Tool '{tool_name}' triggered.")
    print(f"[System] Args: {args}")
    print(f"[System] Switching context to GPU {MINERU_GPU_ID}...")

    # 尝试设置环境变量 (注意：如果在 import 时 torch 已初始化，此处可能无效，建议使用子进程隔离)
    os.environ["CUDA_VISIBLE_DEVICES"] = MINERU_GPU_ID
    
    # 强制设置 args
    args["backend"] = "vlm-transformers"

    result = None
    try:
        # 兼容 StructuredTool (LangChain) 和 Runnable 以及普通函数
        if hasattr(mineru_vlm_pdf_to_markdown, "run"):
            result = mineru_vlm_pdf_to_markdown.run(args)
        elif hasattr(mineru_vlm_pdf_to_markdown, "invoke"):
            result = mineru_vlm_pdf_to_markdown.invoke(args)
        else:
            result = mineru_vlm_pdf_to_markdown(**args)
    except Exception as e:
        return f"Error executing MinerU: {str(e)}"

    # 解析结果
    markdown = ""
    if isinstance(result, dict):
        markdown = result.get("markdown", "")
    elif isinstance(result, str):
        try:
            parsed = json.loads(result)
            markdown = parsed.get("markdown", result) if isinstance(parsed, dict) else result
        except json.JSONDecodeError:
            markdown = result
    
    if not markdown:
        return "Tool executed successfully but returned empty content."
    
    print(f"[System] Tool execution finished. Content length: {len(markdown)} chars.")
    return markdown


def chat_loop():
    """交互式对话主循环"""
    client = OpenAI(base_url=SERVER_URL, api_key="EMPTY")
    tools = _build_tools_schema()
    
    # 初始化历史消息
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful AI assistant. You have access to a tool `mineru_vlm_pdf_to_markdown` to read PDF files and a tool `latex_project_to_text` to convert LaTeX projects or archives to plain text. When the user asks you to read/parse a PDF, call the PDF tool. When the user asks to convert a LaTeX project/archive to text, call the LaTeX tool. After the tool returns the content, answer the user's question based on that content."
        }
    ]

    print("\n" + "="*50)
    print(" 🚀 Chat Session Started")
    print(" 💡 Type your message and press Enter.")
    print(" 💡 Example: '请帮我读一下 /path/to/paper.pdf'")
    print(" 💡 Type 'exit' to quit.")
    print("="*50 + "\n")

    while True:
        try:
            # 1. 等待用户输入
            user_input = input("\nUser: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("[System] Exiting chat.")
                break

            # 2. 添加用户消息到历史
            messages.append({"role": "user", "content": user_input})

            # 3. 调用模型 (第一次)
            print("[System] Thinking...")
            response = client.chat.completions.create(
                model=SERVED_MODEL_NAME,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            msg = response.choices[0].message
            messages.append(msg) # 将模型的回复（可能是文本，可能是工具调用）加入历史

            # 4. 检查是否需要调用工具
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    try:
                        func_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        print(f"[Error] Failed to parse arguments for {func_name}")
                        continue

                    # 执行工具
                    tool_output = _execute_tool_logic(func_name, func_args)

                    # 将工具结果加入历史 (role='tool')
                    messages.append({
                        "role": "tool",
                        "content": tool_output,
                        "tool_call_id": tool_call.id
                    })

                # 5. 工具执行完后，再次调用模型以生成最终回答
                print("[System] Feeding tool output back to model...")
                final_response = client.chat.completions.create(
                    model=SERVED_MODEL_NAME,
                    messages=messages, # 此时包含：User -> Assistant(Call) -> Tool(Result)
                    tools=tools
                )
                final_msg = final_response.choices[0].message
                print(f"\nModel: {final_msg.content}")
                messages.append(final_msg) # 保存最终回答
            
            else:
                # 如果没有调用工具，直接打印回答
                print(f"\nModel: {msg.content}")

        except KeyboardInterrupt:
            print("\n[System] Interrupted by user.")
            break
        except Exception as e:
            print(f"\n[Error] Unexpected error: {e}")


def main() -> None:
    server = _start_vllm_server()
    try:
        _wait_for_ready()
        chat_loop()
    finally:
        print("[System] Shutting down server...")
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()

if __name__ == "__main__":
    main()
