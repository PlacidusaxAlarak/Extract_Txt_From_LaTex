# LaTeX Project Extractor & Reading Agent

This project provides a conversational AI agent designed to extract and analyze text from compressed LaTeX project archives. You can interact with the agent in natural language, ask it to process a `.zip` or `.tar.gz` file containing a LaTeX project, and then ask questions about its content, request summaries, and more.

## Features

- **Conversational Interface**: Interact with the extractor through a simple chat interface.
- **Archive Support**: Automatically decompresses `.zip`, `.tar.gz`, and other common archive formats using the `unpacker` tool.
- **Robust LaTeX Parsing**: Uses Pandoc as the backend to convert LaTeX into a structured format before text extraction.
- **Extensible Filtering**: Includes a Lua filter system (`filters/`) to preprocess and sanitize LaTeX source, allowing for custom handling of non-standard commands.
- **Powered by LLMs**: Utilizes a large language model (via LangChain and OpenAI) to understand instructions and analyze the extracted text.

## Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3.8+**
2.  **Pandoc**: This is a critical external dependency. You must install it separately and ensure it's available in your system's PATH. You can download it from [pandoc.org](https://pandoc.org/installing.html).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd Extract_Text_From_Latex
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Python dependencies:**
    The `requirements.txt` file is minimal. You will also need to install the `langchain` packages used in `reading_agent.py`.
    ```bash
    pip install pypandoc python-dotenv langchain langchain-openai
    ```

4.  **Set up your environment variables:**
    Create a file named `.env` in the project root and add your OpenAI API key:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    ```

## Usage

Run the main agent script from your terminal:

```bash
python reading_agent.py
```

Once the agent is running, you can interact with it. To analyze a LaTeX project, provide the full path to the compressed archive. The agent will perform a two-step process: first decompressing the file, then extracting the text.

**Example Interaction:**

```
> 你: 请帮我总结一下 G:\papers\my_awesome_paper.zip

> 助手: [Agent's thought process and tool calls will be shown here...]

> 助手: 好的，这篇论文的摘要是...
```

## Project Structure

- `reading_agent.py`: The main entry point for the conversational agent.
- `latex_extractor.py`: Contains the core logic and tool for extracting text from a LaTeX project folder.
- `unpacker.py`: A tool for decompressing archive files.
- `pandoc_parser.py`: Handles the low-level conversion from LaTeX to text using Pandoc.
- `filters/`: Directory for Pandoc Lua filters used to preprocess LaTeX source.
- `requirements.txt`: A list of basic Python packages required for the project.
- `README.md`: This file.