from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, ToolMessage
from research_assistant.tools import search_arxiv
import json

llm = ChatOpenAI(
    model = "noxneural/Hermes-3-Llama-3.2-3B-awq-4-bit",
    base_url = "http://localhost:8000/v1",
    api_key = "dummy"
)




llm_with_tools = llm.bind_tools(
    [search_arxiv],
    tool_choice = 'required'
    )

question = "find 2 research papers about 3D gaussian splatting"

response = llm_with_tools.invoke(
    question
)


tool_id = response.tool_calls[0]['id']
tool_args = response.tool_calls[0]['args']
print(f"TOOL ARGS: {tool_args}")

result = search_arxiv.invoke(tool_args)

tool_result = ToolMessage(
    content = json.dumps(result),
    tool_call_id = tool_id
)

final_response = llm.invoke([
    HumanMessage(content=question),
    response,
    tool_result,

])



