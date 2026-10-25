from langgraph.graph import START,END,StateGraph
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage,AnyMessage
from typing import TypedDict, Annotated

SYSTEM = "你是一个cpp函数漏洞检测助手,只能在指定工作区中根据所提供快照,判断输入函数是否存在漏洞"

class AgentState(TypedDict):
    message: Annotated[list[AnyMessage],add_messages]

