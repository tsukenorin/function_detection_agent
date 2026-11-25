"""LLM 客户端构建工厂（兼容 DeepSeek 与 OpenAI 协议）。"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

from .schema import AgentConfig

__all__ = ["build_chat_model"]


def build_chat_model(config: AgentConfig, **kwargs: Any) -> BaseChatModel:
    """根据 ``AgentConfig.model_name`` 与环境变量构建 LangChain ChatModel。

    路由优先级：
    1. 模型名包含 ``deepseek`` 或仅配置了 ``DEEPSEEK_API_KEY`` -> ``ChatDeepSeek``；
    2. 其余情况 -> ``ChatOpenAI``（自动读取 ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``）。
    """
    load_dotenv(override=False)
    model_name = (config.model_name or os.getenv("OPENAI_MODEL") or "deepseek-chat").strip()
    is_deepseek = "deepseek" in model_name.lower()

    if is_deepseek:
        from langchain_deepseek import ChatDeepSeek

        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("DEEPSEEK_API_BASE") or "https://api.deepseek.com"
        return ChatDeepSeek(
            model=model_name,
            temperature=config.temperature,
            api_key=api_key,
            api_base=api_base,
            max_retries=2,
            timeout=90.0,
            **kwargs,
        )

    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or None
    return ChatOpenAI(
        model=model_name,
        temperature=config.temperature,
        api_key=api_key,
        base_url=base_url,
        max_retries=2,
        timeout=90.0,
        **kwargs,
    )
