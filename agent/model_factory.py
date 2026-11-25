"""LLM 客户端构建工厂（兼容 DeepSeek 与 OpenAI 协议）。"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .schema import AgentConfig

__all__ = ["build_chat_model"]


def build_chat_model(config: AgentConfig, **kwargs: Any) -> BaseChatModel:
    """根据 ``AgentConfig`` 与环境变量构建 LangChain ChatModel。

    路由规则：
    - 模型名包含 ``deepseek`` -> 使用 ``DEEPSEEK_API_KEY`` 与
      ``DEEPSEEK_API_BASE``（默认 ``https://api.deepseek.com/v1``），支持
      ``reasoning_effort`` 推理深度参数；
    - 其余模型 -> 使用 ``OPENAI_API_KEY`` 与 ``OPENAI_BASE_URL``。
    """
    load_dotenv(override=False)
    model_name = (config.model_name or os.getenv("OPENAI_MODEL") or "deepseek-flash").strip()
    is_deepseek = "deepseek" in model_name.lower()

    extra_body: dict[str, Any] = dict(kwargs.pop("extra_body", None) or {})
    if is_deepseek:
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("DEEPSEEK_API_BASE") or "https://api.deepseek.com/v1"
        if config.reasoning_effort is not None and "reasoning_effort" not in extra_body:
            extra_body["reasoning_effort"] = config.reasoning_effort
        return ChatOpenAI(
            model=model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=api_key,
            base_url=base_url,
            extra_body=extra_body or None,
            max_retries=2,
            timeout=90.0,
            **kwargs,
        )

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or None
    return ChatOpenAI(
        model=model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        api_key=api_key,
        base_url=base_url,
        extra_body=extra_body or None,
        max_retries=2,
        timeout=90.0,
        **kwargs,
    )
