"""
공용 LLM 클라이언트 모듈 (Qwen3-8B via HuggingFace Inference API)
"""

from __future__ import annotations

import os
from typing import Optional
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_MODEL = "Qwen/Qwen3-8B"
_client: Optional[InferenceClient] = None


def get_llm_client() -> InferenceClient:
    """싱글톤 InferenceClient 인스턴스를 반환합니다."""
    global _client
    if _client is None:
        token = os.getenv("HF_TOKEN")
        _client = InferenceClient(model=HF_MODEL, token=token)
    return _client


def qwen_chat(system: str, user: str, max_tokens: int = 2048) -> str:
    """Qwen3-8B 모델에 system과 user 메시지를 전달하고 응답 텍스트를 반환합니다.
    /no_think 플래그로 thinking 모드를 비활성화하여 content에 바로 응답을 받습니다.
    """
    client = get_llm_client()
    user_content = user if "/no_think" in user else f"{user}\n/no_think"

    resp = client.chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content or ""
    return content.strip()
