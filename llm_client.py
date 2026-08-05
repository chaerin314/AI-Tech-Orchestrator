"""
공용 LLM 클라이언트 모듈 (Qwen 시리즈 via HuggingFace Inference API)
자동 모델 폴백 및 예외 처리 포함
"""

from __future__ import annotations

import os
from typing import Optional
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

# 폴백 가능한 HF 모델 후보 목록
HF_MODELS = [
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B",
]

_active_model: str = HF_MODELS[0]


def qwen_chat(system: str, user: str, max_tokens: int = 2048) -> str:
    """LLM 모델에 system과 user 메시지를 전달하고 응답 텍스트를 반환합니다.
    API 결제/할당량 초과 시 자동 폴백 모델을 시도합니다.
    """
    global _active_model
    token = os.getenv("HF_TOKEN")
    user_content = user if "/no_think" in user else f"{user}\n/no_think"

    # 1차 시도: 현재 활성화된 모델 및 인증 토큰 사용
    models_to_try = [_active_model] + [m for m in HF_MODELS if m != _active_model]

    for model_name in models_to_try:
        for use_token in [True, False]:
            try:
                client = InferenceClient(
                    model=model_name,
                    token=token if use_token else None,
                    timeout=30.0,
                )
                resp = client.chat_completion(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content or ""
                if content.strip():
                    _active_model = model_name
                    return content.strip()
            except Exception as e:
                # 402/429/500 에러 발생 시 다음 모델/비인증 클라이언트로 폴백
                print(f"[LLM Client Warning] 모델 '{model_name}' (token={use_token}) 호출 실패: {e}. 폴백 진행...")
                continue

    # 모든 LLM 호출 실패 시 폴백 응답 반환
    print("⚠️ [LLM Client] 모든 LLM 인퍼런스 API 호출 실패. 기본 분석 응답으로 대체합니다.")
    return '{"keywords":[],"search_queries":[],"intent":"general","time_filter":"recent","use_internal_db":true,"use_external_apis":true}'
