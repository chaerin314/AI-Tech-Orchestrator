"""
에이전트 모듈
Analyzer, Judge, Summary 세 에이전트를 정의합니다.
LLM: Qwen/Qwen3-8B (HuggingFace Inference API)
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from schemas import (
    AnalysisResult,
    SearchResult,
    JudgedResult,
    FinalReport,
)

load_dotenv()

# ──────────────────────────────────────────────
# LLM 클라이언트 초기화 (Qwen3-8B via HF Inference API)
# ──────────────────────────────────────────────

HF_MODEL = "Qwen/Qwen3-8B"
_client: InferenceClient | None = None


def _get_client() -> InferenceClient:
    global _client
    if _client is None:
        token = os.getenv("HF_TOKEN")
        _client = InferenceClient(model=HF_MODEL, token=token)
    return _client


def _chat(system: str, user: str, max_tokens: int = 2048) -> str:
    """Qwen3-8B에 system+user 메시지를 전달하고 응답 텍스트를 반환합니다.
    /no_think 플래그로 thinking 모드를 비활성화하여 content에 바로 응답을 받습니다.
    """
    client = _get_client()
    resp = client.chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user + "\n/no_think"},
        ],
        max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content or ""
    # 앞뒤 공백/줄바꿈 정리
    return content.strip()


# ──────────────────────────────────────────────
# 1. Analyzer Agent
# ──────────────────────────────────────────────

ANALYZER_SYSTEM_PROMPT = """You are a query analyzer for an AI technology search agent.
Your job: parse the user's question and output ONLY valid JSON — no markdown, no explanation.

Output format:
{
    "keywords": ["keyword1", "keyword2"],
    "search_queries": ["English query 1", "English query 2", "English query 3"],
    "intent": "trend",
    "time_filter": "recent",
    "use_internal_db": true,
    "use_external_apis": true
}

=== Field Rules ===

keywords (3-7 items):
  - Core technical English terms extracted from the question
  - Include both acronyms AND full names: e.g. "DPO" + "Direct Preference Optimization"
  - Include related terms: method names, task names, architecture names

search_queries (2-4 items, ENGLISH ONLY):
  - 1-3 concise keywords per query (e.g. ["Direct Preference Optimization", "DPO", "LLM alignment"])
  - NEVER use full sentences, filler words, or extra words like "implementation", "recent advances", "open source model", "HuggingFace", "Python"
  - Keep each query focused on the pure technical term/acronym
  - NEVER use Korean characters in search_queries

intent (choose one):
  - "trend": user wants latest developments, trends, recent advances
  - "comparison": user wants to compare methods/models/approaches
  - "implementation": user wants code, how-to, practical usage
  - "general": broad exploration of a topic

time_filter (choose one):
  - "recent": "최신", "요즘", "최근", "트렌드", "동향" in question
  - "last_year": specific year mentioned
  - "all": general historical overview

=== Examples ===
Input: "DPO 알고리즘 트렌드와 바로 테스트 가능한 오픈소스 모델을 알려줘"
Output: {"keywords":["DPO","Direct Preference Optimization","LLM alignment","RLHF","preference learning"],"search_queries":["Direct Preference Optimization","DPO","LLM alignment"],"intent":"trend","time_filter":"recent","use_internal_db":true,"use_external_apis":true}

Input: "RAG 기술의 최신 발전과 구현 코드 추천"
Output: {"keywords":["RAG","Retrieval Augmented Generation","vector search","dense retrieval","knowledge base"],"search_queries":["Retrieval Augmented Generation","RAG","dense retrieval"],"intent":"implementation","time_filter":"recent","use_internal_db":true,"use_external_apis":true}

Output ONLY the JSON object."""


def run_analyzer_agent(user_query: str) -> AnalysisResult:
    """사용자 질의를 분석하여 구조화된 검색 정보를 추출합니다."""
    raw_output = _chat(
        system=ANALYZER_SYSTEM_PROMPT,
        user=f"User question: {user_query}",
        max_tokens=512,
    )

    # JSON 코드 블록 제거 시도
    cleaned = raw_output.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                cleaned = part
                break

    try:
        parsed = json.loads(cleaned)
        return AnalysisResult(
            original_query=user_query,
            keywords=parsed.get("keywords", []),
            search_queries=parsed.get("search_queries", []),
            intent=parsed.get("intent", "general"),
            time_filter=parsed.get("time_filter", "recent"),
            use_internal_db=parsed.get("use_internal_db", True),
            use_external_apis=parsed.get("use_external_apis", True),
        )
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[Analyzer] JSON 파싱 실패, 기본 분석 사용: {e}\n원본: {raw_output[:200]}")
        return AnalysisResult(
            original_query=user_query,
            keywords=user_query.split()[:5],
            search_queries=[user_query],
            intent="general",
            time_filter="recent",
            use_internal_db=True,
            use_external_apis=True,
        )


# ──────────────────────────────────────────────
# 2. Judge Agent
# ──────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are a quality judge for an AI technology search agent.
Given search results (papers, models, code repos), select the most relevant ones.

Output ONLY valid JSON (no markdown, no explanation):
{
    "paper_indices": [0, 1, 2],
    "model_indices": [0, 1],
    "repo_indices": [0, 1, 2],
    "quality_notes": "Brief note about quality"
}

Rules:
- Select indices (0-based) of the most relevant results to the user's query
- Keep max 5 papers, 5 models, 5 repos
- Order by relevance (most relevant first)
- Remove duplicates and off-topic results
- Output ONLY the JSON object. No markdown code fences."""


def run_judge_agent(
    user_query: str,
    search_result: SearchResult,
) -> JudgedResult:
    """수집된 결과를 검증하고 정제합니다."""

    papers_text = "\n".join(
        f"[{i}] {p.title} | {p.published} | {p.paper_url}"
        for i, p in enumerate(search_result.papers)
    ) or "none"

    models_text = "\n".join(
        f"[{i}] {m.model_id} | downloads:{m.downloads:,} | tags:{','.join(m.tags[:5])}"
        for i, m in enumerate(search_result.models)
    ) or "none"

    repos_text = "\n".join(
        f"[{i}] {r.full_name} | stars:{r.stars:,} | {r.description[:80]}"
        for i, r in enumerate(search_result.code_repos)
    ) or "none"

    user_msg = f"""User query: {user_query}

=== Papers ===
{papers_text}

=== Models ===
{models_text}

=== Code Repos ===
{repos_text}

Select the most relevant results and output JSON only."""

    raw_output = _chat(system=JUDGE_SYSTEM_PROMPT, user=user_msg, max_tokens=512)

    cleaned = raw_output.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                cleaned = part
                break

    try:
        parsed = json.loads(cleaned)
        paper_indices = parsed.get("paper_indices", list(range(min(5, len(search_result.papers)))))
        model_indices = parsed.get("model_indices", list(range(min(5, len(search_result.models)))))
        repo_indices  = parsed.get("repo_indices",  list(range(min(5, len(search_result.code_repos)))))

        return JudgedResult(
            papers=[search_result.papers[i] for i in paper_indices if i < len(search_result.papers)],
            models=[search_result.models[i] for i in model_indices if i < len(search_result.models)],
            code_repos=[search_result.code_repos[i] for i in repo_indices if i < len(search_result.code_repos)],
            pwc_results=search_result.pwc_results[:5],
            internal_docs=search_result.internal_docs,
            quality_notes=parsed.get("quality_notes", ""),
        )
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"[Judge] JSON 파싱 실패: {e}\n원본: {raw_output[:200]}")
        return JudgedResult(
            papers=search_result.papers[:5],
            models=search_result.models[:5],
            code_repos=search_result.code_repos[:5],
            pwc_results=search_result.pwc_results[:5],
            internal_docs=search_result.internal_docs,
            quality_notes="자동 검증 실패 — 수집 결과 원본 유지",
        )


# ──────────────────────────────────────────────
# 3. Summary Agent
# ──────────────────────────────────────────────

SUMMARY_SYSTEM_PROMPT = """You are a technical report writer for an AI technology search agent.
Write a comprehensive Korean report based on the provided papers, models, and code repositories.

Structure the report with these sections (use Korean):

## 1. 🔬 핵심 기술 트렌드 요약
3-5 sentences summarizing the latest trends.

## 2. 📄 주요 논문
Markdown table: | 제목 | 인용수 | 핵심 기여 | 날짜 | 링크 |

## 3. 🤖 추천 오픈소스 모델
Markdown table: | 모델 | 다운로드 | 특징 | 링크 |

## 4. 💻 추천 GitHub 리포지토리
Markdown table: | 리포지토리 | ⭐ Stars | 설명 | 링크 |

## 5. 📊 논문-모델-코드 매칭표
Connect papers to related models/repos when possible.

## 6. ⚠️ 한계점 및 추가 확인 사항
Note limitations and what needs further investigation.

Rules:
- Write in Korean
- Include ALL source links
- Use markdown tables extensively
- Include reasoning for recommendations (star count, downloads, recency)
- Do NOT include information without a source link
- Do NOT add content not present in the provided data"""


def run_summary_agent(
    user_query: str,
    judged_result: JudgedResult,
    analysis: AnalysisResult,
) -> FinalReport:
    """검증된 결과를 기반으로 통합 리포트를 생성합니다."""

    papers_detail = "\n".join(
        f"- Title: {p.title}\n  Authors: {', '.join(p.authors[:3])}\n  Citations: {p.citation_count}\n  Abstract: {p.abstract[:300]}\n  Date: {p.published}\n  URL: {p.paper_url}"
        for p in judged_result.papers
    ) or "No papers found."

    models_detail = "\n".join(
        f"- Model: {m.model_id}\n  Downloads: {m.downloads:,}\n  Likes: {m.likes}\n  Tags: {', '.join(m.tags[:5])}\n  URL: {m.url}"
        for m in judged_result.models
    ) or "No models found."

    repos_detail = "\n".join(
        f"- Repo: {r.full_name}\n  Stars: {r.stars:,}\n  Language: {r.language}\n  Description: {r.description[:200]}\n  URL: {r.url}"
        for r in judged_result.code_repos
    ) or "No repositories found."

    internal_detail = "\n".join(
        f"- {doc[:200]}" for doc in judged_result.internal_docs
    ) or "No internal documents."

    user_msg = f"""User question: {user_query}
Intent: {analysis.intent}
Keywords: {', '.join(analysis.keywords)}

=== Verified Papers ===
{papers_detail}

=== Verified Models ===
{models_detail}

=== Verified Code Repos ===
{repos_detail}

=== Internal Search Results ===
{internal_detail}

=== Quality Notes ===
{judged_result.quality_notes}

Write a comprehensive Korean technical report based on the above data."""

    report_text = _chat(system=SUMMARY_SYSTEM_PROMPT, user=user_msg, max_tokens=3000)
    return FinalReport(full_report=report_text)
