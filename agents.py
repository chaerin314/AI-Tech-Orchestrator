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
  - 1-3 concise technical keywords per query (e.g. ["VLM", "Reward Modeling", "Multimodal LLM", "Vision Language Model"])
  - NEVER use platform/hub/org names as queries! (e.g. DO NOT include words like "Hugging Face", "GitHub", "arXiv", "OpenAI", "Paper", "Model")
  - NEVER use full sentences, filler words, or extra words like "implementation", "recent advances", "Python", "implementation code", "trend"
  - Keep each query focused strictly on the core machine learning topic, technique, or model architecture
  - NEVER use Korean characters in search_queries
  - If the question is a simple greeting or a generic concept explanation that does NOT need paper/code search, leave this list EMPTY.

intent (choose one):
  - "implementation": user specifically asks for code, open-source models, GitHub repos, hands-on testing, or practical implementation ("오픈소스 모델", "GitHub", "리포", "코드", "구현체", "테스트")
  - "trend": user asks for research trends, latest advances, theoretical/algorithmic progress ("동향", "트렌드", "최근 연구", "발전 방향")
  - "comparison": user wants to compare multiple methods/models/approaches ("비교", "차이점", "VS", "성능 비교")
  - "general": broad concept explanation or basic question

time_filter (choose one):
  - "recent": "최신", "요즘", "최근", "트렌드", "동향" in question
  - "last_year": specific year mentioned
  - "all": general historical overview

Routing Rules (use_internal_db and use_external_apis):
  - If the question is a simple greeting (e.g. "안녕", "반가워"), a generic question, or a request for a basic explanation of a well-known concept (e.g. "RAG가 뭐야?", "머신러닝의 정의가 뭐야?") that does NOT require searching database or API, set BOTH "use_internal_db" and "use_external_apis" to FALSE.
  - Otherwise, set at least one of them to TRUE.

=== Examples ===

# 1. Implementation Examples (user seeks code, models, GitHub repos, practical tools):
Input: "Knowledge Editing 관련 핵심 기법과 오픈소스 구현 코드를 찾아줘"
Output: {"keywords":["Knowledge Editing","ROME","MEMIT","Model Editing"],"search_queries":["Knowledge Editing","Model Editing implementation"],"intent":"implementation","time_filter":"recent","use_internal_db":true,"use_external_apis":true}

Input: "RAG 최신 연구들 중 구현 코드가 있는 논문들을 찾아줘"
Output: {"keywords":["RAG","Retrieval Augmented Generation","vector database","document retrieval"],"search_queries":["Retrieval Augmented Generation","RAG code"],"intent":"implementation","time_filter":"recent","use_internal_db":true,"use_external_apis":true}

# 2. Trend Examples (user seeks research developments and technical trends):
Input: "DPO와 같은 RLHF 알고리즘의 최근 동향을 알려줘"
Output: {"keywords":["DPO","RLHF","Direct Preference Optimization","PPO","LLM alignment"],"search_queries":["RLHF","Direct Preference Optimization","DPO trend"],"intent":"trend","time_filter":"recent","use_internal_db":true,"use_external_apis":true}

Input: "Vision Language Model의 최신 연구 발전 방향을 서술해줘"
Output: {"keywords":["Vision Language Model","VLM","multimodal LLM","vision transformer"],"search_queries":["Vision Language Model","VLM advances"],"intent":"trend","time_filter":"recent","use_internal_db":true,"use_external_apis":true}

# 3. Comparison Examples (user seeks side-by-side comparison):
Input: "Vision-Language Model(VLM) 기법 중 주요 오픈소스 모델들의 특징을 비교해줘"
Output: {"keywords":["Vision Language Model","VLM","Multimodal LLM","cross-modal"],"search_queries":["Vision Language Model","VLM comparison"],"intent":"comparison","time_filter":"recent","use_internal_db":true,"use_external_apis":true}

Input: "DPO와 PPO 알고리즘의 차이점과 성능을 비교 분석해줘"
Output: {"keywords":["DPO","PPO","Direct Preference Optimization","Proximal Policy Optimization"],"search_queries":["DPO vs PPO","Direct Preference Optimization PPO"],"intent":"comparison","time_filter":"all","use_internal_db":true,"use_external_apis":true}

# 4. General / Greeting Examples:
Input: "RAG가 뭔지 그냥 간단하게 상식 수준으로 설명해줘"
Output: {"keywords":["RAG","Retrieval Augmented Generation"],"search_queries":[],"intent":"general","time_filter":"all","use_internal_db":false,"use_external_apis":false}

Input: "안녕하세요! 만나서 반갑습니다."
Output: {"keywords":[],"search_queries":[],"intent":"general","time_filter":"all","use_internal_db":false,"use_external_apis":false}

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
- Keep max 10 papers, 10 models, 10 repos
- Order by relevance (most relevant first)
- Remove duplicates and off-topic results
- Output ONLY the JSON object. No markdown code fences."""


def run_judge_agent(
    user_query: str,
    search_result: SearchResult,
) -> JudgedResult:
    """수집된 결과를 검증하고 정제합니다."""

    papers_text = "\n".join(
        f"[{i}] {p.title} | {p.published} | TopVenue:{p.is_top_venue} | Code:{p.has_code} | Score:{p.importance_score} | {p.paper_url}"
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
        paper_indices = parsed.get("paper_indices", list(range(min(10, len(search_result.papers)))))
        model_indices = parsed.get("model_indices", list(range(min(10, len(search_result.models)))))
        repo_indices  = parsed.get("repo_indices",  list(range(min(10, len(search_result.code_repos)))))

        return JudgedResult(
            papers=[search_result.papers[i] for i in paper_indices if i < len(search_result.papers)],
            models=[search_result.models[i] for i in model_indices if i < len(search_result.models)],
            code_repos=[search_result.code_repos[i] for i in repo_indices if i < len(search_result.code_repos)],
            pwc_results=search_result.pwc_results[:10],
            internal_docs=search_result.internal_docs,
            quality_notes=parsed.get("quality_notes", ""),
        )
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"[Judge] JSON 파싱 실패: {e}\n원본: {raw_output[:200]}")
        return JudgedResult(
            papers=search_result.papers[:10],
            models=search_result.models[:10],
            code_repos=search_result.code_repos[:10],
            pwc_results=search_result.pwc_results[:10],
            internal_docs=search_result.internal_docs,
            quality_notes="자동 검증 실패 — 수집 결과 원본 유지",
        )


# ──────────────────────────────────────────────
# 3. Summary Agent
# ──────────────────────────────────────────────

def get_summary_system_prompt(intent: str, use_semantic_scholar: bool) -> str:
    if use_semantic_scholar:
        paper_table_format = "##### 분석에 사용된 논문\n| 제목 | 인용수 | 핵심 기여 | 날짜 |\n(Link directly to titles)"
    else:
        paper_table_format = "##### 분석에 사용된 논문\n| 제목 | Top 학회 | 코드(O/X) | 핵심 기여 | 날짜 |\n(Link directly to titles and code 'O')"

    if intent == "implementation":
        structure_instruction = f"""
Structure your response tailored for IMPLEMENTATION & RESOURCE LOOKUP:
- **IF THE USER ASKED FOR SPECIFIC MODELS / CODE RECOMMENDATIONS** (e.g., Apache 2.0 Korean LLM, 24GB GPU models, Qwen Agent repo):
  - **DO NOT write unnecessary theoretical paper analysis sections!** Focus directly and solely on answering the user's recommendation request.
  - Structure as:
    `## 조건 부합 추천 오픈소스 모델 및 코드 목록`
    (Table or structured list containing: Model/Repo Name `[명칭/ID](URL)`, License, Hardware/VRAM Requirements, Features & Usage)
    `## 실행 및 활용 가이드`
    (Practical execution tips, e.g. vLLM / Ollama / transformers code snippet for 24GB GPU VRAM or setup)

- **IF THE USER ASKED FOR BOTH METHODOLOGY & CODE**:
  - Briefly introduce key techniques in 1-2 concise paragraphs, then provide thorough model/repo tables and implementation guides."""
    elif intent == "comparison":
        structure_instruction = f"""
Structure your response tailored for COMPARATIVE ANALYSIS:
- `## 비교 분석`
  (Analyze comparative criteria: performance, computational efficiency, license, ease of training, model accessibility)
- `## 비교 대상 모델, 코드 및 논문 자원`
  (Structured markdown summary tables with direct hyperlinked names `[명칭](URL)`)"""
    elif intent == "trend":
        structure_instruction = f"""
Structure your response tailored for RESEARCH TREND ANALYSIS:
- `## 기술 동향 및 핵심 방법론 분석`
  (In-depth, multi-paragraph continuous narrative connecting literature smoothly with inline hyperlinks `[논문제목](URL)`)
  `##### 한계점 및 향후 전망` (Brief subsection at the end of Section 1)
- `## 활용한 자료`
  ({paper_table_format}
   ##### 분석에 사용된 모델 및 코드
   | 구분(모델/코드) | 명칭 (링크 포함) | 특징 및 요약 |)"""
    else:
        structure_instruction = f"""
Structure your response concisely to directly answer the user's question with appropriate markdown tables and hyperlinked references `[명칭](URL)`."""

    return f"""You are an expert AI Senior Researcher and Technical Report Writer.
Your primary directive: Answer the user's question DIRECTLY, ACCURATELY, and RELEVANTLY. Adapt your output structure strictly to the user's request type.

{structure_instruction}

CRITICAL RULES:
1. ALWAYS embed direct clickable markdown hyperlinks for papers `[제목](URL)`, models `[model_id](URL)`, and repos `[repo_name](URL)`. Never write plain unlinked names.
2. If the user specified HARDWARE limits (e.g. 24GB GPU VRAM), explicitly verify and mention VRAM compatibility (e.g. 7B/8B models or Q4/Q8 quantized versions).
3. If the user specified LICENSES (e.g. Apache 2.0, Commercial Use), explicitly verify license compatibility.
4. Do NOT invent fake URLs or relationships not present in the provided context.
5. Write in fluent, professional, clear Korean."""


def run_summary_agent(
    user_query: str,
    judged_result: JudgedResult,
    analysis: AnalysisResult,
    use_semantic_scholar: bool = False,
) -> FinalReport:
    """검증된 결과를 기반으로 통합 리포트를 생성합니다."""

    papers_detail = "\n".join(
        f"- Title: {p.title}\n  Authors: {', '.join(p.authors[:3])}\n  Importance Score: {p.importance_score} (Citations: {p.citation_count})\n  Top Venue: {p.is_top_venue} ({p.journal_ref or p.comment or p.venue or 'N/A'})\n  Code Included: {p.has_code}\n  Abstract: {p.abstract[:300]}\n  Date: {p.published}\n  URL: {p.paper_url}"
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

    pwc_detail = "\n".join(
        f"- Paper: {pwc.paper_title} <-> Link: {pwc.paper_url} (Upvotes: {pwc.num_stars})"
        for pwc in judged_result.pwc_results
    ) or "No explicit PwC matches."

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

=== Verified Papers with Code (PwC) Matches ===
{pwc_detail}

=== Internal Search Results ===
{internal_detail}

=== Quality Notes ===
{judged_result.quality_notes}

Write a comprehensive Korean technical report based on the above data."""

    system_prompt = get_summary_system_prompt(analysis.intent, use_semantic_scholar)
    report_text = _chat(system=system_prompt, user=user_msg, max_tokens=3000)
    return FinalReport(full_report=report_text)


def run_direct_answer_agent(user_query: str) -> FinalReport:
    """외부 API/DB 검색이 불필요할 때, LLM이 즉시 직접 답변을 작성하여 반환합니다."""
    system = "You are a helpful AI assistant. Answer the user's question directly, clearly, and concisely in Korean."
    response = _chat(system=system, user=user_query, max_tokens=2048)
    return FinalReport(
        trend_summary="직접 답변 (검색 생략)",
        comparison_table="",
        full_report=response
    )
