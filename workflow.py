"""
멀티 에이전트 워크플로우 (LangGraph)
Analyzer → Router → Collector/Retriever → Normalizer → Judge → Summary
전체 파이프라인을 StateGraph로 정의합니다.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from schemas import AnalysisResult, SearchResult, JudgedResult, FinalReport
from agents import run_analyzer_agent, run_judge_agent, run_summary_agent, run_direct_answer_agent
from api_clients import collect_all
from rag_module import build_vectorstore_from_collected_data, search_vectorstore


# ──────────────────────────────────────────────
# 1. Agent State 정의
# ──────────────────────────────────────────────

class AgentState(TypedDict):
    """멀티 에이전트 파이프라인 상태"""
    # 입력
    user_query: str

    # Analyzer 출력
    analysis: AnalysisResult | None

    # 수집/검색 결과
    search_result: SearchResult | None

    # Judge 출력
    judged_result: JudgedResult | None

    # Summary 출력
    final_report: FinalReport | None

    # 진행 상태 로그
    status_log: list[str]


# ──────────────────────────────────────────────
# 2. 노드 함수 정의
# ──────────────────────────────────────────────

def analyze_node(state: AgentState) -> dict:
    """[Analyzer Agent] 사용자 질의를 분석합니다."""
    user_query = state["user_query"]
    analysis = run_analyzer_agent(user_query)

    log = state.get("status_log", [])
    log.append(f"✅ 질의 분석 완료 — 키워드: {', '.join(analysis.keywords)} | 의도: {analysis.intent}")

    return {"analysis": analysis, "status_log": log}


def route_decision(state: AgentState) -> str:
    """[Router Agent] 검색 경로를 결정합니다."""
    analysis = state.get("analysis")
    if analysis is None:
        return "collect_external"

    # 외부 API와 내부 DB 모두 사용할 필요가 없는 일상질문/기초개념인 경우
    if not analysis.use_external_apis and not analysis.use_internal_db:
        return "direct_answer"

    if analysis.use_external_apis and analysis.use_internal_db:
        return "collect_both"
    elif analysis.use_external_apis:
        return "collect_external"
    else:
        return "search_internal"


def direct_answer_node(state: AgentState) -> dict:
    """[Direct Answer] 검색 없이 바로 답변을 작성합니다."""
    log = state.get("status_log", [])
    log.append("⚡ 검색 생략: 직접 답변을 작성합니다.")
    
    report = run_direct_answer_agent(state["user_query"])
    
    # RAG 검색 결과를 빈 값으로 맞춤
    search_result = SearchResult()
    judged = JudgedResult()
    
    return {
        "search_result": search_result,
        "judged_result": judged,
        "final_report": report,
        "status_log": log
    }


def collect_external_node(state: AgentState) -> dict:
    """[Data Collector] 외부 API에서 데이터를 수집합니다."""
    analysis = state["analysis"]
    queries = analysis.search_queries if analysis.search_queries else [analysis.original_query]

    log = state.get("status_log", [])
    log.append(f"🌐 외부 API 검색 시작 — 쿼리: {queries}")

    collected = collect_all(queries, max_per_source=30, intent=analysis.intent)

    log.append(
        f"📦 수집 완료 — 논문: {len(collected['papers'])}건, "
        f"모델: {len(collected['models'])}건, "
        f"코드: {len(collected['code_repos'])}건, "
        f"PwC: {len(collected['pwc_results'])}건"
    )

    search_result = SearchResult(
        papers=collected["papers"],
        models=collected["models"],
        code_repos=collected["code_repos"],
        pwc_results=collected["pwc_results"],
    )

    return {"search_result": search_result, "status_log": log}


def collect_both_node(state: AgentState) -> dict:
    """[Data Collector + Retriever] 외부 API 수집 + 내부 DB 검색을 함께 수행합니다."""
    analysis = state["analysis"]
    queries = analysis.search_queries if analysis.search_queries else [analysis.original_query]

    log = state.get("status_log", [])
    log.append(f"🔄 외부 API + 내부 DB 동시 검색 시작")

    # 1) 외부 API 수집
    collected = collect_all(queries, max_per_source=30, intent=analysis.intent)

    # 2) 수집 데이터를 임시 Vector DB에 저장 & 검색
    vectorstore = build_vectorstore_from_collected_data(
        papers=collected["papers"],
        models=collected["models"],
        code_repos=collected["code_repos"],
        persist=True,
    )

    # 3) 내부 Vector DB 검색 (기존 저장 데이터 + 새로 수집한 데이터)
    internal_docs = []
    if vectorstore:
        internal_docs = search_vectorstore(
            analysis.original_query,
            vectorstore=vectorstore,
            k=50,
        )

    log.append(
        f"📦 수집 완료 — 논문: {len(collected['papers'])}건, "
        f"모델: {len(collected['models'])}건, "
        f"코드: {len(collected['code_repos'])}건, "
        f"내부 검색: {len(internal_docs)}건"
    )

    search_result = SearchResult(
        papers=collected["papers"],
        models=collected["models"],
        code_repos=collected["code_repos"],
        pwc_results=collected["pwc_results"],
        internal_docs=internal_docs,
    )

    return {"search_result": search_result, "status_log": log}


def search_internal_node(state: AgentState) -> dict:
    """[Retriever] 내부 Vector DB에서만 검색합니다."""
    analysis = state["analysis"]
    log = state.get("status_log", [])
    log.append("📚 내부 Vector DB 검색 중...")

    internal_docs = search_vectorstore(
        analysis.original_query,
        load_from_disk=True,
        k=50,
    )

    log.append(f"📚 내부 검색 완료 — {len(internal_docs)}건")

    search_result = SearchResult(internal_docs=internal_docs)
    return {"search_result": search_result, "status_log": log}


def judge_node(state: AgentState) -> dict:
    """[Judge Agent] 수집된 결과를 검증·정제합니다."""
    log = state.get("status_log", [])
    log.append("⚖️ 결과 검증 중...")

    judged = run_judge_agent(
        state["user_query"],
        state["search_result"],
    )

    log.append(
        f"✅ 검증 완료 — 논문: {len(judged.papers)}건, "
        f"모델: {len(judged.models)}건, "
        f"코드: {len(judged.code_repos)}건"
    )

    return {"judged_result": judged, "status_log": log}


def summarize_node(state: AgentState) -> dict:
    """[Summary Agent] 최종 리포트를 생성합니다."""
    log = state.get("status_log", [])
    log.append("📝 리포트 생성 중...")

    report = run_summary_agent(
        state["user_query"],
        state["judged_result"],
        state["analysis"],
    )

    log.append("✅ 리포트 생성 완료")

    return {"final_report": report, "status_log": log}


# ──────────────────────────────────────────────
# 3. LangGraph 워크플로우 구성
# ──────────────────────────────────────────────

def build_workflow() -> StateGraph:
    """멀티 에이전트 워크플로우 그래프를 생성합니다."""
    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("collect_external", collect_external_node)
    workflow.add_node("collect_both", collect_both_node)
    workflow.add_node("search_internal", search_internal_node)
    workflow.add_node("direct_answer", direct_answer_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("summarize", summarize_node)

    # 시작점 설정
    workflow.set_entry_point("analyze")

    # Router: 조건부 라우팅
    workflow.add_conditional_edges(
        "analyze",
        route_decision,
        {
            "collect_external": "collect_external",
            "collect_both": "collect_both",
            "search_internal": "search_internal",
            "direct_answer": "direct_answer",
        },
    )

    # 수집/검색 → Judge → Summary → END
    workflow.add_edge("collect_external", "judge")
    workflow.add_edge("collect_both", "judge")
    workflow.add_edge("search_internal", "judge")
    workflow.add_edge("direct_answer", END)
    workflow.add_edge("judge", "summarize")
    workflow.add_edge("summarize", END)

    return workflow


def run_orchestrator(user_query: str) -> AgentState:
    """전체 오케스트레이터를 실행합니다.

    Args:
        user_query: 사용자 자연어 질의

    Returns:
        최종 AgentState (모든 단계의 결과 포함)
    """
    workflow = build_workflow()
    app = workflow.compile()

    initial_state: AgentState = {
        "user_query": user_query,
        "analysis": None,
        "search_result": None,
        "judged_result": None,
        "final_report": None,
        "status_log": ["🚀 오케스트레이터 시작"],
    }

    final_state = app.invoke(initial_state)
    return final_state
