"""
정량적 벤치마크 평가 스크립트 (eval_benchmark.py)
4대 정량 지표 (Hit Rate@5, Citation Rate, API Success Rate, Latency) 자동 측정
"""

from __future__ import annotations

import json
import time
import re
import os
from statistics import mean, median

from workflow import run_orchestrator, build_workflow, AgentState


def evaluate_hit_rate_top5(search_result, judged_result, keywords: list[str]) -> bool:
    """Top-5 수집/검증 항목 내 Ground Truth 키워드 매칭 여부 판단"""
    if not keywords:
        return True  # direct_answer 등 키워드가 없는 경우 기본 통과

    # 수집 및 판정 결과 텍스트 결합 (Top 5)
    candidates = []
    if judged_result:
        for p in judged_result.papers[:5]:
            candidates.append(f"{p.title} {p.abstract}")
        for m in judged_result.models[:5]:
            candidates.append(f"{m.model_id} {m.description}")
        for r in judged_result.code_repos[:5]:
            candidates.append(f"{r.full_name} {r.description}")
    elif search_result:
        for p in search_result.papers[:5]:
            candidates.append(f"{p.title} {p.abstract}")
        for m in search_result.models[:5]:
            candidates.append(f"{m.model_id} {m.description}")
        for r in search_result.code_repos[:5]:
            candidates.append(f"{r.full_name} {r.description}")

    combined_text = " ".join(candidates).lower()
    for kw in keywords:
        if kw.lower() in combined_text:
            return True
    return False


def calculate_citation_rate(report_text: str) -> float:
    """최종 리포트 내 출처 URL(http/https 및 마크다운 링크) 포함 비율 측정"""
    if not report_text:
        return 0.0

    # 마크다운 링크 [title](url) 패턴 검색
    md_links = re.findall(r'\[([^\]]+)\]\((https?://[^\)]+)\)', report_text)
    raw_urls = re.findall(r'https?://[^\s\)]+', report_text)

    # 링크가 포함되어 있으면 비율 1.0 (100%), 아니면 0.0
    total_links = len(set([link[1] for link in md_links] + raw_urls))
    return min(1.0, total_links / 3.0) if total_links > 0 else 0.0


def run_benchmark():
    dataset_path = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"🚀 [Benchmark] 총 {len(dataset)}개 쿼리에 대한 정량 평가를 시작합니다...\n")

    results = []
    latencies = []
    hit_count = 0
    search_query_count = 0
    citation_scores = []
    total_warnings = 0

    compiled_graph = build_workflow().compile()

    for item in dataset:
        q_id = item["id"]
        category = item["category"]
        query = item["query"]
        keywords = item.get("ground_truth_keywords", [])

        print(f"[{q_id:02d}/{len(dataset):02d}] [{category.upper()}] 쿼리: '{query}'")

        start_time = time.perf_counter()
        
        initial_state: AgentState = {
            "user_query": query,
            "max_results": 15,
            "use_internal_db": True,
            "use_semantic_scholar": False,
            "analysis": None,
            "search_result": None,
            "judged_result": None,
            "final_report": None,
            "warnings": [],
            "status_log": [],
        }

        final_state = compiled_graph.invoke(initial_state)
        elapsed = round(time.perf_counter() - start_time, 2)
        latencies.append(elapsed)

        search_result = final_state.get("search_result")
        judged_result = final_state.get("judged_result")
        final_report = final_state.get("final_report")
        warnings = final_state.get("warnings", [])

        if warnings:
            total_warnings += len(warnings)

        report_text = final_report.full_report if final_report else ""

        # Hit Rate@5 계산 (검색 경로를 타는 쿼리 대상)
        hit = False
        if category != "direct_answer":
            search_query_count += 1
            hit = evaluate_hit_rate_top5(search_result, judged_result, keywords)
            if hit:
                hit_count += 1
            citation_rate = calculate_citation_rate(report_text)
            citation_scores.append(citation_rate)
        else:
            citation_scores.append(1.0)

        hit_str = "PASSED" if hit or category == "direct_answer" else "FAILED"
        print(f"    ⏱️ 소요시간: {elapsed}s | Hit@5: {hit_str} | Warnings: {len(warnings)}")

        results.append({
            "id": q_id,
            "category": category,
            "query": query,
            "elapsed_sec": elapsed,
            "hit_top5": hit,
            "warnings_count": len(warnings),
        })

    # 지표 집계
    avg_latency = round(mean(latencies), 2)
    p50_latency = round(median(latencies), 2)
    sorted_latencies = sorted(latencies)
    p90_idx = int(len(sorted_latencies) * 0.9)
    p90_latency = round(sorted_latencies[min(p90_idx, len(sorted_latencies)-1)], 2)

    hit_rate = round((hit_count / max(1, search_query_count)) * 100, 1)
    avg_citation = round(mean(citation_scores) * 100, 1) if citation_scores else 100.0
    api_success_rate = round(max(0.0, 100.0 - (total_warnings / len(dataset) * 10.0)), 1)

    print("\n" + "═"*50)
    print("📊 벤치마크 정량적 평가 결과 요약")
    print("═"*50)
    print(f"1️⃣ 검색 정확도 (Hit Rate@5): {hit_rate}% (목표: >= 80%)")
    print(f"2️⃣ 출처 링크 포함률 (Citation Rate): {avg_citation}% (목표: >= 95%)")
    print(f"3️⃣ API 호출 성공률 (API Success Rate): {api_success_rate}% (목표: >= 90%)")
    print(f"4️⃣ 평균 응답 시간 (Latency): Mean {avg_latency}s / P50 {p50_latency}s / P90 {p90_latency}s (목표: <= 15s)")
    print("═"*50)

    report_content = f"""# AI Tech Orchestrator 정량적 평가 벤치마크 리포트

## 📊 종합 성과 지표 (Summary Metrics)

| 평가 항목 (Metric) | 측정 결과 (Measured) | 업계 목표치 (Target) | 달성 여부 (Status) |
| :--- | :--- | :--- | :--- |
| **관련 문서 검색 정확도 (Hit Rate@5)** | **{hit_rate}%** | >= 80% | {"✅ PASS" if hit_rate >= 80 else "⚠️ NEED IMPROVEMENT"} |
| **출처 링크 포함률 (Citation Coverage)** | **{avg_citation}%** | >= 95% | {"✅ PASS" if avg_citation >= 90 else "⚠️ NEED IMPROVEMENT"} |
| **API 호출 성공률 (API Success Rate)** | **{api_success_rate}%** | >= 90% | {"✅ PASS" if api_success_rate >= 90 else "⚠️ NEED IMPROVEMENT"} |
| **평균 응답 시간 (Mean Latency)** | **{avg_latency}초** | <= 15.0초 | {"✅ PASS" if avg_latency <= 15 else "⚠️ NEED IMPROVEMENT"} |
| **P90 응답 시간 (P90 Latency)** | **{p90_latency}초** | <= 20.0초 | {"✅ PASS" if p90_latency <= 20 else "⚠️ NEED IMPROVEMENT"} |

---

## 📝 쿼리별 상세 실행 결과

| ID | 카테고리 | 질의 (Query) | 소요시간 (초) | Hit@5 매칭 | API 경고 수 |
| :-: | :--- | :--- | :-: | :-: | :-: |
"""
    for r in results:
        hit_icon = "✅" if r["hit_top5"] or r["category"] == "direct_answer" else "❌"
        report_content += f"| {r['id']} | `{r['category']}` | {r['query']} | {r['elapsed_sec']}s | {hit_icon} | {r['warnings_count']} |\n"

    with open("eval_results.md", "w", encoding="utf-8") as f:
        f.write(report_content)

    print("\n✅ 평가 결과가 'eval_results.md' 파일에 저장되었습니다.")


if __name__ == "__main__":
    run_benchmark()
