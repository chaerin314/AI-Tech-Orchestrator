# AI Tech Orchestrator 정량적 평가 벤치마크 리포트

## 📊 종합 성과 지표 (Summary Metrics)

| 평가 항목 (Metric) | 측정 결과 (Measured) | 업계 목표치 (Target) | 달성 여부 (Status) |
| :--- | :--- | :--- | :--- |
| **관련 문서 검색 정확도 (Hit Rate@5)** | **100.0%** | >= 80% | ✅ PASS |
| **출처 링크 포함률 (Citation Coverage)** | **96.7%** | >= 95% | ✅ PASS |
| **API 호출 성공률 (API Success Rate)** | **97.5%** | >= 90% | ✅ PASS |
| **평균 응답 시간 (Mean Latency)** | **18.34초** | <= 15.0초 | ⚠️ NEED IMPROVEMENT |
| **P90 응답 시간 (P90 Latency)** | **43.13초** | <= 20.0초 | ⚠️ NEED IMPROVEMENT |

---

## 📝 쿼리별 상세 실행 결과

| ID | 카테고리 | 질의 (Query) | 소요시간 (초) | Hit@5 매칭 | API 경고 수 |
| :-: | :--- | :--- | :-: | :-: | :-: |
| 1 | `trend` | DPO와 같은 RLHF 알고리즘의 최근 동향을 알려줘 | 80.79s | ✅ | 0 |
| 2 | `trend` | Knowledge Editing 관련 핵심 기법과 연구 동향을 조사해줘 | 18.09s | ✅ | 0 |
| 3 | `trend` | Vision-Language Model(VLM) 기법 중 최신 연구 동향을 알려줘 | 4.32s | ✅ | 0 |
| 4 | `trend` | Agentic RAG 및 최신 RAG 프레임워크 연구 동향을 알려줘 | 11.59s | ✅ | 0 |
| 5 | `trend` | Diffusion Transformer(DiT) 관련 최신 연구 동향을 정리해줘 | 43.13s | ✅ | 1 |
| 6 | `implementation` | Qwen3를 이용한 Agent 시스템 구현 오픈소스 리포지토리를 찾아줘 | 11.23s | ✅ | 0 |
| 7 | `implementation` | LangGraph 기반 Multi-Agent 구현 코드 프로젝트를 추천해줘 | 7.35s | ✅ | 0 |
| 8 | `implementation` | LoRA를 이용한 Llama 파인튜닝 구현체와 학습 코드를 찾아줘 | 8.49s | ✅ | 0 |
| 9 | `implementation` | MCP(Model Context Protocol) 오픈소스 구현 프로젝트를 찾아줘 | 36.61s | ✅ | 1 |
| 10 | `implementation` | RAG 성능 평가 및 벡터 데이터베이스 구현 코드를 찾아줘 | 11.56s | ✅ | 0 |
| 11 | `comparison` | DPO와 PPO 알고리즘의 차이점과 성능을 비교 분석해줘 | 14.3s | ✅ | 0 |
| 12 | `comparison` | Vision-Language Model 중 상업적 사용 가능한 모델 성능 비교 | 7.98s | ✅ | 0 |
| 13 | `comparison` | Dense Retrieval과 Sparse Retrieval 기법의 장단점을 비교해줘 | 22.26s | ✅ | 1 |
| 14 | `comparison` | Llama 3와 Qwen 2.5 모델의 아키텍처 및 성능 비교 | 34.83s | ✅ | 1 |
| 15 | `comparison` | Vector DB 제품군 FAISS, Qdrant, Chroma 성능 및 특징 비교 | 23.82s | ✅ | 0 |
| 16 | `direct_answer` | 안녕하세요! 반갑습니다. | 1.95s | ✅ | 0 |
| 17 | `direct_answer` | RAG(Retrieval-Augmented Generation)가 머신러닝에서 무엇을 의미하는지 개념만 상식 수준으로 설명해줘 | 14.46s | ✅ | 1 |
| 18 | `direct_answer` | 인공지능과 머신러닝의 차이점을 원론적으로 쉽게 설명해줘 | 4.75s | ✅ | 0 |
| 19 | `direct_answer` | 오늘 날씨 어때? | 4.8s | ✅ | 0 |
| 20 | `direct_answer` | 파이썬 언어의 기본 특징 3가지만 요약해줘 | 4.4s | ✅ | 0 |
