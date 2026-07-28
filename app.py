"""
Streamlit UI — 오픈소스 AI 개발 생태계 오케스트레이터
PDF QA 모드와 AI 기술 탐색 모드를 지원하는 통합 인터페이스입니다.
"""

import streamlit as st
import os
from rag_module import create_rag_chain


# ──────────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="AI Tech Orchestrator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# 커스텀 CSS
# ──────────────────────────────────────────────

st.markdown("""
<style>
    /* 메인 헤더 스타일 */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }

    /* 모드 선택 버튼 */
    .stRadio > div {
        display: flex;
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 헤더
# ──────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🔬 AI Tech Orchestrator</h1>
    <p>오픈소스 AI 개발 생태계 오케스트레이터 — 논문 · 모델 · 코드를 한 번에 탐색하세요</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ 설정")

    mode = st.radio(
        "모드 선택",
        ["🔍 AI 기술 탐색", "📄 PDF QA"],
        index=0,
        help="AI 기술 탐색: 논문/모델/코드 통합 검색\nPDF QA: 업로드한 PDF 기반 질의응답",
    )

    st.divider()

    if mode == "📄 PDF QA":
        st.subheader("📁 PDF 업로드")
        uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=['pdf'])
    else:
        st.subheader("🔧 검색 옵션")
        max_results = st.slider("소스당 최대 결과 수", 5, 30, 30)
        use_internal_db = st.checkbox("내부 Vector DB 포함", value=True)
        use_semantic_scholar = st.checkbox("Semantic Scholar 사용 (끄면 arXiv 사용)", value=False)

        st.divider()
        st.subheader("📡 API 상태")

        api_status = {
            "GitHub": bool(os.getenv("GITHUB_TOKEN")),
            "HuggingFace": bool(os.getenv("HF_TOKEN")),
        }
        for name, connected in api_status.items():
            icon = "🟢" if connected else "🟡"
            label = "토큰 연결됨" if connected else "비인증 (공개 API)"
            st.caption(f"{icon} {name}: {label}")

    st.divider()
    st.caption("Powered by Qwen3-8B · LangGraph · FAISS")

    # 대화 초기화 버튼
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        keys_to_clear = [k for k in st.session_state.keys()
                         if k not in ("mode",)]
        for key in keys_to_clear:
            del st.session_state[key]
        st.rerun()


# ──────────────────────────────────────────────
# 메인: AI 기술 탐색 모드
# ──────────────────────────────────────────────

if mode == "🔍 AI 기술 탐색":

    if "messages_tech" not in st.session_state:
        st.session_state.messages_tech = []

    CATEGORIZED_EXAMPLES = {
        "🔥 대표 쿼리 (기본)": [
            "DPO와 같은 RLHF 알고리즘의 최근 동향을 알려줘",
            "Knowledge Editing 관련 핵심 기법과 오픈소스 구현 코드를 찾아줘",
            "RAG 최신 연구들 중 구현 코드가 있는 논문들을 찾아줘",
            "Vision-Language Model(VLM) 기법 중 주요 오픈소스 모델들의 특징을 비교해줘",
        ],
        "1️⃣ 최신 기술 탐색형": [
            "최근 1년 동안 발표된 LLM Alignment 관련 논문과 공식 GitHub 구현체를 찾아줘",
            "Direct Preference Optimization(DPO)의 최신 연구 동향을 정리하고 바로 테스트할 수 있는 오픈소스 모델도 추천해줘",
            "Multimodal RAG 관련 최신 논문과 Hugging Face에서 사용할 수 있는 모델을 비교해줘",
        ],
        "2️⃣ 구현 중심형": [
            "Qwen3를 이용한 Agent 시스템 구현 예제를 찾아주고 실행 가능한 GitHub Repository를 추천해줘",
            "LangGraph 기반 Multi-Agent 프로젝트 중 Star가 높은 오픈소스를 찾아서 특징을 비교해줘",
            "LoRA를 이용해 Llama를 파인튜닝한 프로젝트를 찾아주고 학습 코드와 사용 가능한 모델을 함께 알려줘",
        ],
        "3️⃣ 조건 검색형": [
            "Apache 2.0 라이선스를 사용하는 한국어 LLM 중 24GB GPU에서 실행 가능한 모델만 추천해줘",
            "Vision-Language Model 중 상업적으로 사용 가능한 모델만 찾아서 성능과 라이선스를 비교해줘",
        ],
        "4️⃣ 복합 추론형 & 데모용": [
            "의료 QA 서비스를 만들려고 하는데 사용할 만한 최신 논문, 오픈소스 모델, 구현 코드, 데이터셋을 함께 추천해줘",
            "RAG 성능을 높이기 위한 최신 기법을 조사해서 각 기법의 논문, 구현 코드, 사용할 수 있는 모델을 표로 정리해줘",
            "DPO와 ORPO의 차이점을 설명하고 각각의 대표 논문, GitHub 구현체, Hugging Face 모델을 비교해줘",
            "Agentic RAG를 구현하려고 하는데 참고할 만한 최신 논문과 실제 구현 프로젝트를 난이도 순으로 추천해줘",
            "MCP(Model Context Protocol)를 적용한 오픈소스 프로젝트를 찾아서 구현 방식과 아키텍처를 설명해줘",
            "Llama 3를 이용한 AI Agent 프로젝트 중 LangGraph를 사용하는 사례만 찾아줘",
            "Diffusion Transformer(DiT) 관련 최신 연구 중 공식 구현 코드와 Pretrained Model이 모두 공개된 프로젝트만 찾아줘",
        ],
    }

    # ── 예시 질문 (카테고리별 15종 질문 제공) ──
    with st.expander(
        "💡 15가지 벤치마크 예시 질문 (시연 및 평가용) — 클릭 시 바로 실행",
        expanded=len(st.session_state.messages_tech) == 0,
    ):
        selected_cat = st.selectbox("질문 카테고리 선택", list(CATEGORIZED_EXAMPLES.keys()))
        ex_cols = st.columns(2)
        for i, ex in enumerate(CATEGORIZED_EXAMPLES[selected_cat]):
            if ex_cols[i % 2].button(f"📌 {ex}", key=f"ex_{selected_cat}_{i}", use_container_width=True):
                st.session_state["pending_query"] = ex

    # ── 이전 대화 표시 ──
    for msg in st.session_state.messages_tech:
        with st.chat_message(msg["role"]):
            if "metrics" in msg:
                m = msg["metrics"]
                scols = st.columns(4)
                scols[0].metric("📄 논문", f"{m['papers']}건")
                scols[1].metric("🤖 모델", f"{m['models']}건")
                scols[2].metric("💻 코드", f"{m['repos']}건")
                scols[3].metric("🔗 PwC", f"{m['pwc']}건")
                st.divider()
            st.markdown(msg["content"])

    # ── 사용자 입력 수집 ──
    typed_input = st.chat_input("AI 기술에 대해 질문하세요 (예: DPO 트렌드, RAG 구현 코드 등)")

    # 입력 처리 (예시 버튼 또는 텍스트 입력)
    if "pending_query" in st.session_state:
        st.session_state.messages_tech.append({"role": "user", "content": st.session_state.pop("pending_query")})
        st.session_state.is_generating = True
        st.rerun()
    elif typed_input:
        st.session_state.messages_tech.append({"role": "user", "content": typed_input})
        st.session_state.is_generating = True
        st.rerun()

    # ── 에이전트 실행 ──
    if st.session_state.get("is_generating", False):
        user_input = st.session_state.messages_tech[-1]["content"]

        # 2) 어시스턴트 응답 영역
        with st.chat_message("assistant"):

            # 진행 단계를 st.empty() placeholder로 표시 (st.status 미사용 → 배경 흐림 없음)
            progress_ph = st.empty()

            def _show_steps(steps: list[str]):
                """steps 리스트를 progress_ph 안에 렌더링합니다."""
                with progress_ph.container():
                    for s in steps:
                        st.write(s)

            steps: list[str] = []

            # ── Step 1: 질의 분석 ──
            steps.append("🔍 **[Step 1]** 질의 분석 중 (Analyzer Agent)…")
            _show_steps(steps)

            from agents import run_analyzer_agent
            analysis = run_analyzer_agent(user_input)

            steps[-1] = (
                f"✅ **[Step 1]** 질의 분석 완료  \n"
                f"  - 키워드: `{', '.join(analysis.keywords)}`  \n"
                f"  - 의도: `{analysis.intent}` | 시간 필터: `{analysis.time_filter}`  \n"
                f"  - 검색 쿼리: `{', '.join(analysis.search_queries)}`"
            )
            _show_steps(steps)

            # ── Step 2: 데이터 수집 혹은 직접 답변 결정 ──
            if not analysis.use_external_apis and not analysis.use_internal_db:
                steps.append("⚡ **[Step 2]** 검색 생략: 직접 답변을 작성합니다 (Direct Answer)…")
                _show_steps(steps)

                from agents import run_direct_answer_agent
                report = run_direct_answer_agent(user_input)

                # 진행 단계 지우고 최종 답변 표시
                progress_ph.empty()
                st.markdown(report.full_report)

                # 빈 통계 결과를 위한 빈 JudgedResult 설정
                from schemas import JudgedResult
                judged = JudgedResult()
            else:
                # ── Step 2: 데이터 수집 ──
                steps.append("🌐 **[Step 2]** 외부 API 데이터 수집 중 (arXiv · GitHub · HuggingFace)…")
                _show_steps(steps)

                from api_clients import collect_all
                collected = collect_all(
                    analysis.search_queries or [user_input],
                    max_per_source=max_results,
                    intent=analysis.intent,
                    use_semantic_scholar=use_semantic_scholar,
                )

                steps[-1] = (
                    f"✅ **[Step 2]** 수집 완료  \n"
                    f"  - 논문: {len(collected['papers'])}건 | "
                    f"모델: {len(collected['models'])}건 | "
                    f"코드: {len(collected['code_repos'])}건 | "
                    f"PwC: {len(collected['pwc_results'])}건"
                )
                _show_steps(steps)

                # ── Step 3: Vector DB 저장 & 내부 검색 ──
                from schemas import SearchResult
                from rag_module import build_vectorstore_from_collected_data, search_vectorstore

                internal_docs: list[str] = []
                if use_internal_db and (collected["papers"] or collected["models"] or collected["code_repos"]):
                    steps.append("📚 **[Step 3]** Vector DB 구축 및 검색 중…")
                    _show_steps(steps)

                    vectorstore = build_vectorstore_from_collected_data(
                        papers=collected["papers"],
                        models=collected["models"],
                        code_repos=collected["code_repos"],
                        persist=True,
                    )
                    if vectorstore:
                        internal_docs = search_vectorstore(user_input, vectorstore=vectorstore, k=50)

                    steps[-1] = f"✅ **[Step 3]** Vector DB 구축 및 검색 완료"
                    _show_steps(steps)

                search_result = SearchResult(
                    papers=collected["papers"],
                    models=collected["models"],
                    code_repos=collected["code_repos"],
                    pwc_results=collected["pwc_results"],
                    internal_docs=internal_docs,
                )

                # ── Step 4: Judge ──
                steps.append("⚖️ **[Step 4]** 검색 결과 검증 중 (Judge Agent)…")
                _show_steps(steps)

                from agents import run_judge_agent
                judged = run_judge_agent(user_input, search_result)

                steps[-1] = (
                    f"✅ **[Step 4]** 검색 결과 검증 완료  \n"
                    f"  - 논문 {len(judged.papers)}건 | "
                    f"모델 {len(judged.models)}건 | "
                    f"코드 {len(judged.code_repos)}건"
                )
                _show_steps(steps)

                # ── Step 5: Summary ──
                steps.append("📝 **[Step 5]** 리포트 생성 중 (Summary Agent)…")
                _show_steps(steps)

                from agents import run_summary_agent
                report = run_summary_agent(user_input, judged, analysis, use_semantic_scholar=use_semantic_scholar)

                # 진행 단계 지우고 최종 리포트 표시
                progress_ph.empty()

                # API 오류/지연 경고가 있는 경우 표시 및 리포트 본문에 추가
                if collected.get("warnings"):
                    st.warning(
                        "⚠️ **일부 API 요청 중 지연 또는 제한(Rate Limit/Timeout)이 발생하였습니다:**\n" +
                        "\n".join(f"- {w}" for w in collected["warnings"])
                    )
                    warning_suffix = "\n\n---\n\n⚠️ **API 수집 경고:**\n" + "\n".join(f"- {w}" for w in collected["warnings"])
                    report.full_report += warning_suffix

                # 통계 카드
                stat_cols = st.columns(4)
                stat_cols[0].metric("📄 논문", f"{len(judged.papers)}건")
                stat_cols[1].metric("🤖 모델", f"{len(judged.models)}건")
                stat_cols[2].metric("💻 코드", f"{len(judged.code_repos)}건")
                stat_cols[3].metric("🔗 PwC", f"{len(judged.pwc_results)}건")

                st.divider()
                st.markdown(report.full_report)

        # 3) 어시스턴트 메시지 기록 및 렌더링 종료
        st.session_state.messages_tech.append({
            "role": "assistant",
            "content": report.full_report,
            "metrics": {
                "papers": len(judged.papers),
                "models": len(judged.models),
                "repos": len(judged.code_repos),
                "pwc": len(judged.pwc_results),
            }
        })
        st.session_state.is_generating = False
        st.rerun()


# ──────────────────────────────────────────────
# 메인: PDF QA 모드
# ──────────────────────────────────────────────

elif mode == "📄 PDF QA":
    st.markdown("### 📄 PDF 기반 RAG 질의응답")
    st.markdown("업로드한 PDF 문서에 대해 질문해 보세요.")

    if "messages_pdf" not in st.session_state:
        st.session_state.messages_pdf = []

    if uploaded_file:
        # 파일을 로컬에 임시 저장
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 세션 상태를 사용하여 체인을 한 번만 생성 (성능 최적화)
        if "rag_chain" not in st.session_state or st.session_state.get("current_pdf") != uploaded_file.name:
            with st.spinner("📊 문서를 분석 중입니다..."):
                st.session_state.rag_chain = create_rag_chain(temp_path)
                st.session_state.current_pdf = uploaded_file.name
            st.success(f"✅ '{uploaded_file.name}' 분석 완료!")

        # 기존 대화 표시
        for message in st.session_state.messages_pdf:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 사용자 입력 처리
        if prompt := st.chat_input("PDF에 대해 질문하세요"):
            st.session_state.messages_pdf.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    response = st.session_state.rag_chain(prompt)
                    st.markdown(response)
            st.session_state.messages_pdf.append({"role": "assistant", "content": response})
    else:
        st.info("👈 왼쪽 사이드바에서 PDF 파일을 업로드하면 대화가 시작됩니다.")