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

    EXAMPLES = [
        "DPO 알고리즘 트렌드와 바로 테스트 가능한 오픈소스 모델을 알려줘",
        "LLM Alignment 관련 최신 기법과 오픈소스 모델, GitHub 리포를 찾아줘",
        "RAG 기술의 최신 발전과 구현 코드가 있는 프로젝트를 추천해줘",
        "멀티모달 LLM의 최신 논문과 Hugging Face 모델을 비교해줘",
    ]

    # ── 예시 질문 (항상 표시, 대화 없을 때는 펼침) ──
    with st.expander(
        "💡 예시 질문 — 클릭하면 바로 시작됩니다",
        expanded=len(st.session_state.messages_tech) == 0,
    ):
        ex_cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            if ex_cols[i % 2].button(f"📌 {ex}", key=f"ex_{i}", use_container_width=True):
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

            # ── Step 2: 데이터 수집 ──
            steps.append("🌐 **[Step 2]** 외부 API 데이터 수집 중 (Semantic Scholar · GitHub · HuggingFace)…")
            _show_steps(steps)

            from api_clients import collect_all
            collected = collect_all(
                analysis.search_queries or [user_input],
                max_per_source=max_results,
                intent=analysis.intent,
            )

            steps[-1] = (
                f"✅ **[Step 2]** 수집 완료  \n"
                f"  - 논문: {len(collected['papers'])}건 | "
                f"모델: {len(collected['models'])}건 | "
                f"코드: {len(collected['code_repos'])}건 | "
                f"PwC: {len(collected['pwc_results'])}건"
            )
            _show_steps(steps)

            # ── Step 2.5: Vector DB 저장 & 내부 검색 ──
            from schemas import SearchResult
            from rag_module import build_vectorstore_from_collected_data, search_vectorstore

            internal_docs: list[str] = []
            if use_internal_db and (collected["papers"] or collected["models"] or collected["code_repos"]):
                steps.append("📚 **[Step 2.5]** Vector DB 저장 & 내부 검색 중…")
                _show_steps(steps)

                vectorstore = build_vectorstore_from_collected_data(
                    papers=collected["papers"],
                    models=collected["models"],
                    code_repos=collected["code_repos"],
                    persist=True,
                )
                if vectorstore:
                    internal_docs = search_vectorstore(user_input, vectorstore=vectorstore, k=5)

                steps[-1] = f"✅ **[Step 2.5]** 내부 검색 완료 — {len(internal_docs)}건"
                _show_steps(steps)

            search_result = SearchResult(
                papers=collected["papers"],
                models=collected["models"],
                code_repos=collected["code_repos"],
                pwc_results=collected["pwc_results"],
                internal_docs=internal_docs,
            )

            # ── Step 3: Judge ──
            steps.append("⚖️ **[Step 3]** 결과 검증 중 (Judge Agent)…")
            _show_steps(steps)

            from agents import run_judge_agent
            judged = run_judge_agent(user_input, search_result)

            steps[-1] = (
                f"✅ **[Step 3]** 검증 완료  \n"
                f"  - 논문 {len(judged.papers)}건 | "
                f"모델 {len(judged.models)}건 | "
                f"코드 {len(judged.code_repos)}건"
            )
            _show_steps(steps)

            # ── Step 4: Summary ──
            steps.append("📝 **[Step 4]** 리포트 생성 중 (Summary Agent)…")
            _show_steps(steps)

            from agents import run_summary_agent
            report = run_summary_agent(user_input, judged, analysis)

            # 진행 단계 지우고 최종 리포트 표시
            progress_ph.empty()

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