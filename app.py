"""
Streamlit UI — 오픈소스 AI 개발 생태계 오케스트레이터
통합 연구 워크스페이스: 질의 검색 & PDF 문서 첨부 비교 분석
"""

import streamlit as st
import os
import hashlib
import tempfile

from rag_module import extract_pdf_info


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
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 헤더
# ──────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🔬 AI Tech Orchestrator</h1>
    <p>오픈소스 AI 개발 생태계 오케스트레이터 — 자유로운 PDF 논문 첨부 & 논문 · 모델 · 코드 통합 탐색</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 사이드바: 옵션 및 PDF 문서 첨부
# ──────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ 검색 및 첨부 설정")

    st.subheader("📎 PDF 문서 첨부 (선택)")
    uploaded_file = st.file_uploader(
        "비공개/미등록 PDF 논문을 첨부하여 최신 생태계 자원과 비교 분석하세요",
        type=['pdf'],
        help="PDF를 첨부하면 PDF 텍스트 분석기가 자동 작동하여 외부 검색 결과와 통합 비교 리포트를 생성합니다.",
    )

    st.divider()
    st.subheader("🔧 검색 옵션")
    max_results = st.slider("소스당 최대 결과 수", 5, 30, 30)
    use_internal_db = st.checkbox("내부 Vector DB 포함", value=True)
    use_semantic_scholar = st.checkbox("Semantic Scholar 사용 (끄면 arXiv 사용)", value=False)

    st.divider()
    st.subheader("📡 API 연결 상태")

    has_hf_token = bool(os.getenv("HF_TOKEN"))
    has_gh_token = bool(os.getenv("GITHUB_TOKEN"))

    st.caption(f"{'🟢' if has_hf_token else '🟡'} 🤖 **LLM 추론 API (Qwen3-8B)**: {'HF_TOKEN 연결됨' if has_hf_token else '무료 인퍼런스 (자동 폴백)'}")
    st.caption(f"{'🟢' if has_hf_token else '🟡'} 🤗 **HF Hub 검색 API (Model)**: {'HF_TOKEN 연결됨' if has_hf_token else '비인증 공개 API'}")
    st.caption(f"{'🟢' if has_gh_token else '🟡'} 💻 **GitHub 검색 API (Repo)**: {'GITHUB_TOKEN 연결됨' if has_gh_token else '비인증 공개 API'}")

    st.divider()
    st.caption("Powered by Qwen3-8B · LangGraph · FAISS")

    # 대화 초기화 버튼
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ──────────────────────────────────────────────
# PDF 문서 분석 처리 (첨부 시 자동 작동)
# ──────────────────────────────────────────────

attached_pdf_context = None
attached_pdf_docs = None

if uploaded_file:
    file_bytes = uploaded_file.getbuffer()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    file_key = f"{uploaded_file.name}_{file_hash}"

    if st.session_state.get("current_pdf_key") != file_key:
        with st.spinner("📊 첨부된 PDF 논문을 분석 중입니다..."):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                tmp_file.write(file_bytes)
                temp_path = tmp_file.name

            try:
                pdf_context, pdf_docs = extract_pdf_info(temp_path)
                st.session_state.current_pdf_key = file_key
                st.session_state.current_pdf_name = uploaded_file.name
                st.session_state.pdf_context = pdf_context
                st.session_state.pdf_docs = pdf_docs
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        st.sidebar.success(f"✅ '{uploaded_file.name}' 분석 완료!")

    attached_pdf_context = st.session_state.get("pdf_context")
    attached_pdf_docs = st.session_state.get("pdf_docs")
    st.info(f"📄 **현재 첨부된 문서:** `{st.session_state.get('current_pdf_name')}` (PDF 맥락 및 Vector DB 인덱싱 적용 됨)")


# ──────────────────────────────────────────────
# 메인 통합 워크스페이스
# ──────────────────────────────────────────────

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
    "4️⃣ PDF 첨부 및 비교 질문형": [
        "첨부한 논문의 핵심 기법을 요약하고, 관련 최신 arXiv 논문 및 오픈소스 구현체와 비교해줘",
        "이 논문에서 제시된 모델 구조와 유사한 Hugging Face 모델과 GitHub 프로젝트를 찾아줘",
    ],
}

# ── 예시 질문 (클릭 시 바로 실행) ──
with st.expander(
    "💡 벤치마크 예시 질문 — 클릭 시 바로 실행",
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
typed_input = st.chat_input("질문이나 연구 주제를 입력하세요 (예: DPO 트렌드, 첨부 논문과 최신 코드 비교 등)")
st.caption("💡 외부 API 동시 수집 및 LLM 비교 분석 리포트 생성에 평균 10~15초 소요됩니다.")

if "pending_query" in st.session_state:
    st.session_state.messages_tech.append({"role": "user", "content": st.session_state.pop("pending_query")})
    st.session_state.is_generating = True
    st.rerun()
elif typed_input:
    st.session_state.messages_tech.append({"role": "user", "content": typed_input})
    st.session_state.is_generating = True
    st.rerun()

# ── 에이전트 오케스트레이터 실행 ──
if st.session_state.get("is_generating", False):
    user_input = st.session_state.messages_tech[-1]["content"]

    with st.chat_message("assistant"):
        progress_ph = st.empty()

        def _show_steps(steps: list[str]):
            with progress_ph.container():
                st.info("⏱️ 외부 API 수집 및 LLM 인퍼런스 리포트 작성 중입니다 (처리 시 다소 시간이 소요될 수 있습니다).")
                for s in steps:
                    st.write(s)

        steps: list[str] = []

        from workflow import build_workflow, AgentState

        workflow_app = build_workflow().compile()

        initial_state: AgentState = {
            "user_query": user_input,
            "max_results": max_results,
            "use_internal_db": use_internal_db,
            "use_semantic_scholar": use_semantic_scholar,
            "pdf_path": st.session_state.get("current_pdf_name"),
            "pdf_context": attached_pdf_context,
            "pdf_docs": attached_pdf_docs,
            "analysis": None,
            "search_result": None,
            "judged_result": None,
            "final_report": None,
            "warnings": [],
            "status_log": [],
        }

        steps.append("🔍 **[Step 1]** 질의 및 PDF 맥락 분석 중 (Analyzer Agent)…")
        _show_steps(steps)

        accumulated_state: dict = dict(initial_state)

        for event in workflow_app.stream(initial_state):
            for node_name, node_output in event.items():
                accumulated_state.update(node_output)

                if node_name == "analyze":
                    analysis = node_output.get("analysis")
                    if analysis:
                        pdf_tag = " (📄 PDF 맥락 포함)" if attached_pdf_context else ""
                        steps[-1] = (
                            f"✅ **[Step 1]** 분석 완료{pdf_tag}  \n"
                            f"  - 키워드: `{', '.join(analysis.keywords)}`  \n"
                            f"  - 의도: `{analysis.intent}` | 시간 필터: `{analysis.time_filter}`  \n"
                            f"  - 검색 쿼리: `{', '.join(analysis.search_queries)}`"
                        )
                        _show_steps(steps)

                elif node_name == "direct_answer":
                    steps.append("⚡ **[Step 2]** 검색 생략: 직접 답변 작성 완료 (Direct Answer)")
                    _show_steps(steps)

                elif node_name in ("collect_external", "collect_both"):
                    sr = node_output.get("search_result")
                    if sr:
                        steps.append(
                            f"✅ **[Step 2]** 수집 및 Vector DB 검색 완료  \n"
                            f"  - 논문: {len(sr.papers)}건 | "
                            f"모델: {len(sr.models)}건 | "
                            f"코드: {len(sr.code_repos)}건 | "
                            f"PwC: {len(sr.pwc_results)}건"
                            + (f" | 내부/PDF 검색: {len(sr.internal_docs)}건" if sr.internal_docs else "")
                        )
                        _show_steps(steps)

                elif node_name == "search_internal":
                    sr = node_output.get("search_result")
                    if sr:
                        steps.append(
                            f"✅ **[Step 2]** Vector DB / PDF 검색 완료  \n"
                            f"  - 검색 문서: {len(sr.internal_docs)}건"
                        )
                        _show_steps(steps)

                elif node_name == "judge":
                    judged = node_output.get("judged_result")
                    if judged:
                        steps.append(
                            f"✅ **[Step 3]** 검색 결과 검증 완료 (Judge Agent)  \n"
                            f"  - 논문 {len(judged.papers)}건 | "
                            f"모델 {len(judged.models)}건 | "
                            f"코드 {len(judged.code_repos)}건"
                        )
                        _show_steps(steps)

                elif node_name == "summarize":
                    steps.append("📝 **[Step 4]** 최종 비교 리포트 생성 완료 (Summary Agent)")
                    _show_steps(steps)

        progress_ph.empty()

        final_report = accumulated_state.get("final_report")
        report_text = final_report.full_report if final_report else "리포트를 생성할 수 없습니다."
        warnings = accumulated_state.get("warnings", [])

        if warnings:
            st.warning(
                "⚠️ **일부 API 요청 중 지연 또는 제한(Rate Limit/Timeout)이 발생하였습니다:**\n" +
                "\n".join(f"- {w}" for w in warnings)
            )
            warning_suffix = "\n\n---\n\n⚠️ **API 수집 경고:**\n" + "\n".join(f"- {w}" for w in warnings)
            report_text += warning_suffix

        judged = accumulated_state.get("judged_result")
        p_count = len(judged.papers) if judged else 0
        m_count = len(judged.models) if judged else 0
        r_count = len(judged.code_repos) if judged else 0
        pwc_count = len(judged.pwc_results) if judged else 0

        # 통계 카드
        stat_cols = st.columns(4)
        stat_cols[0].metric("📄 논문", f"{p_count}건")
        stat_cols[1].metric("🤖 모델", f"{m_count}건")
        stat_cols[2].metric("💻 코드", f"{r_count}건")
        stat_cols[3].metric("🔗 PwC", f"{pwc_count}건")

        st.divider()
        st.markdown(report_text)

    # 어시스턴트 메시지 기록 및 종료
    st.session_state.messages_tech.append({
        "role": "assistant",
        "content": report_text,
        "metrics": {
            "papers": p_count,
            "models": m_count,
            "repos": r_count,
            "pwc": pwc_count,
        }
    })
    st.session_state.is_generating = False
    st.rerun()