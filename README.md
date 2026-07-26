# 🔬 AI Tech Orchestrator

오픈소스 AI 개발 생태계 오케스트레이터입니다.  
최신 AI 기술(논문, 모델, 오픈소스 코드) 트렌드를 사용자가 자연어로 탐색할 수 있도록 돕는 검색 및 요약 시스템(RAG + 멀티 에이전트 아키텍처)을 제공합니다.

## ✨ 주요 기능
1. **🔍 AI 기술 탐색 모드 (기본)**:
   - **자연어 기반 통합 검색**: "DPO 트렌드", "멀티모달 LLM", "RAG 코드" 등 자연어 질의를 지능적으로 분석하여 관련 생태계 데이터를 자동 수집합니다.
   - **4가지 외부 API 연동**:
     - **Semantic Scholar API**: 인용수 기반 고품질 논문 검색 (기존 arXiv 대체)
     - **GitHub Search API**: Python 기반 AI 구현체 리포지토리 검색 (stars 순)
     - **Hugging Face Hub API**: 오픈소스 모델 및 체크포인트 검색 (downloads/likes 순)
     - **Papers with Code API**: 논문-코드 매핑 정보 조회
   - **멀티 에이전트 시스템**: Analyzer, Router, Judge, Summary 에이전트가 순차적으로 동작하여 데이터를 검증하고 통합 리포트를 작성합니다.
   - **Vector DB 자동 구축**: 검색된 결과를 FAISS(bge-small-en-v1.5 임베딩) 기반 로컬 벡터 DB에 저장하여 내부 검색을 병행합니다.
   - **결과 통계 카드**: 리포트 출력 시 수집된 논문/모델/코드/PwC 결과 수를 한눈에 볼 수 있도록 UI 상단에 카드로 제공합니다.

2. **📄 PDF QA 모드**:
   - 논문 PDF 등을 직접 업로드하고 문서 내용에 대해 Q&A를 수행할 수 있는 로컬 문서 기반 RAG 시스템입니다.

## 🛠 기술 스택
- **언어**: Python 3.10+
- **프레임워크**: Streamlit, LangChain, LangGraph
- **LLM**: **Qwen/Qwen3-8B** (Hugging Face Inference API)
- **임베딩**: **BAAI/bge-small-en-v1.5** (로컬 추론)
- **Vector DB**: FAISS

## 🚀 설치 및 실행 방법

### 1. 환경 설정 및 의존성 설치
```bash
# 가상 환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정
프로젝트 루트 폴더에 `.env` 파일을 생성하고 아래 항목을 기입하세요.
(Hugging Face API 토큰은 필수이며, GitHub 토큰은 선택 사항입니다.)
```env
HF_TOKEN=your_hugging_face_token_here
GITHUB_TOKEN=your_github_token_here
```

### 3. 애플리케이션 실행
```bash
streamlit run app.py
```
브라우저에서 `http://localhost:8501`로 접속하여 이용할 수 있습니다.

## 📦 폴더 구조
- `app.py`: Streamlit 기반 웹 애플리케이션 프론트엔드 및 진입점
- `agents.py`: LLM 시스템 프롬프트를 활용하는 에이전트 로직 (Analyzer, Judge, Summary)
- `api_clients.py`: 외부 API 연동 및 데이터 크롤링 모듈 (Semantic Scholar, GitHub, HF, PwC)
- `rag_module.py`: LangChain 기반 FAISS 벡터 DB 구축, 검색, 및 PDF QA 체인 로직
- `workflow.py`: LangGraph를 활용한 멀티 에이전트 상태 기반 파이프라인 그래프 정의
- `schemas.py`: Pydantic을 이용한 데이터 구조(모델) 정의
- `vectorstore_data/`: (자동 생성) FAISS 인덱스 로컬 저장소
