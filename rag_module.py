"""
RAG 모듈 (확장)
기존 PDF 기반 RAG 체인 + 외부 API 수집 데이터의 Vector DB 저장/검색 기능을 제공합니다.
LLM: Qwen/Qwen3-8B (HuggingFace Inference API)
Embedding: BAAI/bge-small-en-v1.5 (HuggingFace, 로컬 추론)
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.language_models.llms import BaseLLM
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from huggingface_hub import InferenceClient

from llm_client import qwen_chat

# .env 파일에 저장된 API 키 로드
load_dotenv()

# ──────────────────────────────────────────────
# 공통 설정
# ──────────────────────────────────────────────
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "vectorstore_data")

# HuggingFace 임베딩 (로컬 추론, API 키 불필요)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


def _qwen_invoke(prompt_text: str, max_tokens: int = 1500) -> str:
    """Qwen3-8B 공용 클라이언트 호출 헬퍼"""
    return qwen_chat(
        system="You are a helpful assistant. Answer in Korean.",
        user=prompt_text,
        max_tokens=max_tokens,
    )


# ──────────────────────────────────────────────
# 1. 기존 PDF RAG 체인 (유지)
# ──────────────────────────────────────────────

def create_rag_chain(pdf_path):
    """PDF 파일 기반 RAG 체인을 생성합니다. (Qwen3-8B 기반)"""
    # [1단계] 문서 로드 (Document Load)
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    # [2단계] 문서 분할 (Text Split)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,         # 청크 사이즈 조절
        chunk_overlap=100,      # 오버랩 조절
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_documents = text_splitter.split_documents(docs)

    # [3~4단계] 임베딩 및 벡터 DB 저장 (Embedding & Vector DB)
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # [5단계] 검색기(Retriever) 생성
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # [6~8단계] Qwen3-8B 기반 RAG 함수형 체인
    def rag_chain_fn(question: str) -> str:
        # 관련 문서 검색
        related_docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in related_docs)
        prompt_text = f"""다음 컨텍스트만을 참고하여 질문에 답하세요.

컨텍스트:
{context}

질문: {question}

한국어로 답변:"""
        return _qwen_invoke(prompt_text)

    return rag_chain_fn


# ──────────────────────────────────────────────
# 2. 외부 수집 데이터 → Vector DB 저장/검색
# ──────────────────────────────────────────────

def _convert_to_documents(
    papers: list = None,
    models: list = None,
    code_repos: list = None,
) -> list[Document]:
    """API에서 수집한 데이터를 LangChain Document 객체로 변환합니다."""
    documents = []

    if papers:
        for p in papers:
            content = f"[논문] {p.title}\n저자: {', '.join(p.authors[:3])}\n초록: {p.abstract}\n날짜: {p.published}"
            metadata = {
                "source": "Semantic Scholar",
                "type": "paper",
                "paper_id": p.paper_id,
                "url": p.paper_url,
                "title": p.title,
            }
            documents.append(Document(page_content=content, metadata=metadata))

    if models:
        for m in models:
            content = f"[모델] {m.model_id}\n제작자: {m.author}\n태그: {', '.join(m.tags[:5])}\n다운로드: {m.downloads:,}\n설명: {m.description}"
            metadata = {
                "source": "HuggingFace",
                "type": "model",
                "model_id": m.model_id,
                "url": m.url,
                "title": m.model_id,
            }
            documents.append(Document(page_content=content, metadata=metadata))

    if code_repos:
        for r in code_repos:
            content = f"[코드] {r.full_name}\n설명: {r.description}\n언어: {r.language}\n⭐ {r.stars:,}\n토픽: {', '.join(r.topics[:5])}"
            metadata = {
                "source": "GitHub",
                "type": "code",
                "full_name": r.full_name,
                "url": r.url,
                "title": r.full_name,
            }
            documents.append(Document(page_content=content, metadata=metadata))

    return documents


def extract_pdf_info(pdf_path: str) -> tuple[str, list[Document]]:
    """PDF 파일에서 전체 맥락 텍스트와 분할 청크 Document 목록을 추출합니다."""
    if not pdf_path or not os.path.exists(pdf_path):
        return "", []

    try:
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        split_docs = text_splitter.split_documents(docs)

        # PDF 전반부 텍스트 추출 (최대 1500자)
        full_text = "\n\n".join(doc.page_content for doc in docs[:5])[:1500]
        return full_text, split_docs
    except Exception as e:
        print(f"[PDF Extraction Error] {e}")
        return "", []


def build_vectorstore_from_collected_data(
    papers: list = None,
    models: list = None,
    code_repos: list = None,
    pdf_docs: list[Document] = None,
    persist: bool = False,
) -> FAISS | None:
    """수집된 외부 데이터 및 업로드된 PDF 청크를 FAISS Vector DB에 저장합니다.

    Args:
        papers: PaperInfo 리스트
        models: ModelInfo 리스트
        code_repos: CodeRepoInfo 리스트
        pdf_docs: 업로드된 PDF의 LangChain Document 청크 리스트
        persist: True면 로컬 디스크에 저장

    Returns:
        FAISS vectorstore 또는 None (문서 없을 때)
    """
    documents = _convert_to_documents(papers, models, code_repos)
    if pdf_docs:
        documents.extend(pdf_docs)

    if not documents:
        return None

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)

    # 임베딩 → FAISS 저장
    vectorstore = FAISS.from_documents(documents=split_docs, embedding=embeddings)

    # 로컬 영속화
    if persist:
        os.makedirs(VECTORSTORE_DIR, exist_ok=True)
        vectorstore.save_local(VECTORSTORE_DIR)
        print(f"[RAG] Vector DB 저장 완료: {VECTORSTORE_DIR}")

    return vectorstore


def search_vectorstore(
    query: str,
    vectorstore: FAISS = None,
    load_from_disk: bool = False,
    k: int = 5,
) -> list[str]:
    """FAISS Vector DB에서 유사 문서를 검색합니다.

    Args:
        query: 검색 쿼리
        vectorstore: 메모리에 있는 FAISS 인스턴스 (없으면 디스크에서 로드)
        load_from_disk: True면 디스크에서 로드 시도
        k: 반환할 문서 수

    Returns:
        관련 문서 텍스트 목록
    """
    if vectorstore is None and load_from_disk:
        if os.path.exists(VECTORSTORE_DIR):
            try:
                vectorstore = FAISS.load_local(
                    VECTORSTORE_DIR,
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
            except Exception as e:
                print(f"[RAG] Vector DB 로드 실패: {e}")
                return []
        else:
            return []

    if vectorstore is None:
        return []

    try:
        results = vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
    except Exception as e:
        print(f"[RAG] 검색 실패: {e}")
        return []