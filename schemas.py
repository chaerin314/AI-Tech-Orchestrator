"""
데이터 모델 & 스키마 정의
논문, 모델, 코드 리포지토리의 공통 스키마와
에이전트 간 데이터 전달을 위한 구조체를 정의합니다.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# 1. 외부 데이터 소스별 정보 모델
# ──────────────────────────────────────────────

class PaperInfo(BaseModel):
    """arXiv / Papers with Code에서 수집한 논문 정보"""
    title: str = Field(description="논문 제목")
    authors: list[str] = Field(default_factory=list, description="저자 목록")
    abstract: str = Field(default="", description="초록 요약")
    published: str = Field(default="", description="출판일 (YYYY-MM-DD)")
    arxiv_id: str = Field(default="", description="arXiv ID")
    arxiv_url: str = Field(default="", description="arXiv 논문 링크")
    pdf_url: str = Field(default="", description="PDF 다운로드 링크")
    categories: list[str] = Field(default_factory=list, description="arXiv 카테고리")
    source: str = Field(default="arXiv", description="데이터 출처")


class ModelInfo(BaseModel):
    """Hugging Face Hub에서 수집한 모델 정보"""
    model_id: str = Field(description="모델 ID (예: meta-llama/Llama-3)")
    author: str = Field(default="", description="모델 제작자/조직")
    pipeline_tag: str = Field(default="", description="파이프라인 태그 (text-generation 등)")
    downloads: int = Field(default=0, description="총 다운로드 수")
    likes: int = Field(default=0, description="좋아요 수")
    tags: list[str] = Field(default_factory=list, description="태그 목록")
    last_modified: str = Field(default="", description="최종 수정일")
    url: str = Field(default="", description="Hugging Face 모델 페이지 URL")
    description: str = Field(default="", description="모델 카드 설명 (발췌)")
    source: str = Field(default="HuggingFace", description="데이터 출처")


class CodeRepoInfo(BaseModel):
    """GitHub에서 수집한 코드 리포지토리 정보"""
    name: str = Field(description="리포지토리 이름")
    full_name: str = Field(default="", description="전체 경로 (owner/repo)")
    description: str = Field(default="", description="리포지토리 설명")
    url: str = Field(default="", description="GitHub 리포지토리 URL")
    stars: int = Field(default=0, description="스타 수")
    forks: int = Field(default=0, description="포크 수")
    language: str = Field(default="", description="주요 프로그래밍 언어")
    topics: list[str] = Field(default_factory=list, description="토픽 태그")
    updated_at: str = Field(default="", description="최종 업데이트일")
    readme_excerpt: str = Field(default="", description="README 발췌")
    source: str = Field(default="GitHub", description="데이터 출처")


class PapersWithCodeResult(BaseModel):
    """Papers with Code에서 수집한 논문-코드 매핑 정보"""
    paper_title: str = Field(default="", description="논문 제목")
    paper_url: str = Field(default="", description="PwC 논문 페이지 URL")
    arxiv_id: str = Field(default="", description="arXiv ID")
    num_stars: int = Field(default=0, description="GitHub 스타 수 합계")
    repositories: list[dict] = Field(default_factory=list, description="연결된 GitHub 리포지토리 목록")
    source: str = Field(default="PapersWithCode", description="데이터 출처")


# ──────────────────────────────────────────────
# 2. 에이전트 간 데이터 전달 스키마
# ──────────────────────────────────────────────

class AnalysisResult(BaseModel):
    """Analyzer Agent 출력: 질의 분석 결과"""
    original_query: str = Field(description="사용자 원본 질의")
    keywords: list[str] = Field(default_factory=list, description="추출된 핵심 기술 키워드")
    search_queries: list[str] = Field(
        default_factory=list,
        description="각 API에 전달할 검색 쿼리 목록"
    )
    intent: str = Field(
        default="general",
        description="질의 의도 (trend / comparison / implementation / general)"
    )
    time_filter: str = Field(
        default="recent",
        description="시간 범위 필터 (recent / last_year / all)"
    )
    use_internal_db: bool = Field(
        default=True,
        description="내부 Vector DB 검색 사용 여부"
    )
    use_external_apis: bool = Field(
        default=True,
        description="외부 API 호출 사용 여부"
    )


class SearchResult(BaseModel):
    """통합 검색 결과"""
    papers: list[PaperInfo] = Field(default_factory=list)
    models: list[ModelInfo] = Field(default_factory=list)
    code_repos: list[CodeRepoInfo] = Field(default_factory=list)
    pwc_results: list[PapersWithCodeResult] = Field(default_factory=list)
    internal_docs: list[str] = Field(
        default_factory=list,
        description="내부 Vector DB에서 검색된 관련 문서 텍스트"
    )


class JudgedResult(BaseModel):
    """Judge Agent 출력: 검증·정제된 결과"""
    papers: list[PaperInfo] = Field(default_factory=list)
    models: list[ModelInfo] = Field(default_factory=list)
    code_repos: list[CodeRepoInfo] = Field(default_factory=list)
    pwc_results: list[PapersWithCodeResult] = Field(default_factory=list)
    internal_docs: list[str] = Field(default_factory=list)
    quality_notes: str = Field(default="", description="검증 과정에서의 참고 사항")


class FinalReport(BaseModel):
    """Summary Agent 출력: 최종 리포트"""
    trend_summary: str = Field(default="", description="핵심 기술 트렌드 요약")
    comparison_table: str = Field(default="", description="논문/모델/코드 비교표 (마크다운)")
    paper_summaries: str = Field(default="", description="주요 논문 목록 및 요약")
    model_recommendations: str = Field(default="", description="추천 모델 및 근거")
    code_recommendations: str = Field(default="", description="추천 코드 리포지토리 및 근거")
    limitations: str = Field(default="", description="한계점 및 추가 확인 사항")
    full_report: str = Field(default="", description="전체 통합 리포트 (마크다운)")
