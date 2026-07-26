"""
외부 API 클라이언트 모듈
arXiv, GitHub, Hugging Face Hub, Papers with Code API를 연동하여
논문, 모델, 코드 메타데이터를 수집합니다.
질의 의도(intent)에 따라 각 소스별 검색 전략을 최적화합니다.
"""

from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from typing import Optional

import httpx
from dotenv import load_dotenv

from schemas import PaperInfo, ModelInfo, CodeRepoInfo, PapersWithCodeResult

load_dotenv()

# ──────────────────────────────────────────────
# 공통 유틸리티
# ──────────────────────────────────────────────

DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 2
RETRY_DELAY = 1.0


def _request_with_retry(
    method: str,
    url: str,
    headers: Optional[dict] = None,
    params: Optional[dict] = None,
    retries: int = MAX_RETRIES,
) -> httpx.Response | None:
    """HTTP 요청 + 재시도 로직"""
    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=DEFAULT_TIMEOUT, follow_redirects=True) as client:
                resp = client.request(method, url, headers=headers, params=params)
                if resp.status_code == 429:  # Rate Limit
                    wait = RETRY_DELAY * (2 ** attempt)
                    print(f"[Rate Limit] {url} — {wait}초 후 재시도 ({attempt+1}/{retries})")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp
        except httpx.HTTPStatusError as e:
            print(f"[HTTP Error] {url}: {e.response.status_code}")
            if attempt < retries:
                time.sleep(RETRY_DELAY)
                continue
            return None
        except httpx.RequestError as e:
            print(f"[Request Error] {url}: {e}")
            if attempt < retries:
                time.sleep(RETRY_DELAY)
                continue
            return None
    return None


# ──────────────────────────────────────────────
# 1. arXiv API 클라이언트
# ──────────────────────────────────────────────

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}

# ML/AI 관련 카테고리 필터
ARXIV_ML_CATS = "(cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:stat.ML)"


def search_arxiv(query: str, max_results: int = 30, intent: str = "general") -> list[PaperInfo]:
    """arXiv API로 논문을 검색합니다.

    Args:
        query: 검색 쿼리
        max_results: 최대 결과 수
        intent: 질의 의도 (trend/comparison/implementation/general)
    """
    # 따옴표로 감싸 정확도 향상
    quoted = f'"{query}"' if " " in query else query

    # ML 카테고리 필터를 추가해 AI/ML 관련 논문만 수집
    search_q = f"all:{quoted} AND {ARXIV_ML_CATS}"

    # 의도별 정렬 전략
    sort_by = "submittedDate" if intent in ("trend",) else "relevance"

    params = {
        "search_query": search_q,
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": "descending",
    }
    resp = _request_with_retry("GET", ARXIV_API_URL, params=params)
    if resp is None:
        return []

    papers: list[PaperInfo] = []
    try:
        root = ET.fromstring(resp.text)
        entries = root.findall("atom:entry", ARXIV_NS)

        for entry in entries:
            title = entry.findtext("atom:title", "", ARXIV_NS).strip().replace("\n", " ")
            abstract = entry.findtext("atom:summary", "", ARXIV_NS).strip().replace("\n", " ")
            published = entry.findtext("atom:published", "", ARXIV_NS)[:10]

            authors = []
            for author_el in entry.findall("atom:author", ARXIV_NS):
                name = author_el.findtext("atom:name", "", ARXIV_NS)
                if name:
                    authors.append(name)

            # arXiv ID & 링크 추출
            arxiv_id = ""
            arxiv_url = ""
            pdf_url = ""
            for link in entry.findall("atom:link", ARXIV_NS):
                href = link.get("href", "")
                if link.get("title") == "pdf":
                    pdf_url = href
                elif "abs" in href:
                    arxiv_url = href
                    arxiv_id = href.split("/abs/")[-1]

            categories = []
            for cat in entry.findall("atom:category", ARXIV_NS):
                term = cat.get("term", "")
                if term:
                    categories.append(term)

            papers.append(PaperInfo(
                title=title,
                authors=authors[:5],
                abstract=abstract[:500],
                published=published,
                arxiv_id=arxiv_id,
                arxiv_url=arxiv_url,
                pdf_url=pdf_url,
                categories=categories,
            ))
    except ET.ParseError as e:
        print(f"[arXiv Parse Error] {e}")

    return papers


# ──────────────────────────────────────────────
# 2. GitHub Search API 클라이언트
# ──────────────────────────────────────────────

GITHUB_API_URL = "https://api.github.com/search/repositories"


def search_github(query: str, max_results: int = 30, intent: str = "general") -> list[CodeRepoInfo]:
    """GitHub Search API로 리포지토리를 검색합니다."""
    github_token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    # 의도별 쿼리 보강
    if intent == "implementation":
        # 구현 중심: Python + 최소 스타 필터
        q = f"{query} language:Python stars:>50"
    elif intent == "trend":
        # 최신 트렌드: 최근 업데이트 우선
        q = f"{query} language:Python pushed:>2023-01-01"
    else:
        q = f"{query} language:Python"

    params = {
        "q": q,
        "sort": "stars",
        "order": "desc",
        "per_page": min(max_results, 100),
    }
    resp = _request_with_retry("GET", GITHUB_API_URL, headers=headers, params=params)
    if resp is None:
        return []

    repos: list[CodeRepoInfo] = []
    try:
        data = resp.json()
        for item in data.get("items", [])[:max_results]:
            repos.append(CodeRepoInfo(
                name=item.get("name", ""),
                full_name=item.get("full_name", ""),
                description=(item.get("description") or "")[:300],
                url=item.get("html_url", ""),
                stars=item.get("stargazers_count", 0),
                forks=item.get("forks_count", 0),
                language=item.get("language") or "",
                topics=item.get("topics", []),
                updated_at=(item.get("updated_at") or "")[:10],
            ))
    except Exception as e:
        print(f"[GitHub Parse Error] {e}")

    return repos


# ──────────────────────────────────────────────
# 3. Hugging Face Hub API 클라이언트
# ──────────────────────────────────────────────

HF_API_URL = "https://huggingface.co/api/models"

# 의도별 파이프라인 태그 힌트
_INTENT_TO_PIPELINE = {
    "trend": None,          # 필터 없음 (광범위하게)
    "comparison": None,
    "implementation": "text-generation",
    "general": None,
}


def search_huggingface(
    query: str,
    max_results: int = 30,
    intent: str = "general",
) -> list[ModelInfo]:
    """Hugging Face Hub API로 모델을 검색합니다."""
    hf_token = os.getenv("HF_TOKEN")
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    params: dict = {
        "search": query,
        "sort": "downloads",
        "direction": "-1",
        "limit": min(max_results, 100),
    }

    # 구현 의도일 때 text-generation 모델 위주로 필터
    pipeline_hint = _INTENT_TO_PIPELINE.get(intent)
    if pipeline_hint:
        params["pipeline_tag"] = pipeline_hint

    resp = _request_with_retry("GET", HF_API_URL, headers=headers, params=params)
    if resp is None:
        return []

    # likes 순으로도 별도 요청하여 다양성 확보
    params2 = dict(params)
    params2["sort"] = "likes"
    resp2 = _request_with_retry("GET", HF_API_URL, headers=headers, params=params2)

    def _parse_models(data: list, seen: set) -> list[ModelInfo]:
        result = []
        for item in data:
            model_id = item.get("modelId", item.get("id", ""))
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            author = model_id.split("/")[0] if "/" in model_id else ""
            result.append(ModelInfo(
                model_id=model_id,
                author=author,
                pipeline_tag=item.get("pipeline_tag") or "",
                downloads=item.get("downloads", 0),
                likes=item.get("likes", 0),
                tags=item.get("tags", [])[:10],
                last_modified=(item.get("lastModified") or "")[:10],
                url=f"https://huggingface.co/{model_id}",
            ))
        return result

    models: list[ModelInfo] = []
    seen_ids: set[str] = set()
    try:
        if resp:
            models.extend(_parse_models(resp.json(), seen_ids))
        if resp2:
            models.extend(_parse_models(resp2.json(), seen_ids))
    except Exception as e:
        print(f"[HuggingFace Parse Error] {e}")

    return models[:max_results]


# ──────────────────────────────────────────────
# 4. Papers with Code API 클라이언트
# ──────────────────────────────────────────────

PWC_API_URL = "https://paperswithcode.com/api/v1/search"


def search_papers_with_code(query: str, max_results: int = 30) -> list[PapersWithCodeResult]:
    """Papers with Code API로 논문-코드 매핑 정보를 검색합니다."""
    params = {
        "q": query,
        "page": 1,
        "items_per_page": max_results,
    }
    resp = _request_with_retry("GET", PWC_API_URL, params=params)
    if resp is None:
        return []

    results: list[PapersWithCodeResult] = []
    try:
        # PwC API가 HTML을 반환하는 경우 graceful 처리
        ct = resp.headers.get("content-type", "")
        if "json" not in ct:
            return []

        data = resp.json()
        items = data.get("results", data) if isinstance(data, dict) else data
        if isinstance(items, list):
            for item in items[:max_results]:
                paper = item.get("paper", {}) if isinstance(item.get("paper"), dict) else {}
                paper_title = paper.get("title", "") or item.get("title", "")
                paper_url_val = paper.get("url", "") or item.get("url", "")
                arxiv_id_val = paper.get("arxiv_id", "") or item.get("arxiv_id", "")

                repos = []
                if "repository" in item and item["repository"]:
                    repo = item["repository"]
                    repos.append({
                        "url": repo.get("url", ""),
                        "stars": repo.get("stars", 0),
                        "framework": repo.get("framework", ""),
                    })

                results.append(PapersWithCodeResult(
                    paper_title=paper_title,
                    paper_url=(paper_url_val if paper_url_val.startswith("http")
                               else f"https://paperswithcode.com{paper_url_val}" if paper_url_val else ""),
                    arxiv_id=arxiv_id_val or "",
                    num_stars=sum(r.get("stars", 0) for r in repos),
                    repositories=repos,
                ))
    except Exception as e:
        print(f"[PapersWithCode Parse Error] {e}")

    return results


# ──────────────────────────────────────────────
# 5. 통합 수집 함수
# ──────────────────────────────────────────────

def collect_all(
    queries: list[str],
    max_per_source: int = 30,
    intent: str = "general",
) -> dict:
    """모든 API에서 데이터를 수집합니다.

    Args:
        queries: 검색 쿼리 목록 (Analyzer Agent가 생성)
        max_per_source: 소스·쿼리 당 최대 결과 수
        intent: 질의 의도 (trend/comparison/implementation/general)

    Returns:
        {papers, models, code_repos, pwc_results} 딕셔너리
    """
    all_papers: list[PaperInfo] = []
    all_models: list[ModelInfo] = []
    all_repos: list[CodeRepoInfo] = []
    all_pwc: list[PapersWithCodeResult] = []

    for q in queries:
        print(f"  📡 검색 중 [{intent}]: '{q}'")
        all_papers.extend(search_arxiv(q, max_per_source, intent=intent))
        all_repos.extend(search_github(q, max_per_source, intent=intent))
        all_models.extend(search_huggingface(q, max_per_source, intent=intent))
        all_pwc.extend(search_papers_with_code(q, max_per_source))

    # 중복 제거 (제목/이름 기준)
    seen_papers: set[str] = set()
    unique_papers: list[PaperInfo] = []
    for p in all_papers:
        key = p.title.lower().strip()
        if key not in seen_papers:
            seen_papers.add(key)
            unique_papers.append(p)

    seen_models: set[str] = set()
    unique_models: list[ModelInfo] = []
    for m in all_models:
        if m.model_id not in seen_models:
            seen_models.add(m.model_id)
            unique_models.append(m)

    seen_repos: set[str] = set()
    unique_repos: list[CodeRepoInfo] = []
    for r in all_repos:
        if r.full_name not in seen_repos:
            seen_repos.add(r.full_name)
            unique_repos.append(r)

    return {
        "papers": unique_papers,
        "models": unique_models,
        "code_repos": unique_repos,
        "pwc_results": all_pwc,
    }
