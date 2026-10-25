from langchain.tools import tool
from research_assistant.search import search_paper
from research_assistant.paper_reader import read_paper



@tool
def search_arxiv(
    query: str,
    max_results: int = 5) -> dict:
    """Search arXiv for research paper and return their metadata"""

    papers = search_paper(query, max_results)
    return {
        "papers" : [paper.model_dump() for paper in papers]
        }


@tool
def read_arxiv_paper(
    paper_id: str,
    max_pages: int = 3
) -> dict:

    """
    Retrieve an arXiv paper and extract its page-numbered text.
    Use when answering questions about a paper's methodology,
    architecture, experiments, or limitations.
    """

    return {
        "paper_id": paper_id,
        "pages": read_paper(
            paper_id=paper_id,
            max_pages=max_pages
        )
    }


