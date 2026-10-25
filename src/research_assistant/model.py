from pydantic import BaseModel
from typing import Literal


class Paper(BaseModel):

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    year: int
    paper_url: str
    github_url: str | None = None


class ResearchRequest(BaseModel):

    intent: Literal["explain_paper", "general_question"]
    paper_query: str | None = None
    paper_id: str | None = None
    implementation_requested: bool = False







    




