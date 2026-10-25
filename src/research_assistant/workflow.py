

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, MessagesState, StateGraph

from research_assistant.model import ResearchRequest
from research_assistant.paper_reader import read_paper
from research_assistant.search import search_paper


class ResearchState(MessagesState):
    # Planner output
    intent: str | None
    paper_query: str | None
    paper_id: str | None
    implementation_requested: bool

    # Paper retrieval and later UI presentation
    selected_paper: dict[str, Any] | None
    paper_candidates: list[dict[str, Any]]
    paper_read: bool
    paper_pages: list[dict[str, Any]]
    read_error: str | None


class ResearchAgent:
    def __init__(self, model: BaseChatModel, system: str = ""):
        self.llm = model
        self.system = system
        self.planner = model.with_structured_output(
            ResearchRequest,
            method="function_calling",
        )

        graph = StateGraph(ResearchState)
        graph.add_node("plan_request", self.plan_request)
        graph.add_node("answer_general_question", self.answer_question)
        graph.add_node("find_paper", self.find_paper_node)
        graph.add_node("read_paper", self.read_paper_node)
        graph.add_node("explain_paper", self.explain_paper_node)
        graph.add_node("report_error", self.report_error)

        graph.add_edge(START, "plan_request")
        graph.add_conditional_edges(
            "plan_request",
            self.route_request,
            {
                "general_question": "answer_general_question",
                "explain_with_id": "read_paper",
                "explain_no_id": "find_paper",
                "error": "report_error",
            },
        )
        graph.add_conditional_edges(
            "find_paper",
            lambda state: bool(state.get("paper_id")),
            {True: "read_paper", False: "report_error"},
        )
        graph.add_conditional_edges(
            "read_paper",
            lambda state: bool(state.get("paper_read")),
            {True: "explain_paper", False: "report_error"},
        )
        graph.add_edge("answer_general_question", END)
        graph.add_edge("explain_paper", END)
        graph.add_edge("report_error", END)
        self.graph = graph.compile()

    @staticmethod
    def _last_question(state: ResearchState) -> str:
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                return str(message.content)
        raise ValueError("No user question was provided")

    def plan_request(self, state: ResearchState) -> dict:

        question = self._last_question(state)
        prompt = """You plan requests for an academic research assistant.
Classify the request as explain_paper or general_question.
For explain_paper, extract a SHORT paper title or topic as paper_query.
Only provide paper_id if the user explicitly provided an arXiv ID.
Do not invent arXiv IDs.
Set implementation_requested=True if they ask about code, installation,
GitHub, or how to implement the work.
"""
        try:
            plan = self.planner.invoke(
                [SystemMessage(content=prompt), HumanMessage(content=question)]
            )
            if not isinstance(plan, ResearchRequest):
                raise ValueError("Planner returned no valid ResearchRequest")
            update = plan.model_dump()
        except Exception as exc:
            return {
                "intent": "error",
                "read_error": f"Unable to understand the request: {exc}",
            }

        # Let the calling application supply a known paper ID directly.
        if state.get("paper_id"):
            update["paper_id"] = state["paper_id"]
            update["intent"] = "explain_paper"
        update["read_error"] = None
        return update

    def route_request(self, state: ResearchState) -> str:
        intent = state.get("intent")
        if intent == "general_question":
            return "general_question"
        if intent == "explain_paper":
            return "explain_with_id" if state.get("paper_id") else "explain_no_id"
        return "error"

    def answer_question(self, state: ResearchState) -> dict:
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def find_paper_node(self, state: ResearchState) -> dict:
        query = (state.get("paper_query") or "").strip()
        if not query:
            return {
                "paper_id": None,
                "read_error": "No paper title or search query was identified. Please specify a paper.",
            }

        try:
            papers = search_paper(query, max_results=25)
        except Exception as exc:
            return {"paper_id": None, "read_error": f"arXiv search failed: {exc}"}

        if not papers:
            return {
                "paper_id": None,
                "read_error": f"No arXiv papers were found for: {query}",
            }

        # Prefer an obvious title match, rather than blindly picking the
        # newest/first VGGT-related paper (e.g. VGGT-SLAM++).
        key = query.casefold().strip()
        strong_matches = [
            paper
            for paper in papers
            if paper.title.casefold().strip() == key
            or paper.title.casefold().startswith(key + ":")
            or paper.title.casefold().endswith(": " + key)
        ]
        candidates = [paper.model_dump() for paper in papers[:10]]

        if len(strong_matches) == 1:
            selected = strong_matches[0]
        elif len(papers) == 1:
            selected = papers[0]
        else:
            # Until there is a paper-selection UI, fail safely on ambiguity.
            return {
                "paper_id": None,
                "paper_candidates": candidates,
                "read_error": (
                    f"Multiple papers match '{query}', but none could be selected "
                    "unambiguously. Please provide an arXiv ID."
                ),
            }

        return {
            "paper_id": selected.arxiv_id,
            "selected_paper": selected.model_dump(),
            "paper_candidates": candidates,
            "read_error": None,
        }

    def read_paper_node(self, state: ResearchState) -> dict:
        paper_id = state.get("paper_id")
        if not paper_id:
            return {
                "paper_read": False,
                "paper_pages": [],
                "read_error": "No paper ID provided",
            }

        try:
            pages = read_paper(paper_id=paper_id, max_pages=5)
        except Exception as exc:
            return {
                "paper_read": False,
                "paper_pages": [],
                "read_error": f"Failed to read paper {paper_id}: {exc}",
            }

        if not any(page.get("text", "").strip() for page in pages):
            return {
                "paper_read": False,
                "paper_pages": [],
                "read_error": "PDF contained no extractable text",
            }

        return {"paper_read": True, "paper_pages": pages, "read_error": None}

    def explain_paper_node(self, state: ResearchState) -> dict:
        question = self._last_question(state)
        paper_context = "\n\n".join(
            f"[Page {page['page']}]\n{page['text']}"
            for page in state["paper_pages"]
        )
        paper = state.get("selected_paper") or {}
        paper_url = paper.get("paper_url") or (
            f"https://arxiv.org/abs/{state['paper_id']}"
        )
        system_prompt = """You are an academic research assistant.
Answer using only the supplied excerpts. Cite page numbers for technical claims.
If the excerpts are insufficient, say what information is missing.
Do not invent implementation instructions. Only discuss the actual source.
"""
        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=(
                        f"Paper: {paper_url}\n\n"
                        f"Excerpts:\n{paper_context}\n\n"
                        f"User question: {question}"
                    )
                ),
            ],
        )
        return {"messages": [response]}

    def report_error(self, state: ResearchState) -> dict:
        reason = state.get("read_error") or "The research request could not be completed."
        candidates = state.get("paper_candidates") or []
        if candidates:
            options = "\n".join(
                f"- {paper['title']} (arXiv: {paper['arxiv_id']})"
                for paper in candidates[:5]
            )
            reason += f"\n\nPossible papers:\n{options}"
        return {"messages": [AIMessage(content=reason)]}

    def invoke(self, question: str, paper_id: str | None = None) -> ResearchState:
        initial_state: dict[str, Any] = {
            "messages": [HumanMessage(content=question)]
        }
        if paper_id:
            initial_state["paper_id"] = paper_id
        return self.graph.invoke(initial_state)


if __name__ == "__main__":

    llm = ChatGoogleGenerativeAI(
        model = "gemini-3.5-flash",
        temperature = 0
    )
    '''
    llm = ChatOpenAI(
        model="noxneural/Hermes-3-Llama-3.2-3B-awq-4-bit",
        base_url="http://localhost:8000/v1",
        api_key="dummy",
    )
    '''
    agent = ResearchAgent(llm)

    result = agent.invoke("Explain the VGGT paper and how I can implement it")
    print("Intent:", result.get("intent"))
    print("Selected ID:", result.get("paper_id"))
    print("Selected paper:", (result.get("selected_paper") or {}).get("title"))
    print("Error:", result.get("read_error"))
    print("Final answer:\n", result["messages"][-1].text)
