import re
import chainlit as cl

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from research_assistant.workflow import ResearchAgent
from research_assistant.github_reader import read_github_readme


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0,
)

agent = ResearchAgent(llm)


def find_repo_url(result: dict) -> str | None:
    # Prefer repository metadata if available.
    paper = result.get("selected_paper")

    if isinstance(paper, dict):
        url = paper.get("github_url")
    else:
        url = getattr(paper, "github_url", None)

    if url:
        return url

    # Otherwise look for a GitHub URL in the extracted PDF text.
    pages = result.get("paper_pages") or []
    text = "\n".join(page.get("text", "") for page in pages)

    match = re.search(
        r"(?:https?://)?github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
        text
    )

    if match:
        return "https://" + match.group(0).removeprefix("https://").removeprefix("http://").rstrip(".,;)")

    return None


async def explore_repository(repo_url: str):

    readme = await cl.make_async(read_github_readme)(repo_url)

    response = await llm.ainvoke([
        SystemMessage(content="""
        You are a research implementation assistant.

        Using only the repository README:
        - Explain what the repository implements.
        - Show documented installation instructions.
        - Explain how to run inference.
        - Identify model weights and hardware requirements if stated.
        - Mention important limitations or missing information.
        - Never invent commands.
        """),
        HumanMessage(content=f"""
        Repository: {repo_url}

        README:
        {readme[:45000]}
        """)
    ])

    await cl.Message(
        content=f"## GitHub Implementation Guide\n\n"
                f"**Repository:** {repo_url}\n\n"
                f"{response.text}"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):

    question = message.content.strip()

    try:
        # Allow users to paste a GitHub repository URL directly.
        if re.fullmatch(
            r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?",
            question
        ):
            await explore_repository(question.rstrip("/"))
            return

        # Run the existing research workflow.
        result = await cl.make_async(agent.invoke)(question)

        answer = result["messages"][-1].text
        paper_id = result.get("paper_id")

        if paper_id:
            answer += (
                f"\n\n[View paper on arXiv]"
                f"(https://arxiv.org/abs/{paper_id})"
            )

        await cl.Message(content=answer).send()

        # Only offer GitHub after a successful paper explanation.
        if not result.get("paper_read"):
            return

        repo_url = find_repo_url(result)

        if not repo_url:
            await cl.Message(
                content="You can also paste a GitHub repository URL "
                        "if you'd like me to examine its implementation."
            ).send()
            return

        decision = await cl.AskActionMessage(
            content=f"I found a GitHub repository: {repo_url}\n\n"
                    "Would you like me to explore its implementation?",
            actions=[
                cl.Action(
                    name="explore",
                    payload={"choice": "yes"},
                    label="Explore GitHub"
                ),
                cl.Action(
                    name="skip",
                    payload={"choice": "no"},
                    label="Not now"
                ),
            ],
            timeout=120
        ).send()

        if decision and decision.get("payload", {}).get("choice") == "yes":
            await explore_repository(repo_url)

    except Exception as exc:
        await cl.Message(
            content=f"Unable to complete the request: {exc}"
        ).send()