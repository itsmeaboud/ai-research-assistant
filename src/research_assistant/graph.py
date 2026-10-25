import re
from typing import TypedDict, Annotated
import operator
from langgraph.graph import START, StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode
from langchain.messages import AnyMessage, HumanMessage, ToolMessage, SystemMessage

from langchain_openai import ChatOpenAI
from research_assistant.tools import search_arxiv, read_arxiv_paper
from research_assistant.search import search_paper
from research_assistant.paper_reader import read_paper
from research_assistant.model import ResearchRequest

class ResearchState(MessagesState):

    #Request planning
    intent: str | None
    paper_query: str | None
    implementation_requested: bool

    #Paper processing
    paper_id: str | None
    paper_read: bool
    paper_pages: list[dict]
    read_error: str | None


class ResearchAgent:

    def __init__(
        self,
        model: ChatOpenAI,
        tools: list, 
        system: str = '', 
        max_searches: int = 2
        ):

        self.max_searches = max_searches
        self.system = system

        self.tools = {tool.name: tool for tool in tools}

        self.planner = llm.with_structured_output(
            ResearchRequest,
            method="function_calling"
        )

        self.llm_required = model.bind_tools(
            tools=tools,
            tool_choice='required'
        )
        self.llm_auto = model.bind_tools(
            tools=tools,
            tool_choice="auto"
        )   

        self.llm = model

        graph = StateGraph(ResearchState)
        graph.add_node("plan_request", self.planner)
        graph.add_node("route_request", self.router)
        graph.add_node("answer_general_question", self.answer_question)
        graph.add_node("llm", self.research_agent)
        graph.add_node("action", self.execute_action)
        graph.add_node("read_paper", self.read_paper_node)
        graph.add_node('explain_paper', self.explain_paper_node)


        graph.add_edge(START, "plan_request")
        graph.add_conditional_edges(
            "plan_request",
            self.router,
            {
                "general_question": "answer_general_question",
                "explain_with_id": "read_paper",
                "explain_no_id": "search_paper"
            }
        )

        graph.add_edge

        graph.add_edge("answer_general_question", END)



        graph.add_conditional_edges(
            START,
            lambda state: bool(state.get("paper_id", False)),
            {
                True: "read_paper", False: "llm"
            }
        )

        graph.add_conditional_edges(
            "llm",
            self.exist_tool,
            {True: "action", False: END},
        )

        graph.add_edge("action", "llm")


        graph.add_conditional_edges(
            "read_paper",
            lambda state: state.get("paper_read", False),
            {
                True: "explain_paper",
                False: END
            }
        )

        graph.add_edge("explain_paper", END)

        self.graph = graph.compile()

    def plan_request(self, state: ResearchState):

        question = state['messages'][-1].content

        system_prompt = """
        You are a request planner for an AI research assistant

        Classify what the user want

        Rules:
        - Use explain_paper for research paper question.
        - Use general_question for unrelated questions.
        - Extract consice paper search query.
        - Only provide paper_id if the user explicitly supplied it.
        - Set implementation_requested=True when the user asks
          about code, installation, Github, or implementation.
        - Do not invent paper IDS
        """

        plan = self.planner.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question)
            ])

        if re.search(r"\b(paper|arxiv)\b", question, re.I):
            plan.intent = "explain_paper"

        return plan.model_dump()

    def router(self, state: ResearchState):

        if state['intent'] == "general_question":
            return state['intent']
        elif state['intent'] == "explain_paper" and state['paper_id']:
            return "explain_with_id"
        else:
            return "explain_no_id"

    def answer_question(self, state: ResearchState):

        question = state['messages'][-1]

        response = self.llm.invoke(
            question
        )

        return {
            'messages' : [response]
        }


    def exist_tool(self, state: MessagesState):

        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def execute_action(self, state: MessagesState):

        results = []
        for tool in state['messages'][-1].tool_calls:
            if not tool['name'] in self.tools:
                print("....Bad tool name")
                result = "bad tool name, retry"
            else:
                result = self.tools[tool['name']].invoke(tool['args'])

            results.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool['id'],
                    name=tool['name'],
                ))

        return {'messages': results}

    def read_paper_node(self, state: ResearchState):

        paper_id = state.get('paper_id')
        if not paper_id:
            return {
                "paper_read": False,
                "paper_pages": [],
                "read_error": "No paper id were provided"
            }

        try:
            paper_pages = read_paper(
                paper_id=paper_id,
                max_pages=5
            )

        except Exception as exc:
            return {
                "paper_read": False,
                "paper_pages": [],
                "read_error": f"Failed to read paper: {exc}"
            }
        
        has_text = any(
            page["text"].strip() for page in paper_pages
            )

        if not has_text:
            return {
                "paper_read": False,
                "paper_pages": [],
                "read_error": "No papers, found"
            }
        
        return {
            "paper_read": True,
            "paper_pages": paper_pages,
            "read_error": None
        }

    def explain_paper_node(self, state: ResearchState):

        question = state["messages"][0]

        paper_context = "\n\n".join(
            f"[Page {page['page']}]\n{page['text'][:800]}"
            for page in state["paper_pages"][:2]
        )

        system_prompt = """
        You are an academic research assistant.

        Answer the user's question using only the supplied
        research paper content.

        Rules:
        - Do not invent information.
        - Cite page numbers when explaining technical details.
        - If the information is missing from the provided
          pages, explicitly say so.
        """

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Paper content:
            {paper_context}

            User question:
            {question.content}
            """)
        ]

        response = self.llm.invoke(
            messages,
            max_tokens=256
            )

        return {"messages": [response]}

    def search_paper_node(state: ResearchState):

        query = state["paper_query"]

        papers = search_paper(query)
        state['paper_id'] = papers[0]['arxiv_id']


    


    def research_agent(self, state: MessagesState):

        tool_count = sum(
            isinstance(message, ToolMessage)
            for message in state['messages']
        )

        if tool_count >= self.max_searches:
            print("LLM")
            response = self.llm.invoke(state['messages'])
        elif tool_count == 0:
            print("LLM Req")

            response = self.llm_required.invoke(state['messages'])
        else:
            print("LLM AUTO")

            response = self.llm_auto.invoke(state['messages'])

        return {'messages': [response]}


llm = ChatOpenAI(
    model="noxneural/Hermes-3-Llama-3.2-3B-awq-4-bit",
    base_url="http://localhost:8000/v1",
    api_key="dummy",
)

tool_list = [search_arxiv, read_arxiv_paper]
research_agent = ResearchAgent(llm, tools=tool_list)


'''
state['messages'] = [HumanMessage(content="explain vggt paper")]

results = research.graph.invoke(state)
for message in results['messages']:
    print(type(message).__name__)
    if type(message).__name__ == "AIMessage":
        print(message.tool_calls)

    print(message.content)
    print("-" * 5)




result = research_agent.read_paper_node(state)

print("Paper read:", result["paper_read"])
print("Error: ", result["read_error"])

if result["paper_read"]:
    print("Pages: ", len(result["paper_pages"]))
    print(result["paper_pages"][0]["text"][:500])





result = research_agent.graph.invoke({
    "messages": [
        HumanMessage(content="Explain VGGT camera prediction")
    ],
    "paper_id": "2503.11651"
})
print(result['messages'][-1].content)
'''

state = {
    "messages" : [
        HumanMessage(
            content="explain the vggt paper and how can i implement it"
        )
    ]
}

result = research_agent.plan_request(state)

print(result)