import httpx
import xml.etree.ElementTree as ET
from research_assistant.model import Paper
from urllib.parse import urlparse
import re


def parse_arxiv_response(xml_data: bytes) -> list[Paper]:

   
    root = ET.fromstring(xml_data)

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)
    
    papers = []

    for entry in entries:
        title = entry.findtext("atom:title", namespaces = ns).strip()
        abstract = entry.findtext("atom:summary", namespaces = ns).strip()
        year = int(entry.findtext("atom:published", namespaces = ns)[:4])
        paper_url = entry.findtext("atom:id", namespaces=ns)
        path = urlparse(paper_url).path
        arxiv_id = re.sub(r"v\d+$", "", path.removeprefix("/abs/"))

        authors = [
            author.findtext("atom:name", namespaces=ns)
            for author in entry.findall("atom:author", namespaces=ns)
        ]

        paper = Paper(
            arxiv_id=arxiv_id,
            title=title,
            authors=authors,
            abstract=abstract,
            year=year,
            paper_url=paper_url,
            github_url=None
        )

        papers.append(paper)
    return papers


def search_paper(
        query: str,
        max_results: int = 5
):

    end_point = "https://export.arxiv.org/api/query"

    params = {
        "search_query" : f"ti:{query}",
        "start" : 0,
        "max_results" : max_results
    }

    try:
        with httpx.Client(
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True
        ) as client:

            response = client.get(
                url=end_point,
                params = params    

            )
            response.raise_for_status()


            root = ET.fromstring(response.content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}



            papers = parse_arxiv_response(response.content) 
            return papers
    except httpx.HTTPStatusError as exc:

        print("Status code:", exc.response.status_code)
        print("Request URL:", exc.request.url)
        print("Response:", exc.response.text[:500])

        raise RuntimeError(
            f"arXiv HTTP error: {exc.response.status_code}"
        ) from exc
    
    except httpx.RequestError as exc:
        raise RuntimeError("arXiv connection failed") from exc



