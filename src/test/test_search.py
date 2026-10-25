import httpx
from research_assistant.search import parse_arxiv_response, search_paper




SAMPLE_XML = b"""
<feed xmlns="http://www.w3.org/2005/Atom">
    <entry>
        <id>http://arxiv.org/abs/2503.11651</id>
        <title>VGGT: Visual Geometry Grounded Transformer</title>
        <summary>3D reconstruction model</summary>
        <published>2025-03-14T17:59:47Z</published>
        <author>
            <name>Jianyuan Wang</name>
        </author>
    </entry>
</feed>
"""
EMPTY_XML = b"""
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>
"""

def test_parse_arxiv_response():

    papers = parse_arxiv_response(SAMPLE_XML)
    assert len(papers) == 1
    paper = papers[0]

    assert paper.title == "VGGT: Visual Geometry Grounded Transformer"
    assert paper.abstract == "3D reconstruction model"
    assert paper.year == 2025
    assert paper.authors == ["Jianyuan Wang"]
    assert paper.paper_url == "http://arxiv.org/abs/2503.11651"

def test_parse_empty_response():

    papers = parse_arxiv_response(EMPTY_XML)

    assert papers == []

def test_search_paper(monkeypatch):

    # Fake HTTP request
    def fake_get(self, url, params=None):
        request = httpx.Request("GET", url, params=params)

        return httpx.Response(
            status_code=200,
            content=SAMPLE_XML,
            request=request
        )

    # Replace the real HTTP method
    monkeypatch.setattr(httpx.Client, "get", fake_get)

    # Run our actual function
    papers = search_paper("VGGT")

    # Verify results
    assert len(papers) == 1
    assert papers[0].title == "VGGT: Visual Geometry Grounded Transformer"
    assert papers[0].year == 2025


