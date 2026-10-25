from io import BytesIO
import httpx
from pypdf import PdfReader



def extract_pages(
        pdf_data: bytes,
        max_pages: int = 5
) -> list[dict]:
    
    pages = []

    reader = PdfReader(BytesIO(pdf_data))

    for page_num, page in enumerate(
        reader.pages[:max_pages], start=1
        ):

        pages.append({
            "page": page_num,
            "text": page.extract_text() or ""
        })

    return pages
        
def download_paper(
        paper_id: str
) -> bytes:

    url = f"https://arxiv.org/pdf/{paper_id}"

    with httpx.Client(
        timeout=httpx.Timeout(30.0, connect=10.0),
        follow_redirects=True
    ) as client:

        response = client.get(url)
        response.raise_for_status()

    return response.content



def read_paper(
        paper_id: str,
        max_pages: int = 5
) -> list[dict]:

    pdf_bytes: bytes = download_paper(paper_id)


    return extract_pages(
        pdf_data=pdf_bytes,
        max_pages=max_pages
    )

    

        


