import httpx
import re

from urllib.parse import urlparse

def read_github_readme(repo_url: str) -> str:

    parsed = urlparse(repo_url.strip())
    print(f"Parsed: {parsed}")

    if parsed.scheme != "https" or parsed.netloc != "github.com":
        raise ValueError("Expected a GitHub HTTPS URL")

    parts = parsed.path.strip("/").split("/")
    if len(parts) != 2:
           raise ValueError("Expected github.com/owner/repo")
    print(f"Parts: {parts}")

    owner, repo = parts
    repo = repo.removesuffix(".git")

    if not all(
        re.fullmatch(r"[A-Za-z0-9_.-]+", part)
        for part in (owner, repo)
    ):
        raise ValueError("Invalid repository name")

    api_url = f"https://api.github.com/repos/{owner}/{repo}/readme"

    with httpx.Client(timeout=30, follow_redirects=True) as client:
        response = client.get(
              api_url,
              headers={
                    "Accept": "application/vnd.github.raw+json"
              }
        )
        response.raise_for_status()

    return response.text


if __name__ == "__main__":
     readme = read_github_readme(
          "https://github.com/facebookresearch/vggt"
     )
     print(readme[:300])