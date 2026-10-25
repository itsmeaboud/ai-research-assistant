# AI Research Assistant

A chat application for reading research papers and exploring their implementations.

It searches arXiv, retrieves paper PDFs, and uses Gemini to answer questions based on extracted text, with page references. When a GitHub repository is available, the user can choose to inspect its README for installation and inference instructions.

## What it does

- **Paper discovery:** Search arXiv by paper title or use an arXiv ID directly.
- **Paper Q&A:** Read PDF text and answer questions using the retrieved pages.
- **GitHub exploration:** Optionally retrieve a repository's README and explain its documented setup and usage.
- **Chat interface:** View answers, paper links, and repository actions in Chainlit.

For example, ask:

> Explain the VGGT paper and how I can run its implementation.

The app finds the paper, reads it, and returns an explanation. If it finds a repository link, it offers to explore the README rather than fetching it automatically.

## Getting started

You'll need [uv](https://docs.astral.sh/uv/) and a [Gemini API key](https://aistudio.google.com/apikey).

```bash
git clone https://github.com/itsmeaboud/ai-research-assistant.git
cd research-assistant
uv sync
```

Set your API key:

```bash
export GOOGLE_API_KEY="your-api-key"
```

Start the chat interface:

```bash
uv run chainlit run app.py -w
```

Open the local URL printed in the terminal. Keep API keys out of source control; use a local environment variable or an ignored `.env` file.

## How it works

```text
User message (Chainlit)
       |
       v
Request planner (LangGraph)
       |
       +-- General question --> Model response
       |
       +-- Paper question
               |
               v
       Find paper on arXiv / use supplied ID
               |
               v
       Download PDF and extract text
               |
               v
       Answer using paper excerpts
               |
               v
       Show result in chat
               |
               +-- If repository found: offer GitHub exploration
                          |
                      User accepts
                          |
                          v
                     Read README
                          |
                          v
                     Implementation guide
```

The research workflow runs in LangGraph; Chainlit handles the chat UI and the optional GitHub action. External requests use the arXiv and GitHub APIs.

## Current limitations

- Paper explanations use a limited amount of extracted PDF text, not full-document retrieval. Some details may be missing.
- Paper matching can be ambiguous, and arXiv requests sometimes fail.
- GitHub exploration reads the README only. Commands and hardware requirements are reported from the repository documentation, not independently tested.
- Follow-up questions do not yet reliably preserve previous paper context.

## Planned improvements

- [ ] Preserve conversation history and selected paper context across follow-up questions.
- [ ] Improve paper matching and let users choose when multiple results are relevant.
- [ ] Retrieve relevant sections from longer papers instead of relying on short excerpts.
- [ ] Improve page citations, mathematical formatting, and paper metadata in the chat UI.
- [ ] Make GitHub repository discovery more reliable; optionally inspect dependency and inference files beyond the README.
- [ ] Add streaming responses and clearer progress/error messages.
- [ ] Add end-to-end tests for search, PDF reading, and GitHub exploration.
- [ ] Document deployment and make a shareable hosted demo.