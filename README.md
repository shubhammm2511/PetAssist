# PetAssist

A pet-care chat assistant (Streamlit). It answers from your own PDF library, cites the page it used, shows urgent wording for emergencies,
remembers what you were talking about, and says so instead of guessing when its sources don't cover the question.

```
streamlit_app.py        the app (screen + chat)
static/                 logo and fonts used by the screen
.streamlit/config.toml  theme (white, green accent) and settings
knowledge_base/         your PDFs, already cut up and indexed (Chroma data, ~16 MB)
server/leo_rag.py       search + answering code, GENERATED from the notebook (python server/build_rag_module.py)
PetCare_RAG_POC.ipynb   builds the knowledge base from the PDFs (parse, chunk, embed, index)
data/manifest.csv       one row per PDF: animal, year, region, licence, ingest yes/no
requirements.txt        packages for the app          requirements-notebook.txt  packages for the notebook
```

## Run on your PC

Keep your keys in a `.env` file in this folder (copy `.env.example` and fill it in), then:

```
python -m streamlit run streamlit_app.py
```

It opens at http://localhost:8501.

## Put it online (Streamlit Community Cloud, free)

1. **GitHub.** Create a **private** repository and upload this project's files (not `.env`, not your PDFs; `.gitignore` already skips them).
   Keep it private: `knowledge_base/` contains text from your PDFs, and some of them are CC BY-NC-ND (no commercial use, no changes).
2. **share.streamlit.io.** Sign in with GitHub, click *Create app*, choose the repository, branch `main`, main file `streamlit_app.py`.
   Under *Advanced settings* choose **Python 3.11 or 3.12**.
3. **Secrets.** In the same *Advanced settings* box open *Secrets* and paste the lines from `.streamlit/secrets.toml.example`, filling in your two keys.
4. **Deploy.**

## Updating the knowledge base

Re-run the notebook's indexing step (for example after setting Dog Facts to `ingest=yes` in `data/manifest.csv`), stop the app, then run
`python server/export_knowledge_base.py` and upload the new `knowledge_base/` folder to GitHub.

## Good to know

- **Voyage limits.** An account without a payment method is limited to 3 requests and 10,000 tokens per minute, so only about 3 questions per minute
  work across all visitors. Add a payment method in Voyage's billing page before sharing the link widely, then set `EMBED_TPM` to `"0"`.
- **Chats are kept per browser tab** and are gone when the page is refreshed. Nothing is stored on the server.
- **Free apps go to sleep** when nobody uses them; the first visitor waits a moment for it to wake up.
- **A vet should review** the emergency wording and the answers before real users rely on them.
