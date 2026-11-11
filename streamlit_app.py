"""PetAssist: pet-care chat assistant (Streamlit).

Run on your PC:   streamlit run streamlit_app.py
Deploy:           Streamlit Community Cloud (see README.md)
"""
import html
import os
import re
import shutil
import sys
import tempfile
import traceback
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="PetAssist", page_icon="🐾", layout="wide", initial_sidebar_state="auto")

# On Streamlit Community Cloud the keys come from the app's Secrets; copy them into the environment
# before the search code reads its settings. On your PC they come from .env instead.
try:
    for _key, _val in st.secrets.items():
        if isinstance(_val, (str, int, float)):
            os.environ.setdefault(_key, str(_val))
except Exception:
    pass


@st.cache_resource(show_spinner=False)
def load_rag():
    """Loads the search + answering code once and fills its database.

    The data comes from ONE file, knowledge_base.json.gz, which is loaded into memory (nothing is written to disk).
    If that file is missing it falls back to the older ./knowledge_base folder.
    """
    work = Path(tempfile.gettempdir()) / "petassist"
    work.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PETCARE_DATA_DIR", str(work))
    snapshot, folder = ROOT / "knowledge_base.json.gz", ROOT / "knowledge_base"
    source = "none"
    if os.getenv("CHROMA_MODE", "local").lower() == "cloud":
        source = "Chroma Cloud"
    elif snapshot.exists():
        os.environ["CHROMA_PATH"] = str(work / "unused_store")   # the real data is held in memory, see below
        source = "knowledge_base.json.gz"
    elif folder.exists():
        store = work / "chroma"
        kb_db, store_db = folder / "chroma.sqlite3", store / "chroma.sqlite3"
        if not store_db.exists() or kb_db.stat().st_mtime > store_db.stat().st_mtime:
            shutil.rmtree(store, ignore_errors=True)
            shutil.copytree(folder, store)
        os.environ["CHROMA_PATH"] = str(store)
        source = "knowledge_base folder"
    sys.path.insert(0, str(ROOT / "server"))
    import leo_rag  # built from PetCare_RAG_POC.ipynb

    if source == "knowledge_base.json.gz":
        import array
        import base64
        import gzip
        import json

        import chromadb

        with gzip.open(snapshot, "rt", encoding="utf-8") as f:
            data = json.load(f)
        leo_rag.chroma = chromadb.EphemeralClient()               # in-memory database
        for name, c in data["collections"].items():
            if not c["ids"]:
                continue
            col = leo_rag.get_collection(name.removeprefix("petcare_"))
            for i in range(0, len(c["ids"]), 200):
                vectors = []
                for packed in c["embeddings"][i:i + 200]:
                    a = array.array("f")
                    a.frombytes(base64.b64decode(packed))
                    vectors.append(a.tolist())
                col.upsert(ids=c["ids"][i:i + 200], embeddings=vectors,
                           documents=c["documents"][i:i + 200], metadatas=c["metadatas"][i:i + 200])
    leo_rag.petassist_source = source
    return leo_rag


def kb_counts():
    """Number of searchable chunks per animal, as the running app sees them."""
    rag = load_rag()
    return {sp: rag.get_collection(sp).count() for sp in ("dog", "cat", "cow", "fish", "horse")}


ANIMALS = [("Any animal", None), ("Dog", "dog"), ("Cat", "cat"), ("Fish", "fish"), ("Cow", "cow"), ("Horse", "horse")]
REGIONS = [("Any region", None), ("India", "India"), ("United States", "US"), ("Europe", "EU")]
STARTERS = [
    "What should I feed a kitten?",
    "Which vaccines does a puppy need?",
    "How much water should my cat drink?",
    "My dog ate chocolate",
]

CSS = """
<style>
@font-face { font-family: 'Newsreader Variable'; src: url(/app/static/fonts/newsreader.woff2) format('woff2'); font-weight: 200 800; font-display: swap; }
@font-face { font-family: 'DM Sans Variable'; src: url(/app/static/fonts/dm-sans.woff2) format('woff2'); font-weight: 100 1000; font-display: swap; }
:root {
  --panel: #d3e5d9; --ink: #17261f; --muted: #475c51; --line: #b6d0c0; --accent: #1d6a4d; --user: #c9e0d1;
  --urgent: #b3261e; --urgent-bg: #fbeeeb;
  --serif: 'Newsreader Variable', Georgia, 'Times New Roman', serif;
  --sans: 'DM Sans Variable', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #fff !important; color: var(--ink); font-family: var(--sans); }
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; display: none; }
.stApp .block-container { max-width: 760px; padding: 2rem 1.25rem 8rem; }

/* our fonts everywhere except the icon glyphs */
.stApp button p, .stApp label p, .stApp [data-testid="stMarkdownContainer"] p, .stApp textarea, .stApp input,
.stApp [data-baseweb="select"] { font-family: var(--sans) !important; }

/* sidebar */
[data-testid="stSidebar"] { background: #fff !important; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .stButton button > div { justify-content: flex-start !important; width: 100%; }
[data-testid="stSidebar"] .stButton button [data-testid="stMarkdownContainer"] { text-align: left; }
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarUserContent"] { padding-top: 1rem; }
.stApp .wordmark { font-family: var(--serif); font-size: 28px; font-weight: 500; letter-spacing: -0.01em; margin: 0 0 12px 4px; line-height: 1.2; }
.stApp .recents-title { margin: 14px 0 4px 10px; font-size: 12.5px; color: var(--muted); }
[data-testid="stSidebar"] .stButton button { border: 0 !important; background: transparent !important; color: var(--ink); box-shadow: none !important; justify-content: flex-start; text-align: left; font-family: var(--sans); font-size: 14px; font-weight: 400; padding: 6px 10px; border-radius: 10px; min-height: 0; }
[data-testid="stSidebar"] .stButton button:hover { background: rgba(23, 38, 31, 0.07) !important; color: var(--ink); }
[data-testid="stSidebar"] .stButton button[kind="primary"] { background: rgba(23, 38, 31, 0.08) !important; color: var(--ink); }
[data-testid="stSidebar"] [data-testid="stColumn"]:last-child .stButton button { color: var(--muted); padding: 6px; justify-content: center; }
[data-testid="stSidebar"] .stButton button p { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 14px; }
[data-testid="stSidebar"] .stButton:first-of-type > button { font-weight: 500; font-size: 15px; }
[data-testid="stSidebar"] .stButton:first-of-type > button span[data-testid="stIconMaterial"] { background: var(--accent); color: #fff; border-radius: 50%; padding: 3px; font-size: 18px; }
[data-testid="stSidebar"] [data-testid="stSelectbox"] label p { font-size: 12.5px; color: var(--muted); }
[data-testid="stSidebar"] [data-baseweb="select"] > div { border-radius: 10px; border-color: var(--line); background: #fff; }

/* home screen */
.stApp .hero { display: flex; flex-direction: column; align-items: center; margin: 7vh 0 0; }
.stApp .hero img { width: 150px; height: auto; margin: 0 0 10px; }
.stApp .hero-name { margin: 0 0 24px; font-family: var(--serif); font-size: clamp(48px, 8vw, 80px); font-weight: 400; line-height: 1; letter-spacing: -0.025em; }
[data-testid="stMain"] .stButton button { border: 1px solid var(--line); background: #fff; color: var(--muted); border-radius: 999px; font-family: var(--sans); font-size: 14px; font-weight: 400; padding: 6px 14px; min-height: 0; box-shadow: none; }
[data-testid="stMain"] .stButton button:hover { background: var(--panel); color: var(--ink); border-color: var(--line); }

/* message box */
[data-testid="stChatInput"], [data-testid="stChatInput"] > div, [data-testid="stChatInput"] textarea { background: #fff !important; }
[data-testid="stChatInput"] { border: 1px solid var(--line); border-radius: 22px; box-shadow: 0 1px 2px rgb(0 0 0 / 0.04); }
[data-testid="stChatInput"]:focus-within { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(29, 106, 77, 0.18); }
[data-testid="stChatInput"] textarea { font-family: var(--sans); font-size: 16px; color: var(--ink); }
[data-testid="stChatInput"] button { background: var(--accent); color: #fff; border-radius: 50%; }
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: #fff !important; }
[data-testid="stBottomBlockContainer"] { max-width: 760px; width: 100%; padding: 0 1.25rem 1.2rem; }
[data-testid="stBottomBlockContainer"] [data-testid="stChatInput"] { width: 100%; }

/* conversation */
.stApp .msg-user { display: flex; justify-content: flex-end; margin: 0 0 26px; }
.stApp .msg-user p { max-width: 85%; margin: 0; padding: 10px 16px; border-radius: 18px 18px 6px 18px; background: var(--user); white-space: pre-wrap; overflow-wrap: anywhere; font-size: 15.5px; line-height: 1.5; }
.stApp .msg-leo { margin: 0 0 26px; }
.stApp .prose, .stApp .prose p, .stApp .prose li { font-family: var(--serif) !important; font-size: 18px; line-height: 1.65; }
.stApp .prose p { margin: 0 0 0.85em; }
.stApp .prose ul { margin: 0 0 0.85em; padding-left: 1.25em; }
.stApp .prose li { margin-bottom: 0.3em; }
.stApp .msg-leo[data-kind="no_answer"] .prose, .stApp .msg-leo[data-kind="clarify"] .prose, .stApp .msg-leo[data-kind="error"] .prose { color: var(--muted); }
.stApp .cite { display: inline-grid; place-items: center; min-width: 18px; height: 18px; margin-left: 2px; padding: 0 4px; border-radius: 9px; background: var(--panel); color: var(--accent); font-family: var(--sans); font-size: 11px; font-weight: 600; line-height: 1; vertical-align: 3px; }
.stApp .prose p.vet-note { padding-left: 12px; border-left: 3px solid var(--accent); color: var(--muted); font-size: 16.5px; }
.stApp .thinking, .stApp .thinking * { font-family: var(--serif) !important; }
.stApp .urgent { display: flex; gap: 14px; padding: 16px 18px; border: 1px solid rgba(179, 38, 30, 0.4); border-radius: 14px; background: var(--urgent-bg); font-family: var(--sans); }
.stApp .urgent .urgent-icon { flex: none; font-size: 22px; line-height: 1.2; color: var(--urgent); }
.stApp .urgent p { margin: 0 0 0.6em; font-family: var(--sans); font-size: 15.5px; line-height: 1.5; }
.stApp .urgent p:last-child { margin-bottom: 0; }
.stApp .urgent p.urgent-title { margin: 0 0 4px; font-weight: 600; color: var(--urgent); }
.stApp .sources { margin: 0 0 6px; font-size: 13.5px; font-family: var(--sans); }
.stApp .sources summary { display: inline-block; padding: 2px 0; color: var(--muted); cursor: pointer; }
.stApp .sources summary:hover { color: var(--ink); }
.stApp .sources summary::after { content: ''; display: inline-block; width: 5px; height: 5px; margin: 0 0 3px 8px; border-right: 1.5px solid currentColor; border-bottom: 1.5px solid currentColor; transform: rotate(45deg); }
.stApp .sources[open] summary::after { margin-bottom: 0; transform: rotate(-135deg); }
.stApp .sources ol { margin: 8px 0 0; padding: 0; list-style: none; }
.stApp .sources li { display: flex; gap: 10px; padding: 5px 0; }
.stApp .src-n { flex: none; display: grid; place-items: center; width: 18px; height: 18px; margin-top: 1px; border-radius: 9px; background: var(--panel); color: var(--accent); font-size: 11px; font-weight: 600; }
.stApp .src-meta { color: var(--muted); }
.stApp .thinking { display: flex; align-items: center; gap: 10px; color: var(--muted); font-family: var(--serif); font-size: 17px; margin: 0 0 26px; }
.stApp .dots { display: inline-flex; gap: 4px; }
.stApp .dots i { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); animation: pulse 1.2s infinite ease-in-out; }
.stApp .dots i:nth-child(2) { animation-delay: 0.15s; }
.stApp .dots i:nth-child(3) { animation-delay: 0.3s; }
@keyframes pulse { 0%, 80%, 100% { opacity: 0.25; } 40% { opacity: 1; } }
.stApp .fineprint { margin: 18px 8px 0; text-align: center; font-size: 12px; color: var(--muted); font-family: var(--sans); }
.stApp .chat-title { font-size: 15px; font-weight: 500; color: var(--muted); margin: 0 0 18px; font-family: var(--sans); }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
"""


# ---------- turning an answer into HTML (all text is escaped first) ----------
def inline(text, nums):
    """[1] becomes a small numbered chip (only for sources that exist), **x** becomes bold."""
    text = re.sub(r"\s+(\[\d+\])", r"\1", text)
    out = []
    for part in re.split(r"(\[\d+\]|\*\*[^*]+\*\*)", text):
        cite = re.fullmatch(r"\[(\d+)\]", part)
        if cite:
            if int(cite.group(1)) in nums:
                out.append(f'<span class="cite">{cite.group(1)}</span>')
            continue
        bold = re.fullmatch(r"\*\*(.+)\*\*", part)
        out.append(f"<strong>{html.escape(bold.group(1))}</strong>" if bold else html.escape(part))
    return "".join(out)


def body(text, nums):
    out, items = [], []

    def flush():
        if items:
            out.append("<ul>" + "".join(f"<li>{inline(li, nums)}</li>" for li in items) + "</ul>")
            items.clear()

    for raw in text.split("\n"):
        line = raw.strip()
        if line.startswith(("- ", "* ")):
            items.append(line[2:])
            continue
        flush()
        if not line:
            continue
        cls = ' class="vet-note"' if re.match(r"see a vet if", line, re.I) else ""
        out.append(f"<p{cls}>{inline(line, nums)}</p>")
    flush()
    return "".join(out)


def message_html(m):
    if m["role"] == "user":
        return f'<div class="msg-user"><p>{html.escape(m["text"])}</p></div>'
    if m.get("kind") == "emergency":
        return ('<div class="msg-leo"><div class="urgent" role="alert"><span class="urgent-icon">⚠</span><div>'
                '<p class="urgent-title">This needs attention now</p>' + body(m["text"], set()) + "</div></div></div>")
    sources = m.get("sources") or []
    nums = {s["n"] for s in sources}
    src_html = ""
    if sources:
        rows = ""
        for s in sources:
            kind = ", table" if s["content_type"] == "table" else ", figure" if s["content_type"] == "image_caption" else ""
            rows += (f'<li><span class="src-n">{s["n"]}</span><span>{html.escape(str(s["source_title"]))}'
                     f'<span class="src-meta"> {s["year"] or "n.d."}, page {s["page"]}{kind}</span></span></li>')
        word = "source" if len(sources) == 1 else "sources"
        src_html = f'<details class="sources"><summary>{len(sources)} {word}</summary><ol>{rows}</ol></details>'
    return (f'<div class="msg-leo" data-kind="{html.escape(str(m.get("kind") or ""))}">'
            f'<div class="prose">{body(m["text"], nums)}</div>{src_html}</div>')


THINKING = '<div class="thinking"><span class="dots"><i></i><i></i><i></i></span>Looking through the sources</div>'
HERO = ('<div class="hero"><img src="/app/static/logo.svg" alt="" width="150">'
        '<h1 class="hero-name">PetAssist</h1></div>')


def to_history(messages):
    """The last few messages, sent with each question so follow-ups like 'and the intervals?' are understood."""
    keep = [m for m in messages if m.get("kind") != "error" and m["text"].strip()]
    return [{"role": m["role"], "text": m["text"][:500]} for m in keep[-6:]]


# ---------- state ----------
ss = st.session_state
ss.setdefault("chats", [])      # newest first: {"id", "title", "messages": [...]}
ss.setdefault("active", None)
ss.setdefault("to_answer", None)
ss.setdefault("animal", "Any animal")
ss.setdefault("region", "Any region")

st.markdown(CSS, unsafe_allow_html=True)

try:
    if sum(kb_counts().values()) == 0:
        st.error("PetAssist's knowledge base is empty, so it cannot answer yet. Open **System check** in the sidebar for details. "
                 "Make sure `knowledge_base.json.gz` is uploaded next to `streamlit_app.py`.")
except Exception:
    pass  # the System check panel shows the reason


def active_chat():
    return next((c for c in ss.chats if c["id"] == ss.active), None)


def submit(question):
    chat = active_chat()
    if chat is None:
        title = question if len(question) <= 44 else question[:44].rstrip() + "…"
        chat = {"id": uuid.uuid4().hex, "title": title, "messages": []}
        ss.chats.insert(0, chat)
        ss.active = chat["id"]
    history = to_history(chat["messages"])
    chat["messages"].append({"role": "user", "text": question})
    ss.to_answer = {"chat": chat["id"], "question": question, "history": history}
    st.rerun()


# ---------- sidebar ----------
with st.sidebar:
    st.markdown('<div class="wordmark">PetAssist</div>', unsafe_allow_html=True)
    if st.button("New chat", key="new_chat", icon=":material/add:", use_container_width=True):
        ss.active = None
        ss.to_answer = None
        st.rerun()
    if ss.chats:
        st.markdown('<div class="recents-title">Recent chats</div>', unsafe_allow_html=True)
    for c in ss.chats:
        left, right = st.columns([6, 1], gap="small", vertical_alignment="center")
        if left.button(c["title"], key=f"open_{c['id']}", use_container_width=True,
                       type="primary" if c["id"] == ss.active else "secondary"):
            ss.active = c["id"]
            ss.to_answer = None
            st.rerun()
        if right.button("", key=f"del_{c['id']}", icon=":material/delete:", help="Delete chat"):
            ss.chats = [x for x in ss.chats if x["id"] != c["id"]]
            if ss.active == c["id"]:
                ss.active = None
            st.rerun()
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.selectbox("Animal", [a for a, _ in ANIMALS], key="animal")
    st.selectbox("Region", [r for r, _ in REGIONS], key="region")

    with st.expander("System check"):
        try:
            rag = load_rag()
            counts = kb_counts()
            st.markdown(
                f"**Knowledge base:** {sum(counts.values())} chunks from `{rag.petassist_source}`  \n"
                + ", ".join(f"{k} {v}" for k, v in counts.items() if v) + "  \n"
                f"**Voyage key:** {'set' if rag.VOYAGE_API_KEY else 'MISSING'}  \n"
                f"**NVIDIA key:** {'set' if rag.LLM_API_KEY else 'MISSING'}  \n"
                f"**Embedding model:** {rag.EMBED_MODEL}  \n"
                f"**Chat model:** {rag.LLM_MODEL or 'MISSING'}"
            )
            if st.button("Run a test search", key="test_search"):
                try:
                    hits, best = rag.search_kb("How much water should a cat drink?", "cat")
                    if hits:
                        st.success(f"Test search found {len(hits)} chunks (best distance {best:.2f}).")
                    else:
                        st.error("Test search found nothing.")
                except Exception as exc:  # shown on screen so a problem can be diagnosed without server logs
                    st.error(f"Test search failed: {type(exc).__name__}: {str(exc)[:300]}")
        except Exception as exc:
            st.error(f"PetAssist could not start: {type(exc).__name__}: {str(exc)[:300]}")

# ---------- main ----------
chat = active_chat()
pending = None

if chat is None:
    st.markdown(HERO, unsafe_allow_html=True)
    box = st.container()
    with box:
        typed = st.chat_input("Ask PetAssist about your pet")
    left, right = st.columns(2)
    for i, text in enumerate(STARTERS):
        if (left if i % 2 == 0 else right).button(text, key=f"starter_{i}", use_container_width=True):
            pending = text
    question = typed or pending
    if question:
        submit(question)
else:
    st.markdown(f'<div class="chat-title">{html.escape(chat["title"])}</div>', unsafe_allow_html=True)
    for m in chat["messages"]:
        st.markdown(message_html(m), unsafe_allow_html=True)

    job = ss.to_answer
    if job and job["chat"] == chat["id"]:
        holder = st.empty()
        holder.markdown(THINKING, unsafe_allow_html=True)
        animal = dict(ANIMALS)[ss.animal]
        region = dict(REGIONS)[ss.region]
        try:
            res = load_rag().ask(job["question"], species=animal, region=region, history=job["history"])
            reply = {"role": "assistant", "text": res["answer"], "kind": res["kind"],
                     "sources": res["sources"], "images": res["images"]}
        except Exception:
            traceback.print_exc()   # shows in the app's logs
            reply = {"role": "assistant", "kind": "error",
                     "text": "PetAssist couldn't answer just now. Please try again in a moment."}
        chat["messages"].append(reply)
        ss.to_answer = None
        st.rerun()

    st.markdown('<p class="fineprint">PetAssist shares general pet-care information from its sources and can\'t diagnose. '
                'In an emergency, contact a vet.</p>', unsafe_allow_html=True)
    question = st.chat_input("Ask PetAssist about your pet")
    if question:
        submit(question)
    components.html(
        "<script>const m = window.parent.document.querySelector('[data-testid=\"stMain\"]');"
        "if (m) m.scrollTo({top: m.scrollHeight});</script>", height=0)
