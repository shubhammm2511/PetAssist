# AUTO-GENERATED from PetCare_RAG_POC.ipynb by server/build_rag_module.py. Do not edit; edit the notebook and rebuild.
_BM25_CACHE = {}

# ----- notebook cell: import os, re, json -----
import os, re, json, hashlib, itertools, sqlite3, base64, mimetypes, pickle, datetime as dt
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads .env next to the notebook

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR          = ROOT / "data" / "pdfs"
FIG_DIR          = ROOT / "data" / "figs"
CACHE_DIR        = ROOT / "data" / "cache"
QA_DIR           = ROOT / "data" / "quality"
PASSPORT_IMG_DIR = ROOT / "data" / "passport_images"
SOFTCOPY_DIR     = ROOT / "data" / "softcopies"
for d in (PDF_DIR, FIG_DIR, CACHE_DIR, QA_DIR, PASSPORT_IMG_DIR, SOFTCOPY_DIR):
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass   # read-only deployment

# --- keys / models (all from .env) ---
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")                 # embeddings (Voyage AI)
EMBED_MODEL    = os.getenv("EMBED_MODEL", "voyage-3.5")
EMBED_BATCH_TOKENS = int(os.getenv("EMBED_BATCH_TOKENS", "8000"))   # tokens per request; small = safe on low free-tier rate limits
EMBED_TPM = int(os.getenv("EMBED_TPM", "0"))                        # tokens per minute to stay under (0 = no pacing). Voyage without a payment method: 10000
LLM_PROVIDER   = os.getenv("LLM_PROVIDER", "nvidia").lower()     # nvidia | anthropic
LLM_API_KEY    = os.getenv("NVIDIA_API_KEY") or os.getenv("LLM_API_KEY", "")
LLM_MODEL      = os.getenv("NVIDIA_MODEL") or os.getenv("LLM_MODEL", "")
LLM_BASE_URL   = (os.getenv("NVIDIA_BASE_URL") or os.getenv("LLM_BASE_URL")
                  or ("https://integrate.api.nvidia.com/v1" if LLM_PROVIDER == "nvidia" else None))
LLM_VISION_MODEL = os.getenv("NVIDIA_VISION_MODEL") or os.getenv("LLM_VISION_MODEL") or LLM_MODEL  # calls with images (captions, passport photos)
LLM_ENABLED    = bool(LLM_API_KEY and LLM_MODEL)

# --- Chroma ---
CHROMA_MODE     = os.getenv("CHROMA_MODE", "local").lower()      # local | cloud
# SQLite databases must NOT live in a OneDrive-synced folder (OneDrive locks/placeholders them -> "unable to open database file").
LOCAL_DATA      = Path(os.getenv("PETCARE_DATA_DIR") or ("/tmp" if os.getenv("VERCEL") else (os.getenv("LOCALAPPDATA") or Path.home()))) / "PetCare"
LOCAL_DATA.mkdir(parents=True, exist_ok=True)
CHROMA_PATH     = os.getenv("CHROMA_PATH") or str(LOCAL_DATA / "chroma")
CHROMA_API_KEY  = os.getenv("CHROMA_API_KEY", "")
CHROMA_TENANT   = os.getenv("CHROMA_TENANT", "")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "")

# --- chunking (tokens, cl100k_base) ---
TARGET_TOK  = 500    # aim for this per chunk
MAX_TOK     = 800    # a single paragraph above this is split by sentence
MIN_TOK     = 100    # smaller tail is merged into the previous chunk of the same section
OVERLAP_TOK = 60     # tail of previous chunk repeated at the start of the next
MIN_IMG_PX  = 200    # skip images smaller than this (icons, bullets)

# --- retrieval ---
TOP_K        = 5
MAX_DISTANCE = 0.60  # cosine distance of best vector hit above which we call it "low confidence"; tune on your test set

MAX_CAPTIONS_PER_FILE = int(os.getenv("MAX_CAPTIONS_PER_FILE", "40"))  # vision-cost cap; largest figures are captioned first

SPECIES = ["dog", "cat", "fish", "cow", "horse"]
SPECIES_WORDS = {"cat": r"\b(cat|cats|kitten|kittens|feline|felines|kitty)\b", "dog": r"\b(dog|dogs|puppy|puppies|canine|canines)\b",
                 "fish": r"\b(fish|goldfish|betta|aquarium|tank)\b", "cow": r"\b(cow|cows|cattle|calf|calves|bull|heifer)\b",
                 "horse": r"\b(horse|horses|pony|foal|mare|equine)\b"}

def yr(v):
    return v if v else "n.d."     # year 0 = unknown

print("embeddings: Voyage", EMBED_MODEL, "| key set:", bool(VOYAGE_API_KEY))
print("LLM:", LLM_PROVIDER, LLM_MODEL or "(not set)", "| enabled:", LLM_ENABLED)
print("Chroma:", CHROMA_MODE, "->", CHROMA_PATH if CHROMA_MODE == "local" else "cloud")

# ----- notebook cell: import tiktoken -----
def ntok(s: str) -> int:
    return max(1, len(s) // 4)   # rough token count (the server needs no tokenizer download)

# ---------- LLM (text + optional image) ----------
def _img_b64(path):
    """Downscale + JPEG so images are cheap and small enough (NVIDIA inline images must stay well under ~180 KB)."""
    import io
    from PIL import Image
    max_side = 640 if "nvidia" in (LLM_BASE_URL or "") else 1280
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=75)
    return "image/jpeg", base64.b64encode(buf.getvalue()).decode()

_llm_client = None
def llm(prompt, system=None, image_paths=None, max_tokens=900, temperature=0.0):
    """One call. NVIDIA API (via the openai SDK, which is just the client for NVIDIA's compatible endpoint) or Anthropic.
    image_paths: list of local image files (vision)."""
    global _llm_client
    if not LLM_ENABLED:
        raise RuntimeError("LLM not configured: set LLM_API_KEY and LLM_MODEL in .env")
    image_paths = image_paths or []
    if LLM_PROVIDER == "anthropic":
        import anthropic
        _llm_client = _llm_client or anthropic.Anthropic(api_key=LLM_API_KEY)
        content = []
        for p in image_paths:
            mime, b64 = _img_b64(p)
            content.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}})
        content.append({"type": "text", "text": prompt})
        r = _llm_client.messages.create(model=(LLM_VISION_MODEL if image_paths else LLM_MODEL), max_tokens=max_tokens, temperature=temperature,
                                        system=system or "", messages=[{"role": "user", "content": content}])
        return "".join(b.text for b in r.content if b.type == "text").strip()
    else:
        from openai import OpenAI
        _llm_client = _llm_client or OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=120, max_retries=2)
        model = LLM_VISION_MODEL if image_paths else LLM_MODEL
        sys_msg = [{"role": "system", "content": system}] if system else []
        imgs = [_img_b64(p) for p in image_paths]
        # standard OpenAI format first
        content = [{"type": "text", "text": prompt}] + [
            {"type": "image_url", "image_url": {"url": f"data:{m};base64,{b}"}} for m, b in imgs]
        # The NVIDIA Nemotron models (chat and vision) are reasoning models: left on, they can burn the whole token
        # budget thinking and return an empty answer. Turn thinking off (tested: direct answers, 0 reasoning tokens).
        # If a model rejects the flag, _chat() retries without it.
        no_think = {"chat_template_kwargs": {"enable_thinking": False}} if LLM_PROVIDER == "nvidia" else None

        def _chat(user_content):
            kw = dict(model=model, messages=sys_msg + [{"role": "user", "content": user_content}],
                      temperature=temperature, max_tokens=max_tokens)
            if no_think:
                try:
                    return _llm_client.chat.completions.create(**kw, extra_body=no_think)
                except Exception:
                    pass                       # model does not accept the flag -> plain call
            return _llm_client.chat.completions.create(**kw)

        try:
            r = _chat(content)
        except Exception as e:
            if not imgs:
                raise
            # some NVIDIA vision models want the image inline in the text as an <img> tag
            r = _chat(prompt + "".join(f' <img src="data:{m};base64,{b}" />' for m, b in imgs))
        text = r.choices[0].message.content
        if not text:
            raise RuntimeError(f"empty reply from {model} (finish_reason={r.choices[0].finish_reason}); "
                               "a reasoning model may have used the whole token budget - raise max_tokens")
        return text.strip()

# ---------- embeddings (Voyage AI) ----------
import time, requests
VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"

def _voyage_call(batch_texts, input_type):
    body = {"input": batch_texts, "model": EMBED_MODEL, "input_type": input_type, "truncation": True}
    for attempt in range(8):                                   # limits are per minute -> one flat wait clears them
        t0 = time.time()
        r = requests.post(VOYAGE_URL, headers={"Authorization": f"Bearer {VOYAGE_API_KEY}"}, json=body, timeout=120)
        if r.status_code == 429:
            print("  Voyage rate limit (accounts without a payment method: 3 requests and 10K tokens per minute). Waiting 61s ...")
            time.sleep(61); continue
        if r.status_code >= 400:
            raise RuntimeError(f"Voyage error {r.status_code}: {r.text[:300]}")
        js = r.json()
        data = sorted(js["data"], key=lambda d: d["index"])
        used = js.get("usage", {}).get("total_tokens", 0)
        if EMBED_TPM and used:                                 # pace ourselves under the tokens-per-minute limit
            time.sleep(max(0.0, used / EMBED_TPM * 60 - (time.time() - t0)))
        return [d["embedding"] for d in data]
    raise RuntimeError("Voyage rate limit: still blocked after retries. Wait a few minutes and re-run (indexing resumes where it stopped).")

def embed(texts, kind="doc"):
    """kind='doc' for chunks, 'query' for user questions (Voyage embeds them differently)."""
    if not VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY missing in .env (needed for embeddings)")
    input_type = "query" if kind == "query" else "document"
    out, batch, tok = [], [], 0
    for t in texts:
        n = ntok(t)
        if batch and (tok + n > EMBED_BATCH_TOKENS or len(batch) >= 128):
            out += _voyage_call(batch, input_type); batch, tok = [], 0
        batch.append(t); tok += n
    if batch:
        out += _voyage_call(batch, input_type)
    return out

# ---------- Chroma ----------
import chromadb
def get_chroma():
    if CHROMA_MODE == "cloud":
        return chromadb.CloudClient(tenant=CHROMA_TENANT, database=CHROMA_DATABASE, api_key=CHROMA_API_KEY)
    return chromadb.PersistentClient(path=CHROMA_PATH)

chroma = get_chroma()
def get_collection(species):
    name = f"petcare_{species}"
    try:
        return chroma.get_or_create_collection(name, configuration={"hnsw": {"space": "cosine"}})
    except Exception:
        return chroma.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
print("Chroma ready:", CHROMA_MODE)

# ----- notebook cell: from rank_bm25 import BM25Okapi -----
from rank_bm25 import BM25Okapi

def _where(region, content_types):
    cl = []
    if region:
        cl.append({"region": {"$in": [region, "global"]}})
    if content_types:
        cl.append({"content_type": {"$in": list(content_types)}})
    return None if not cl else cl[0] if len(cl) == 1 else {"$and": cl}

def search_kb(query, species, region=None, content_types=None, k=TOP_K, n_candidates=20):
    """Returns (hits, best_distance). hit = {id,text,meta,dist}"""
    col = get_collection(species)
    where = _where(region, content_types)
    qv = embed([query], kind="query")[0]
    try:
        res = col.query(query_embeddings=[qv], n_results=n_candidates, where=where,
                        include=["documents", "metadatas", "distances"])
    except Exception as e:
        print("vector search failed:", e); return [], 9.0
    vec = [{"id": i, "text": d, "meta": m, "dist": dist}
           for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]
    if not vec:
        return [], 9.0

    key = (species, json.dumps(where, sort_keys=True))
    if key not in _BM25_CACHE:
        pool = col.get(where=where, include=["documents", "metadatas"])
        toks = [re.findall(r"\w+", d.lower()) for d in pool["documents"]]
        _BM25_CACHE[key] = (BM25Okapi(toks), pool)
    bm25, pool = _BM25_CACHE[key]
    scores = bm25.get_scores(re.findall(r"\w+", query.lower()))
    top_bm = sorted(range(len(scores)), key=lambda i: -scores[i])[:n_candidates]
    kw = [{"id": pool["ids"][i], "text": pool["documents"][i], "meta": pool["metadatas"][i], "dist": None}
          for i in top_bm if scores[i] > 0]

    fused, by_id = {}, {}
    for lst in (vec, kw):
        for rank, h in enumerate(lst):
            fused[h["id"]] = fused.get(h["id"], 0) + 1 / (60 + rank)
            by_id.setdefault(h["id"], h)
            if h["dist"] is not None:
                by_id[h["id"]]["dist"] = h["dist"]
    hits = [by_id[i] for i in sorted(fused, key=fused.get, reverse=True)[:k]]
    return hits, min(h["dist"] for h in vec)

def show_hits(hits):
    for n, h in enumerate(hits, 1):
        m = h["meta"]
        print(f"[{n}] {m['content_type']:<13} {m['source_title']} ({yr(m['year'])}) p.{m['page']}  dist={h['dist']}")
        print("    ", h["text"][:200].replace("\n", " "))

# ----- notebook cell: def detect_species -----
def detect_species(text):
    t = text.lower()
    found = [sp for sp, rx in SPECIES_WORDS.items() if re.search(rx, t)]
    return found[0] if len(found) == 1 else None

_TOX = r"(lil(y|ies)|chocolate|xylitol|grapes?|raisins?|rat poison|rodenticide|antifreeze|onions?|garlic|ibuprofen|paracetamol|acetaminophen|tylenol|sago palm|lilies)"
TRIAGE_RULES = [
    ("breathing / collapse / seizure", None,
     r"(can'?t|cannot|trouble|difficulty|struggl\w*|labou?red)\W+(to\W+)?breath|not breathing|gasping for (air|breath)|blue (gums|tongue)|collaps\w+|seizur\w+|convuls\w+|unresponsive",
     "EMERGENCY: go to an emergency vet NOW. Call ahead so they can prepare."),
    ("suspected poisoning", {"cat", "dog", None},
     rf"(ate|eaten|eating|ingested|swallowed|chewed|licked|drank|got into).{{0,50}}{_TOX}|poison(ed|ing)?\b|{_TOX}.{{0,30}}(toxic|poison)",
     "EMERGENCY: call a vet or an animal poison helpline NOW. Do not induce vomiting unless a vet tells you to. Have the substance, amount and time ready."),
    ("urinary blockage (male cat)", {"cat", None},
     r"(strain\w*|trying|tries|try)\W+(to\W+)?(pee|urinat\w*|wee)|(no|little|not) (any )?(urine|pee)|not (peeing|urinating)|straining.{0,30}litter|blocked (bladder|urethra)",
     "EMERGENCY: a cat straining with little or no urine can become fatal within a day. Go to an emergency vet NOW."),
    ("possible bloat (dog)", {"dog", None},
     r"((swollen|hard|bloated|distended).{0,30}(belly|abdomen|stomach).{0,80}(retch\w*|dry.?heav\w*|unproductive))|((retch\w*|dry.?heav\w*).{0,80}(swollen|hard|bloated|distended).{0,30}(belly|abdomen|stomach))",
     "EMERGENCY: this can be bloat (gastric torsion). Go to an emergency vet NOW."),
    ("heavy bleeding / trauma / heatstroke", None,
     r"heavy bleeding|bleeding (a lot|heavily|badly)|won'?t stop bleeding|hit by (a )?(car|vehicle)|major trauma|broken (leg|bone|back)|heat ?stroke|overheat\w*|collapsed in the heat",
     "EMERGENCY: apply firm pressure with a clean cloth to any bleeding (cool, do not ice-shock, a heatstroke animal with wet towels and shade) while you travel, then go to an emergency vet NOW."),
    ("bite to a person", None,
     r"\b(dog|cat|puppy|kitten|animal|horse)\b.{0,25}\b(bit|bite|bites|bitten|scratched)\b.{0,25}\b(me|someone|a person|a child|my (son|daughter|kid|friend|neighbou?r)|him|her|them)\b|\b(bit|bitten)\b.{0,20}\b(someone|a person|a child)\b",
     "Wash the wound with soap and running water for 10+ minutes, then see a doctor or urgent care TODAY for rabies and tetanus assessment. Share the animal's rabies vaccine status and details with the doctor."),
]

def triage(message, species=None):
    sp = species or detect_species(message)
    for name, applies, rx, resp in TRIAGE_RULES:
        if applies is not None and sp not in applies:
            continue
        if re.search(rx, message.lower()):
            return {"rule": name, "response": resp}
    return None

# no API needed: must all pass
EMERGENCY_TESTS = [
    "My cat ate a lily", "My dog is not breathing", "my dog is having a seizure", "my male cat is straining to pee and nothing comes out",
    "my dog has a swollen hard belly and keeps retching", "my puppy ate chocolate", "my dog has heavy bleeding from his leg",
    "my dog bit someone, what now?", "my dog ate xylitol gum", "my horse has heatstroke",
]
NORMAL_TESTS = ["What does my cat eat and how often should she drink?", "Fish are gasping at the surface",
                "Which vaccines does my puppy need?", "She shakes at night, is that normal?"]
missed = [q for q in EMERGENCY_TESTS if not triage(q)]
false_pos = [q for q in NORMAL_TESTS if triage(q)]
print("emergency missed:", missed); print("false positives:", false_pos)
assert not missed and not false_pos

# ----- notebook cell: from dateutil.relativedelta import relat -----
from dateutil.relativedelta import relativedelta

DB_PATH = Path(os.getenv("PASSPORT_DB") or (LOCAL_DATA / "passport.db"))   # outside OneDrive, same reason as Chroma
db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.row_factory = sqlite3.Row
db.executescript("""
CREATE TABLE IF NOT EXISTS pets(id INTEGER PRIMARY KEY, name TEXT, species TEXT, breed TEXT, birth_date TEXT,
                                microchip TEXT, owner TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS vaccinations(id INTEGER PRIMARY KEY, pet_id INTEGER, vaccine TEXT, date_given TEXT, batch TEXT,
                                clinic TEXT, next_due TEXT, source_image TEXT, confirmed INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS treatments(id INTEGER PRIMARY KEY, pet_id INTEGER, type TEXT, product TEXT, date TEXT,
                                next_due TEXT, source_image TEXT, confirmed INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS health_events(id INTEGER PRIMARY KEY, pet_id INTEGER, date TEXT, note TEXT, photo TEXT);
CREATE TABLE IF NOT EXISTS reminders(id INTEGER PRIMARY KEY, pet_id INTEGER, what TEXT, due_date TEXT, status TEXT DEFAULT 'open');
""")
db.commit()

PASSPORT_DISCLAIMER = ("This record is the owner's own note. It does not replace an official vet-signed vaccination "
                       "certificate or an EU pet passport.")

# GENERAL placeholder intervals (days) used only when the passport/guideline gives none. Verify against your PDF tables + a vet.
GENERAL_INTERVAL_DAYS = {"rabies": 365, "dhpp": 365, "dapp": 365, "fvrcp": 365, "leptospirosis": 365, "bordetella": 365, "felv": 365}

def parse_date(s):
    if not s: return None
    try: return dt.date.fromisoformat(str(s)[:10])
    except ValueError: return None

def add_pet(name, species, breed="", birth_date="", microchip="", owner=""):
    cur = db.execute("INSERT INTO pets(name,species,breed,birth_date,microchip,owner) VALUES(?,?,?,?,?,?)",
                     (name, species, breed, birth_date, microchip, owner))
    db.commit(); return cur.lastrowid

def find_pet(name):
    return db.execute("SELECT * FROM pets WHERE lower(name)=lower(?) ORDER BY id DESC", (name,)).fetchone()

def known_pet_in(text):
    for r in db.execute("SELECT * FROM pets"):
        if re.search(rf"\b{re.escape(r['name'])}\b", text, re.I):
            return r
    return None

def log_vaccine(pet_id, vaccine, date_given, batch="", clinic="", next_due=None, source_image="", confirmed=1):
    """If next_due is not given, look up the interval: guideline-table citation + placeholder fallback."""
    note, due = "", parse_date(next_due)
    if due is None and parse_date(date_given):
        pet = db.execute("SELECT * FROM pets WHERE id=?", (pet_id,)).fetchone()
        days = next((d for k, d in GENERAL_INTERVAL_DAYS.items() if k in vaccine.lower()), None)
        if days:
            due = parse_date(date_given) + dt.timedelta(days=days)
            note = f"next due set from a general {days}-day interval - confirm with your vet."
        try:
            hits, _ = search_kb(f"{vaccine} booster interval schedule", pet["species"], content_types=["table", "text"], k=1)
            if hits:
                m = hits[0]["meta"]; note += f" Guideline: {m['source_title']} ({yr(m['year'])}) p.{m['page']}."
        except Exception:
            pass
    cur = db.execute("INSERT INTO vaccinations(pet_id,vaccine,date_given,batch,clinic,next_due,source_image,confirmed) VALUES(?,?,?,?,?,?,?,?)",
                     (pet_id, vaccine, str(date_given or ""), batch, clinic, due.isoformat() if due else None, source_image, confirmed))
    if due:
        db.execute("INSERT INTO reminders(pet_id,what,due_date) VALUES(?,?,?)", (pet_id, f"{vaccine} booster", due.isoformat()))
    db.commit()
    return {"id": cur.lastrowid, "next_due": due.isoformat() if due else None, "note": note.strip()}

def next_due(pet_name, vaccine=None):
    pet = find_pet(pet_name)
    if not pet: return None
    q = "SELECT * FROM vaccinations WHERE pet_id=? AND next_due IS NOT NULL"
    args = [pet["id"]]
    if vaccine:
        q += " AND lower(vaccine) LIKE ?"; args.append(f"%{vaccine.lower()}%")
    return [dict(r) for r in db.execute(q + " ORDER BY next_due", args)]

def upcoming_reminders(days=30):
    lim = (dt.date.today() + dt.timedelta(days=days)).isoformat()
    return [dict(r) for r in db.execute(
        "SELECT r.*, p.name FROM reminders r JOIN pets p ON p.id=r.pet_id WHERE r.status='open' AND r.due_date<=? ORDER BY r.due_date", (lim,))]
print("passport DB ready:", DB_PATH)

# ----- notebook cell: SYSTEM_PROMPT = -----
SYSTEM_PROMPT = """You are a pet-care assistant. Rules that always apply:
- Answer ONLY from the numbered SOURCES provided. If they do not answer the exact question, say what they DO say about the closest situation (for example the recommended age to start) and tell the user to confirm with a vet. Only if the sources say nothing relevant at all, reply with exactly: NO_ANSWER
- Ages may be given in days or months: convert to weeks (7 days = 1 week, 1 month is about 4 weeks) before comparing with the sources, and mention the conversion once, briefly, not after every line.
- Start with ONE sentence that directly answers the question (for example whether the pet's age is earlier or later than the standard schedule in the sources), then give the supporting details.
- Add a "See a vet if:" line only when the sources name specific warning signs. Otherwise leave that line out.
- Cite every factual sentence or bullet with its source number like [1] or [2][3]. Never cite a number that does not exist.
- No diagnosis, no prescriptions, no drug doses; point to a vet instead.
- Vaccine schedules are general guidance: name the source and say to confirm with a vet because rules vary by country.
- If a PET RECORD block is given, use it for the pet's own dates (cite it as [record]).
- For health questions, finish with one line starting exactly "See a vet if:" listing warning signs that appear in the sources.
- Write plain text only. For comparisons use the symbols ≤ and ≥ (or words). Never use LaTeX or dollar signs for maths.
Keep the answer short and practical."""

def build_context(hits):
    return "\n\n".join(f"[{i}] ({h['meta']['source_title']}, {yr(h['meta']['year'])}, p.{h['meta']['page']}, {h['meta']['content_type']})\n{h['text']}"
                       for i, h in enumerate(hits, 1))

_MATH = [(r"\\leq?\b", "≤"), (r"\\geq?\b", "≥"), (r"\\times\b", "×"), (r"\\pm\b", "±"), (r"\\approx\b", "≈"),
         (r"\\sim\b", "~"), (r"\\mu\b", "µ"), (r"\\(?:rightarrow|to)\b", "→"), (r"\\%", "%")]
def clean_math(text):
    """Models sometimes write LaTeX (e.g. $\le$16). Turn the common bits into plain symbols."""
    for pat, sym in _MATH:
        text = re.sub(pat, sym, text)
    return re.sub(r"\$\s*([≤≥×±≈~µ→<>%])\s*\$", r"\1", text)

def expand_query(q):
    """Guidelines speak in weeks, owners in days/months: add the week equivalent to the search text only."""
    extra = [f"{max(1, round(int(n) / 7))} weeks of age" for n in re.findall(r"(\d+)\s*-?\s*days?\b", q, re.I)]
    extra += [f"{int(n) * 4} weeks of age" for n in re.findall(r"(\d+)\s*-?\s*months?\b", q, re.I)]
    return q if not extra else f"{q} ({'; '.join(extra)})"

_FILLER = re.compile(r"^\s*See a vet if:\s*(?:[\[\(]\s*)?(?:(?:there are|any)\s+)*"
                     r"(?:(?:no|none|not)\b|(?:warning\s+)?signs\s+(?:listed|mentioned|noted))[^\n]*$", re.I | re.M)
_CONV = re.compile(r"\s*\((?:conversion|converting)[^)]*\)", re.I)
def tidy_answer(text):
    """Drop content-free 'See a vet if...' lines, and keep an age-conversion note only the first time it appears."""
    text = _FILLER.sub("", text)
    text = re.sub(r"\s*\((?:conversion|converting)[^)]*?\b(\w+) to \1\b[^)]*\)", "", text, flags=re.I)   # '(converting weeks to weeks)'
    text = re.sub(r"\s*\((?:conversion|converting)\s+([^()]*?)\s+to\s+\1\s*\)", "", text, flags=re.I)    # '(converting 16 weeks to 16 weeks)'
    seen = []
    def keep_first(m):
        if seen:
            return ""
        seen.append(1)
        return m.group(0)
    return _CONV.sub(keep_first, text).strip()

RELAXED_RULE = ("\n- IMPORTANT: the SOURCES were found relevant to this question. Do NOT reply NO_ANSWER. Summarize what the "
                "sources say that relates to the question, and state plainly which part of the question they do not cover.")

VACCINE_NOTE = "Note: vaccine schedules are general guidance and rules differ by country, so please confirm with your vet."
def add_vaccine_note(question, text):
    """Design rule: every vaccine answer says to confirm with a vet (added by code, not left to the model)."""
    if re.search(r"vaccin|booster|\bshots?\b|\bjabs?\b", question + " " + text, re.I) and \
            not re.search(r"(confirm|confirmed|check|checked) with (your|a|the) vet", text, re.I):
        return text + "\n\n" + VACCINE_NOTE
    return text

def enforce_citations(answer, n_sources, allow_record=False):
    """Drop long claim lines with no valid citation (the 'final check')."""
    keep, dropped = [], 0
    for line in answer.splitlines():
        s = line.strip()
        if not s: keep.append(line); continue
        nums = [int(x) for x in re.findall(r"\[(\d+)\]", s)]
        has_valid = any(1 <= x <= n_sources for x in nums) or (allow_record and "[record]" in s)
        exempt = s.lower().startswith(("see a vet", "note:", "please confirm", "confirm with")) or len(s) < 60
        if has_valid or exempt:
            keep.append(re.sub(r"\[(\d+)\]", lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_sources else "", line))
        else:
            dropped += 1
    return "\n".join(keep).strip(), dropped

def passport_context(pet, question):
    vac = None
    mv = re.search(r"(rabies|dhpp|dapp|fvrcp|lepto\w*|bordetella|felv|booster|vaccine)", question, re.I)
    rows = next_due(pet["name"], mv.group(1) if mv and mv.group(1).lower() not in ("booster", "vaccine") else None) or []
    if not rows: return "", rows
    txt = f"PET RECORD (owner's own note, not an official certificate): {pet['name']} ({pet['species']}, {pet['breed'] or 'breed n/a'})\n"
    txt += "\n".join(f"- {r['vaccine']}: given {r['date_given'] or 'n/a'}, next due {r['next_due']}" for r in rows)
    return txt, rows

def species_from_history(history):
    """The animal the user last talked about earlier in this chat (puppy -> dog)."""
    for m in reversed(history or []):
        if m.get("role") == "user":
            sp = detect_species(m.get("text", ""))
            if sp:
                return sp
    return None

def make_standalone(question, history, species):
    """Turn a short follow-up ('and the intervals?') into a complete question, using the chat so far."""
    if not history:
        return question
    words = question.split()
    users = [m["text"] for m in history if m.get("role") == "user"]
    bare_animal = len(words) <= 2 and detect_species(question) is not None     # e.g. the reply 'dog' to 'Which animal?'
    fallback = question
    if bare_animal:
        earlier = [t for t in users if not (len(t.split()) <= 2 and detect_species(t))]
        if earlier:
            fallback = f"{earlier[-1]} ({question.strip()})"
    elif not (len(words) <= 14 or re.match(r"\s*(and|also|what about|how about|then|so|ok|okay)\b", question, re.I)):
        return question                                          # long message: already complete
    if not LLM_ENABLED:
        return fallback
    convo = "\n".join(f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('text', '')[:400]}" for m in history[-6:])
    try:
        q = llm("Rewrite the user's latest message as ONE complete, standalone pet-care question. Carry over only the animal and "
                "the topic from the conversation. Do NOT carry over a specific age, number or other detail unless the latest "
                "message itself mentions it. If it is already complete, return it unchanged. "
                "Reply with the question only.\n\nCONVERSATION:\n" + convo + "\n\nLATEST MESSAGE: " + question, max_tokens=80)
        q = q.strip().strip('"')
        return q if 3 <= len(q) <= 300 else fallback
    except Exception:
        return fallback

def ask(question, pet_name=None, species=None, region=None, history=None):
    """history: earlier messages of this chat, [{'role': 'user'|'assistant', 'text': str}, ...] (oldest first)."""
    out = {"kind": None, "answer": "", "sources": [], "images": []}
    history = [m for m in (history or []) if isinstance(m, dict) and m.get("text")][-8:]
    pet = find_pet(pet_name) if pet_name else known_pet_in(question)
    species = species or (pet["species"] if pet else None) or detect_species(question) or species_from_history(history)

    t = triage(question, species)                              # 1. emergency: skips everything else
    if t:
        out.update(kind="emergency", answer=f"{t['response']}\n\n(Matched rule: {t['rule']}.)"); return out
    if not species:                                            # 2. species routing
        out.update(kind="clarify", answer="Which animal is this about (dog, cat, fish, cow or horse)?"); return out

    standalone = make_standalone(question, history, species)   # follow-ups become complete questions
    out["question_used"] = standalone

    pctx, prow = "", []                                        # 3. passport tool
    if pet and re.search(r"\b(next|due|booster|when|passport|record|last)\b", standalone, re.I):
        pctx, prow = passport_context(pet, standalone)

    hits, best = search_kb(expand_query(standalone), species, region)   # 4. knowledge-base tool (+ one retry)
    if (not hits or best > MAX_DISTANCE) and LLM_ENABLED:
        try:
            q2 = llm(f"Rewrite this pet-care question as a short keyword-rich search query (species: {species}). Reply with the query only.\n\n{standalone}", max_tokens=40)
            h2, b2 = search_kb(q2, species, region)
            if h2 and b2 < best: hits, best = h2, b2
        except Exception: pass
    if (not hits or best > MAX_DISTANCE) and not pctx:
        out.update(kind="no_answer", answer="I could not find this in my sources, so I won't guess. Please check with a vet."); return out

    out["sources"] = [{"n": i, **{k: h["meta"][k] for k in ("source_title", "year", "page", "content_type", "image_path")}}
                      for i, h in enumerate(hits, 1)]
    out["images"] = [h["meta"]["image_path"] for h in hits if h["meta"]["image_path"]]
    if not LLM_ENABLED:                                        # retrieval-only mode
        out.update(kind="retrieval_only", answer=build_context(hits)); return out

    prompt = f"{pctx}\n\nSOURCES:\n{build_context(hits)}\n\nQUESTION: {standalone}"   # 5. cited answer
    ans = clean_math(llm(prompt, system=SYSTEM_PROMPT, max_tokens=700))
    gave_up = lambda a: bool(re.match(r"\s*NO_ANSWER\b", a) or (re.match(r"\s*i (do not|don't) know\b", a, re.I) and len(a) < 200))
    if gave_up(ans) and best <= MAX_DISTANCE:                  # search found relevant pages: try once more, less strictly
        ans = clean_math(llm(prompt, system=SYSTEM_PROMPT + RELAXED_RULE, max_tokens=700))
    if gave_up(ans):
        out.update(kind="no_answer", sources=[], images=[],
                   answer="I couldn't find a clear answer to that in my sources, so I won't guess. Please check with a vet.")
        return out
    ans = add_vaccine_note(standalone, tidy_answer(ans))
    clean, dropped = enforce_citations(ans, len(hits), allow_record=bool(pctx))
    if dropped and len(clean) < 40:
        out.update(kind="no_answer", answer="I could not support an answer with my sources, so I won't guess. Please check with a vet."); return out
    out.update(kind="answer", answer=clean)
    return out


def show(res):
    print(f"[{res['kind']}]"); print(res["answer"])
    if res["sources"]:
        print("\nSources:")
        for s in res["sources"]:
            print(f"  [{s['n']}] {s['source_title']} ({yr(s['year'])}) p.{s['page']} - {s['content_type']}")
    for img in res["images"][:2]:
        display(Image(filename=str(ROOT / img)))
