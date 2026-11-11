"""Packs the knowledge base built by the notebook into ONE file, ./knowledge_base.json.gz, which is what the app (and GitHub) uses.

A single file is easier to upload than Chroma's folder of database and index files, and it does not depend on the
Chroma version or on any hidden files arriving intact.

Run this after you re-run the notebook's indexing step (for example after adding Dog Facts):
    python server/export_knowledge_base.py
"""
import array
import base64
import gzip
import json
import os
import sys
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parent.parent
src = Path(os.getenv("CHROMA_PATH") or Path(os.getenv("LOCALAPPDATA") or Path.home()) / "PetCare" / "chroma")
out = ROOT / "knowledge_base.json.gz"

if not (src / "chroma.sqlite3").exists():
    sys.exit(f"No knowledge base found at {src}. Run the notebook's indexing step first.")

client = chromadb.PersistentClient(path=str(src))
snapshot = {"version": 1, "collections": {}}
for col in client.list_collections():
    ids, docs, metas, vecs = [], [], [], []
    total = col.count()
    for offset in range(0, total, 200):
        page = col.get(limit=200, offset=offset, include=["documents", "metadatas", "embeddings"])
        ids += page["ids"]
        docs += page["documents"]
        metas += page["metadatas"]
        # float32 packed as base64: exact, and about 4 KB per chunk
        vecs += [base64.b64encode(array.array("f", [float(x) for x in e]).tobytes()).decode("ascii") for e in page["embeddings"]]
    snapshot["collections"][col.name] = {"ids": ids, "documents": docs, "metadatas": metas, "embeddings": vecs}
    print(f"  {col.name}: {len(ids)} chunks")

with gzip.open(out, "wt", encoding="utf-8", compresslevel=9) as f:
    json.dump(snapshot, f, separators=(",", ":"))
print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
