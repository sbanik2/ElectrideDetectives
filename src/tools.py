"""Every tool an agent can call. Configure once with setup_tools(config, encoder).

web_search(query)        -> general web results (DuckDuckGo via ddgs)
calculator(expression)   -> safe arithmetic
literature_search(query) -> arXiv search via the arxiv-mcp-server (titles + full abstracts)
fetch_paper(arxiv_id)    -> download full text into the knowledge-base folder and index it
kb_search(query)         -> semantic search over every PDF / arXiv paper in the knowledge base
                            "[arXiv:2103.12345] phonon anharmonic" restricts to one source

All tools take ONE string and always return a string (never raise), so an agent
always sees either evidence or an explicit failure message.
"""
import os, sys, json, glob, re, time, shutil, asyncio, threading
import numpy as np
import pymupdf
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
# Defaults; setup_tools() overwrites them from config.yaml.
KB_DIR = os.path.abspath("inputs/kb")
LOG_PATH = os.path.abspath("outputs/arxiv_mcp.log")

WEB_MAX_RESULTS = 3        # web_search
CALC_MAX_CHARS = 120       # calculator
MAX_RESULTS = 3            # literature
TOP_K = 4                  # context passages per kb_search
ARXIV_MIN_INTERVAL = 3.0   # arXiv asks for >= 3 s between API requests


# ----------------------------------------------------------------------------- web + arithmetic
def web_search(query: str) -> str:
    """Live web search. Requires: %pip install ddgs"""
    try:
        from ddgs import DDGS
        hits = list(DDGS().text(query, max_results=WEB_MAX_RESULTS))
        return json.dumps([{"title": h.get("title"), "url": h.get("href"), "snippet": h.get("body")} for h in hits])
    except Exception as exc:
        return f"Web search unavailable: {type(exc).__name__}: {exc}"

def calculator(expression: str) -> str:
    """Safe basic arithmetic only (no Python code execution)."""
    import ast, operator
    ops = {ast.Add:operator.add, ast.Sub:operator.sub, ast.Mult:operator.mul,
           ast.Div:operator.truediv, ast.Pow:operator.pow, ast.USub:operator.neg}
    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int,float): return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops: return ops[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            a,b = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and (abs(a)>1e6 or abs(b)>10): raise ValueError("Exponent too large")
            return ops[type(node.op)](a,b)
        raise ValueError("Only basic arithmetic expressions are allowed")
    try:
        if len(expression)>CALC_MAX_CHARS: raise ValueError("Expression too long")
        return str(evaluate(ast.parse(expression, mode="eval").body))
    except Exception as exc:
        return f"Calculation unavailable: {type(exc).__name__}: {exc}"


# ----------------------------------------------------------------------------- arXiv via MCP
def _server_command():
    """arxiv-mcp-server next to this kernel's python (bin/ on Linux/mac, Scripts\\ on Windows)."""
    here = os.path.join(os.path.dirname(sys.executable), "arxiv-mcp-server")
    for cand in (here, here + ".exe", shutil.which("arxiv-mcp-server")):
        if cand and os.path.exists(cand):
            return cand
    return "arxiv-mcp-server"

ARXIV_SERVER = None        # built in setup_tools() once KB_DIR is known

# Every call starts a fresh server process, so the server's own rate limiter resets each
# time; this lock + interval is what actually keeps 5 parallel agents under arXiv's limit.
_ARXIV_LOCK = threading.Lock()
_last_arxiv_call = [0.0]

async def _arxiv_call(tool, args):
    with open(LOG_PATH, "a") as log:
        async with stdio_client(ARXIV_SERVER, errlog=log) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                res = await s.call_tool(tool, args)
    text = "".join(getattr(c, "text", "") for c in res.content)
    if getattr(res, "isError", False) and not text.lstrip().startswith("{"):
        raise RuntimeError(text[:300] or f"MCP tool {tool} returned an error")
    return text                           # JSON error payloads are reported by _parse_status

def _run_coro(coro):
    """asyncio.run that also works when called from a notebook cell (which has a running loop)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)          # normal case: worker thread from asyncio.to_thread
    box = {}
    def target():
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:
            box["error"] = exc
    t = threading.Thread(target=target)
    t.start(); t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]

def _run_arxiv(tool, args):
    with _ARXIV_LOCK:
        wait = ARXIV_MIN_INTERVAL - (time.monotonic() - _last_arxiv_call[0])
        if wait > 0:
            time.sleep(wait)
        try:
            return _run_coro(_arxiv_call(tool, args))
        finally:
            _last_arxiv_call[0] = time.monotonic()

def _parse_status(raw):
    """Server replies are JSON; failures carry status 'error' or 'rate_limited'."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None, raw
    if isinstance(data, dict) and data.get("status") in ("error", "rate_limited"):
        return None, f"{data['status']}: {data.get('message', '')}"
    return data, None

_ID_RE = re.compile(r"(\d{4}\.\d{4,5}|[a-z][a-z\-]*(?:\.[A-Z]{2})?/\d{7})(v\d+)?")

def _normalize_id(text):
    """Accepts 2103.12345, arXiv:2103.12345v2, hep-th/9901001 or an arxiv.org URL."""
    m = _ID_RE.search(text.strip())
    return m.group(0) if m else text.strip()

def _bare_id(pid):
    return re.sub(r"v\d+$", "", pid)


def literature_search(query: str) -> str:
    """arXiv search via MCP: top MAX_RESULTS papers with full abstracts."""
    try:
        raw = _run_arxiv("search_papers",
                         {"query": query, "max_results": MAX_RESULTS, "abstract_mode": "full"})
        data, err = _parse_status(raw)
        if err:
            return f"Literature search FAILED (not an empty result) - {err}"
        papers = data.get("papers", []) if isinstance(data, dict) else []
        if not papers:
            return f"Literature search returned no arXiv papers for {query!r}; try other or fewer keywords."
        for p in papers:                              # remember titles for the local-collection listing
            _TITLES["arXiv:" + _bare_id(_normalize_id(p.get("versioned_id") or p["id"]))] = p.get("title", "")
        return json.dumps([{"arxiv_id": _bare_id(_normalize_id(p.get("versioned_id") or p["id"])),
                            "title": p.get("title", ""), "authors": p.get("authors", [])[:3],
                            "year": (p.get("published") or "")[:4], "abstract": p.get("abstract", ""),
                            "url": p.get("url", "")}
                           for p in papers])
    except Exception as exc:
        return f"Literature search unavailable: {type(exc).__name__}: {exc}"


# ----------------------------------------------------------------------------- local RAG
CHUNK_WORDS, OVERLAP = 150, 30   # MiniLM truncates at 256 word-pieces; sci text is ~1.4-1.7 pieces/word
MIN_WORDS = 20                   # only drop near-empty text (figure-only pages, page numbers)
SKIP_REFERENCE_CHUNKS = True     # bibliography chunks are keyword-dense and outrank real content

rag_encoder = None               # the swarm's shared SentenceTransformer, set in setup_tools()
_ENC_LOCK = threading.Lock()     # agents call tools from parallel threads
_KB_LOCK = threading.Lock()      # serialises index rebuilds / appends
_KB = ([], None)                 # (chunks, vecs) replaced as ONE object: readers never see a mismatch
KB_CHUNKS, KB_VECS = [], None    # read-only mirrors for inspection in the notebook
_TITLES = {}                     # source -> title (PDF metadata, or arXiv title seen in literature_search)

def _embedding_dim():
    """sentence-transformers renamed get_sentence_embedding_dimension -> get_embedding_dimension."""
    get = getattr(rag_encoder, "get_embedding_dimension", None) or rag_encoder.get_sentence_embedding_dimension
    return get()

def _encode(texts):
    """Normalized embeddings; the lock lets peer selection and kb_search share one model safely."""
    with _ENC_LOCK:
        return np.asarray(rag_encoder.encode(texts, normalize_embeddings=True, batch_size=64),
                          dtype=np.float32)

def _set_kb(chunks, vecs):
    global _KB, KB_CHUNKS, KB_VECS
    _KB = (chunks, vecs)
    KB_CHUNKS, KB_VECS = chunks, vecs

_YEAR = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")
_CITE = re.compile(r"et al\.|Phys\.\s?Rev|J\.\s?(?:Chem|Phys|Appl|Am)|Nat(?:ure)?\.?\s|doi|arXiv:|Proc\.|Lett\.")

def _looks_like_references(text):
    """Heuristic: a slice of a bibliography has many years AND many journal/citation markers."""
    return len(_YEAR.findall(text)) >= 6 and len(_CITE.findall(text)) >= 5

def _chunk_words(words, source, page, chunks):
    """Evenly spaced windows overlapping by >= OVERLAP words, ending exactly at the last word,
    so no text is dropped and no two windows are near-duplicates."""
    n = len(words)
    if n < MIN_WORDS:
        return
    m = 1 if n <= CHUNK_WORDS else -(-(n - CHUNK_WORDS) // (CHUNK_WORDS - OVERLAP)) + 1
    for s in np.linspace(0, max(n - CHUNK_WORDS, 0), m).round().astype(int):
        text = " ".join(words[s:s + CHUNK_WORDS])
        if SKIP_REFERENCE_CHUNKS and _looks_like_references(text):
            continue
        chunks.append({"source": source, "page": page, "text": text})

def _source_name(path):
    stem, ext = os.path.splitext(os.path.basename(path))
    if ext.lower() == ".md":
        logical = stem.replace("__", "/")            # server stores hep-th/9901001 as hep-th__9901001
        if _ID_RE.fullmatch(logical):
            return "arXiv:" + logical
    return os.path.basename(path)

def _chunks_for_file(path):
    chunks, source = [], _source_name(path)
    if path.lower().endswith(".pdf"):
        with pymupdf.open(path) as doc:
            title = ((doc.metadata or {}).get("title") or "").strip()
            if len(title) > 10 and not title.lower().startswith(("microsoft word", "untitled")):
                _TITLES[source] = title
            for page_no, page in enumerate(doc, start=1):
                _chunk_words(page.get_text().split(), source, page_no, chunks)
    else:
        with open(path, encoding="utf-8", errors="replace") as f:
            _chunk_words(f.read().split(), source, None, chunks)
    return chunks

def _kb_files(kb_dir=None):
    kb_dir = KB_DIR if kb_dir is None else kb_dir
    return sorted(glob.glob(os.path.join(kb_dir, "*.pdf")) + glob.glob(os.path.join(kb_dir, "*.md")))

def load_kb_chunks(kb_dir=None):
    chunks = []
    for path in _kb_files(kb_dir):
        try:
            chunks.extend(_chunks_for_file(path))
        except Exception as exc:                      # one broken PDF must not kill the whole index
            print(f"KB: skipped {os.path.basename(path)} ({type(exc).__name__}: {exc})")
    return chunks

def build_kb_index():
    """(Re)embed every chunk in inputs/kb. Call again after adding PDFs by hand."""
    with _KB_LOCK:
        chunks = load_kb_chunks()
        dim = _embedding_dim()
        vecs = _encode([c["text"] for c in chunks]) if chunks else np.zeros((0, dim), np.float32)
        _set_kb(chunks, vecs)
    indexed = {c["source"] for c in chunks}
    print(f"KB index: {len(chunks)} chunks from {len(indexed)} files in {KB_DIR}")
    for path in _kb_files():
        if _source_name(path) not in indexed:
            print(f"KB: no extractable text in {os.path.basename(path)} (scanned image? run OCR first)")

def kb_sources():
    """[(source, title, n_passages)] for every paper currently in the knowledge base."""
    chunks, _ = _KB
    counts = {}
    for c in chunks:
        counts[c["source"]] = counts.get(c["source"], 0) + 1
    return [(src, _TITLES.get(src, ""), n) for src, n in sorted(counts.items())]

def add_to_kb_index(path):
    """Embed one (new or re-downloaded) file and merge it into the index; returns #chunks added."""
    with _KB_LOCK:
        chunks, vecs = _KB
        if vecs is None:
            vecs = np.zeros((0, _embedding_dim()), np.float32)
        new = _chunks_for_file(path)
        source = _source_name(path)
        keep = [i for i, c in enumerate(chunks) if c["source"] != source]
        new_vecs = _encode([c["text"] for c in new]) if new else vecs[:0]
        _set_kb([chunks[i] for i in keep] + new, np.vstack([vecs[keep], new_vecs]))
        return len(new)

_FILTER = re.compile(r"^\s*\[([^\]]+)\]\s*(.*)$", re.S)

def kb_search(query: str, k: int = None) -> str:
    """RAG over local papers in inputs/kb: top-k chunks with source + page.
    Optional source filter: "[arXiv:2103.12345] keywords" or "[myfile.pdf] keywords"."""
    try:
        k = TOP_K if k is None else k
        chunks, vecs = _KB                            # one consistent snapshot
        if not chunks:
            return "Knowledge base is empty (no papers in inputs/kb, or build_kb_index() not run)."
        note, wanted = "", None
        m = _FILTER.match(query)
        if m:
            key = m.group(1).strip().lower().removeprefix("arxiv:")
            query = m.group(2).strip() or m.group(1)
            wanted = {c["source"] for c in chunks if key in c["source"].lower()}
            if not wanted:
                note = f"No local source matches {m.group(1)!r}; searched all papers. "
                wanted = None
        scores = vecs @ _encode([query])[0]          # cosine similarity (vectors are normalised)
        order = np.argsort(-scores)
        if wanted is not None:
            order = [i for i in order if chunks[i]["source"] in wanted]
        picked = list(order[:k])
        result = json.dumps([{"source": chunks[i]["source"], "page": chunks[i]["page"],
                              "score": round(float(scores[i]), 3), "text": chunks[i]["text"]}
                             for i in picked])
        return note + result
    except Exception as exc:
        return f"KB search unavailable: {type(exc).__name__}: {exc}"


def fetch_paper(arxiv_id: str) -> str:
    """Download an arXiv paper's full text into inputs/kb via MCP, then index it."""
    try:
        pid = _normalize_id(arxiv_id)
        raw = _run_arxiv("download_paper", {"paper_id": pid})
        data, err = _parse_status(raw)
        if err:
            return f"Download FAILED for {pid} - {err}"
        stored = _bare_id((data or {}).get("paper_id") or pid) if isinstance(data, dict) else _bare_id(pid)
        md_path = os.path.join(KB_DIR, stored.replace("/", "__") + ".md")
        if not os.path.exists(md_path):
            return f"Download FAILED for {pid}: server reported success but {os.path.basename(md_path)} is missing."
        n = add_to_kb_index(md_path)
        if n == 0:
            return f"Downloaded arXiv:{stored} but no readable text was extracted."
        return (f"Downloaded arXiv:{stored} ({n} passages indexed). In a later round read it with "
                f"kb_search, e.g. '[arXiv:{stored}] your keywords'.")
    except Exception as exc:
        return f"Paper download unavailable: {type(exc).__name__}: {exc}"


# ----------------------------------------------------------------------------- registry + setup
ALL_TOOLS = {"web_search": web_search, "calculator": calculator, "literature_search": literature_search,
             "fetch_paper": fetch_paper, "kb_search": kb_search}

def setup_tools(config, encoder):
    """Apply config.yaml (tools + paths sections), share the swarm's encoder, build the KB index.
    Returns the {name: function} registry of enabled tools."""
    global KB_DIR, LOG_PATH, ARXIV_SERVER, rag_encoder, WEB_MAX_RESULTS, CALC_MAX_CHARS, MAX_RESULTS
    global ARXIV_MIN_INTERVAL, TOP_K, CHUNK_WORDS, OVERLAP, MIN_WORDS, SKIP_REFERENCE_CHUNKS
    t, paths = config["tools"], config["paths"]
    KB_DIR, LOG_PATH = os.path.abspath(paths["kb_dir"]), os.path.abspath(paths["arxiv_log"])
    os.makedirs(KB_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    ARXIV_SERVER = StdioServerParameters(command=_server_command(), args=["--storage-path", KB_DIR])
    WEB_MAX_RESULTS = t["web_search"]["max_results"]
    CALC_MAX_CHARS = t["calculator"]["max_expression_chars"]
    MAX_RESULTS = t["literature_search"]["max_results"]
    ARXIV_MIN_INTERVAL = t["literature_search"]["min_interval_s"]
    kb = t["kb_search"]
    TOP_K, CHUNK_WORDS, OVERLAP = kb["top_k"], kb["chunk_words"], kb["overlap_words"]
    MIN_WORDS, SKIP_REFERENCE_CHUNKS = kb["min_words"], kb["skip_reference_chunks"]
    rag_encoder = encoder
    unknown = set(t["enabled"]) - set(ALL_TOOLS)
    if unknown:
        raise ValueError(f"config tools.enabled has unknown tools: {sorted(unknown)}; available: {list(ALL_TOOLS)}")
    build_kb_index()
    return {name: ALL_TOOLS[name] for name in t["enabled"]}
