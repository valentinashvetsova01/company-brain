"""Al Dente Company Brain - backend entry point."""

import asyncio
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv(override=True)

from agent import run_agent  # noqa: E402 – must load .env first

app = FastAPI(title="Al Dente Company Brain")

_STATIC = Path(__file__).resolve().parent / "static"
_FILES = _STATIC / "files"
_FILES.mkdir(parents=True, exist_ok=True)

app.mount("/files", StaticFiles(directory=_FILES), name="files")


# ── /ask (frozen schema) ─────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    verticale: str
    artifact_url: str | None = None


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        result = run_agent(request.question)
        return AskResponse(**result)
    except Exception as exc:
        return AskResponse(
            answer=f"I cannot answer right now due to an unexpected error: {exc}",
            sources=[],
            verticale="kb",
            artifact_url=None,
        )


# ── /graph-data ───────────────────────────────────────────────────────────────

_GRAPH_CACHE: dict = {}
_GRAPH_CACHE_TTL = 600  # 10 minutes

_SKU_RE = re.compile(r"\b(PAS-[A-Z]{3}-[A-Z0-9]{3,5}|RAW-[A-Z]{3}-[A-Z0-9]{3,5})\b")
_CUST_RE = re.compile(r"\b(CUST-\d{4})\b")


async def _fetch_all(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    params: dict | None = None,
) -> list[dict]:
    """Fetch all pages from a paginated endpoint, capped at 400 items."""
    base = {**(params or {}), "limit": "200", "offset": "0"}
    items: list[dict] = []
    try:
        r = await client.get(url, params=base, headers=headers, timeout=12)
        r.raise_for_status()
        body = r.json()
        items.extend(body.get("data", []))
        total = body.get("pagination", {}).get("total", 0)
        # fetch second page if needed (rare for graph)
        if total > 200 and len(items) < 400:
            base2 = {**base, "offset": "200"}
            r2 = await client.get(url, params=base2, headers=headers, timeout=12)
            if r2.status_code == 200:
                items.extend(r2.json().get("data", []))
    except Exception:
        pass
    return items


async def _fetch_bom(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    sku: str,
) -> list[dict]:
    try:
        r = await client.get(url, params={"sku": sku, "limit": "50"}, headers=headers, timeout=8)
        if r.status_code == 200:
            return r.json().get("data", [])
    except Exception:
        pass
    return []


async def _build_graph() -> dict:
    base_url = os.environ.get("MOCK_API_BASE_URL", "").rstrip("/")
    token = os.environ.get("MOCK_API_TOKEN", "")

    if not base_url or not token:
        return {
            "nodes": [], "edges": [], "stats": {},
            "error": "API credentials not configured — deploy backend/.env",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    headers = {"Authorization": f"Bearer {token}"}
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_edges: set[tuple] = set()

    def add_edge(frm: str, to: str, label: str, etype: str) -> None:
        key = (frm, to, label)
        if key not in seen_edges:
            seen_edges.add(key)
            edges.append({"from": frm, "to": to, "label": label, "type": etype})

    async with httpx.AsyncClient() as client:
        # ── parallel fetch of main collections ──────────────────────────────
        customers, suppliers, fg_items, rm_items = await asyncio.gather(
            _fetch_all(client, f"{base_url}/crm/customers", headers),
            _fetch_all(client, f"{base_url}/erp/suppliers", headers),
            _fetch_all(client, f"{base_url}/erp/inventory", headers, {"type": "finished_good"}),
            _fetch_all(client, f"{base_url}/erp/inventory", headers, {"type": "raw_material"}),
            return_exceptions=False,
        )

        # ── supplier nodes ───────────────────────────────────────────────────
        supplier_map: dict[str, str] = {}  # id -> name
        for s in suppliers:
            sid = s.get("id") or s.get("supplier_id")
            if not sid:
                continue
            name = s.get("name", sid)
            supplier_map[sid] = name
            nodes.append({
                "id": sid, "label": _short_label(sid, name, 18),
                "type": "supplier",
                "meta": {"name": name, "category": s.get("category"), "country": s.get("country")},
            })

        # ── raw-material nodes ───────────────────────────────────────────────
        rm_set: set[str] = set()
        rm_supplier: dict[str, str] = {}  # sku -> supplier_id
        for rm in rm_items:
            sku = rm.get("sku")
            if not sku or not sku.startswith("RAW-"):
                continue
            rm_set.add(sku)
            below = bool(rm.get("below_min"))
            sup_id = rm.get("supplier_id")
            if sup_id:
                rm_supplier[sku] = sup_id
            nodes.append({
                "id": sku, "label": _short_label(sku, rm.get("description", ""), 16),
                "type": "raw_material", "below_min": below,
                "meta": {
                    "description": rm.get("description"), "on_hand": rm.get("on_hand"),
                    "min_stock": rm.get("min_stock"), "unit": rm.get("unit"),
                    "below_min": below, "supplier_id": sup_id,
                },
            })
            if sup_id and sup_id in supplier_map:
                add_edge(sku, sup_id, "from", "supply")

        # ── finished-good nodes (cap at 50) ──────────────────────────────────
        fg_list = _prioritize_fg(fg_items, limit=50)
        fg_set: set[str] = set()
        for fg in fg_list:
            sku = fg.get("sku")
            if not sku or not sku.startswith("PAS-"):
                continue
            fg_set.add(sku)
            below = bool(fg.get("below_min"))
            nodes.append({
                "id": sku, "label": _short_label(sku, fg.get("description", ""), 16),
                "type": "product", "below_min": below,
                "meta": {
                    "description": fg.get("description"), "on_hand": fg.get("on_hand"),
                    "min_stock": fg.get("min_stock"), "unit": fg.get("unit"),
                    "below_min": below,
                },
            })

        # ── BOM edges (fetch in parallel, limited to first 20 SKUs) ──────────
        bom_skus = list(fg_set)[:20]
        bom_results = await asyncio.gather(
            *[_fetch_bom(client, f"{base_url}/erp/bom", headers, sku) for sku in bom_skus],
            return_exceptions=True,
        )
        for sku, bom_data in zip(bom_skus, bom_results):
            if isinstance(bom_data, Exception) or not bom_data:
                continue
            for comp in bom_data:
                raw_sku = (
                    comp.get("raw_material_sku") or comp.get("raw_sku")
                    or comp.get("component_sku") or comp.get("sku")
                )
                if raw_sku and raw_sku.startswith("RAW-") and raw_sku in rm_set:
                    add_edge(sku, raw_sku, "uses", "bom")

        # ── customer nodes (cap at 30 active first) ─────────────────────────
        cust_list = _prioritize_customers(customers, limit=30)
        for c in cust_list:
            cid = c.get("id") or c.get("customer_id")
            if not cid:
                continue
            name = c.get("name", cid)
            nodes.append({
                "id": cid, "label": _short_label(cid, name, 18),
                "type": "customer",
                "meta": {
                    "name": name, "channel": c.get("channel"),
                    "status": c.get("status"), "city": c.get("city"),
                },
            })

    # ── KB document nodes + edges ────────────────────────────────────────────
    kb_dir = Path(__file__).parent / "data" / "kb"
    all_ids_in_graph = {n["id"] for n in nodes}
    for doc_path in sorted(kb_dir.glob("*.md")):
        doc_id = doc_path.stem
        try:
            content = doc_path.read_text(encoding="utf-8")
        except Exception:
            continue
        title_m = re.search(r"^#\s+(.+)", content, re.MULTILINE)
        title = title_m.group(1).strip() if title_m else doc_id
        nodes.append({
            "id": doc_id,
            "label": f"{doc_id}\n{title[:22]}",
            "type": "kb_doc",
            "meta": {"title": title, "id": doc_id},
        })
        # Edges to mentioned SKUs / customer IDs present in graph
        mentioned_skus = set(_SKU_RE.findall(content)) & all_ids_in_graph
        mentioned_custs = set(_CUST_RE.findall(content)) & all_ids_in_graph
        for target in list(mentioned_skus)[:4] + list(mentioned_custs)[:2]:
            add_edge(doc_id, target, "covers", "kb")

    # ── stats ────────────────────────────────────────────────────────────────
    type_counts = {}
    for n in nodes:
        type_counts[n["type"]] = type_counts.get(n["type"], 0) + 1

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            **type_counts,
            "edges": len(edges),
            "below_min": sum(1 for n in nodes if n.get("below_min")),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _short_label(id_: str, desc: str, max_desc: int) -> str:
    desc = (desc or "").strip()
    if desc:
        return f"{id_}\n{desc[:max_desc]}"
    return id_


def _prioritize_fg(items: list[dict], limit: int) -> list[dict]:
    """Sort finished goods: below-min first, then alphabetically."""
    below = [i for i in items if i.get("below_min")]
    above = [i for i in items if not i.get("below_min")]
    return (below + above)[:limit]


def _prioritize_customers(items: list[dict], limit: int) -> list[dict]:
    """Sort customers: active first, then prospect, then inactive."""
    order = {"active": 0, "prospect": 1, "inactive": 2}
    return sorted(items, key=lambda c: order.get(c.get("status", ""), 9))[:limit]


@app.get("/graph-data")
async def graph_data() -> JSONResponse:
    cached = _GRAPH_CACHE.get("data")
    if cached and time.time() - _GRAPH_CACHE.get("ts", 0) < _GRAPH_CACHE_TTL:
        return JSONResponse(content=cached)
    try:
        data = await _build_graph()
    except Exception as exc:
        data = {
            "nodes": [], "edges": [], "stats": {},
            "error": f"Graph build failed: {exc}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    _GRAPH_CACHE["data"] = data
    _GRAPH_CACHE["ts"] = time.time()
    return JSONResponse(content=data)
