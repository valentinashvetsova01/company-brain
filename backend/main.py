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
    max_items: int = 1000,
) -> list[dict]:
    """Fetch all pages until total is exhausted."""
    base = {**(params or {}), "limit": "200"}
    items: list[dict] = []
    offset = 0
    while len(items) < max_items:
        try:
            r = await client.get(
                url, params={**base, "offset": str(offset)},
                headers=headers, timeout=15,
            )
            if r.status_code != 200:
                break
            body = r.json()
            page = body.get("data", [])
            if not page:
                break
            items.extend(page)
            total = body.get("pagination", {}).get("total", 0)
            offset += len(page)
            if offset >= total:
                break
        except Exception:
            break
    return items


async def _fetch_bom(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    sku: str,
) -> list[dict]:
    """Return the flat list of component rows for a finished-good SKU.

    The API wraps components: {"data":[{"sku":…,"components":[{raw_sku:…}]}]}.
    We unwrap and return the inner components list directly.
    """
    try:
        r = await client.get(url, params={"sku": sku, "limit": "50"},
                             headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json().get("data", [])
            # Nested structure: each row has a "components" sub-list
            for item in data:
                if isinstance(item, dict) and "components" in item:
                    return item["components"]
            # Fallback: flat list of component rows
            return data
    except Exception:
        pass
    return []


def _ga(rec: dict, keys: list[str], default=None):
    """Return first non-empty value among candidate field names."""
    for k in keys:
        v = rec.get(k)
        if v is not None and str(v).strip():
            return v
    return default


def _nid(v) -> str | None:
    """Normalize an entity ID string."""
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _short_label(id_: str, desc: str, max_desc: int) -> str:
    desc = (desc or "").strip()
    return f"{id_}\n{desc[:max_desc]}" if desc else id_


async def _build_graph() -> dict:
    """Build the entity graph: customers, products, raw materials, suppliers only.

    Edges are fully aggregated:
      customer → product  ("buys")   — one edge per unique customer/SKU pair from orders
      product  → raw_material ("uses")  — from BOM
      raw_material → supplier ("from") — from inventory supplier_id
    Transaction records (orders, lots, invoices, shipments, calls) are NOT nodes;
    they are the source of aggregated edges only.
    """
    base_url = os.environ.get("MOCK_API_BASE_URL", "").rstrip("/")
    token = os.environ.get("MOCK_API_TOKEN", "")

    if not base_url or not token:
        return {
            "nodes": [], "edges": [], "stats": {},
            "error": "API credentials not configured — deploy backend/.env",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    headers = {"Authorization": f"Bearer {token}"}
    nodes_map: dict[str, dict] = {}
    edges_list: list[dict] = []
    seen_edges: set[tuple] = set()

    def add_node(id_: str, type_: str, label: str, meta: dict,
                 below_min: bool = False) -> None:
        if not id_ or id_ in nodes_map:
            return
        nodes_map[id_] = {
            "id": id_, "type": type_, "label": label,
            "meta": meta, "below_min": below_min,
        }

    def add_edge(frm: str | None, to: str | None, label: str, etype: str) -> None:
        if not frm or not to or frm not in nodes_map or to not in nodes_map:
            return
        key = (frm, to, etype)
        if key in seen_edges:
            return
        seen_edges.add(key)
        edges_list.append({"from": frm, "to": to, "label": label, "type": etype})

    # ── Phase 1: fetch entity data + orders in parallel ───────────────────
    async with httpx.AsyncClient() as client:
        raw_results = await asyncio.gather(
            _fetch_all(client, f"{base_url}/crm/customers",      headers),
            _fetch_all(client, f"{base_url}/erp/inventory",      headers, {"type": "finished_good"}),
            _fetch_all(client, f"{base_url}/erp/inventory",      headers, {"type": "raw_material"}),
            _fetch_all(client, f"{base_url}/erp/suppliers",      headers),
            _fetch_all(client, f"{base_url}/crm/orders",         headers),
            return_exceptions=True,
        )
        (customers, fg_items, rm_items, suppliers, orders) = [
            r if not isinstance(r, Exception) else [] for r in raw_results
        ]
        record_counts = {
            "customers": len(customers), "finished_goods": len(fg_items),
            "raw_materials": len(rm_items), "suppliers": len(suppliers),
            "orders": len(orders),
        }

        # ── Phase 2: create entity nodes ──────────────────────────────────

        for s in suppliers:
            sid = _nid(_ga(s, ["id", "supplier_id"]))
            if not sid:
                continue
            name = _ga(s, ["name"], sid)
            add_node(sid, "supplier", _short_label(sid, name, 20),
                     {"name": name, "category": s.get("category"),
                      "country": s.get("country")})

        for rm in rm_items:
            sku = _nid(_ga(rm, ["sku", "id"]))
            if not sku or not sku.startswith("RAW-"):
                continue
            below = bool(rm.get("below_min"))
            sup_id = _nid(_ga(rm, ["supplier_id", "supplier"]))
            add_node(sku, "raw_material", _short_label(sku, rm.get("description", ""), 18),
                     {"description": rm.get("description"), "on_hand": rm.get("on_hand"),
                      "min_stock": rm.get("min_stock"), "unit": rm.get("unit"),
                      "below_min": below, "supplier_id": sup_id},
                     below_min=below)

        for fg in fg_items:
            sku = _nid(_ga(fg, ["sku", "id"]))
            if not sku or not sku.startswith("PAS-"):
                continue
            below = bool(fg.get("below_min"))
            add_node(sku, "product", _short_label(sku, fg.get("description", ""), 18),
                     {"description": fg.get("description"), "on_hand": fg.get("on_hand"),
                      "min_stock": fg.get("min_stock"), "unit": fg.get("unit"),
                      "below_min": below},
                     below_min=below)

        for c in customers:
            cid = _nid(_ga(c, ["id", "customer_id"]))
            if not cid:
                continue
            name = _ga(c, ["name"], cid)
            add_node(cid, "customer", _short_label(cid, name, 20),
                     {"name": name, "channel": c.get("channel"),
                      "status": c.get("status"), "city": c.get("city")})

        # ── Phase 3: BOM for all finished-good SKUs (parallel) ────────────
        fg_skus = [k for k in nodes_map if k.startswith("PAS-")]
        sem = asyncio.Semaphore(8)

        async def _bom_sem(sku_: str):
            async with sem:
                return sku_, await _fetch_bom(
                    client, f"{base_url}/erp/bom", headers, sku_)

        bom_results = await asyncio.gather(
            *[_bom_sem(s) for s in fg_skus],
            return_exceptions=True,
        )

    # ── Phase 4: build edges ───────────────────────────────────────────────

    # product → raw_material (BOM)
    for result in bom_results:
        if isinstance(result, Exception):
            continue
        sku, bom_data = result
        for comp in (bom_data or []):
            raw_sku = _nid(_ga(comp, ["raw_material_sku", "raw_sku",
                                       "component_sku", "material_sku", "sku"]))
            if raw_sku and raw_sku.startswith("RAW-"):
                add_edge(sku, raw_sku, "uses", "bom")

    # raw_material → supplier (supply)
    for node in list(nodes_map.values()):
        if node["type"] == "raw_material":
            sup_id = node["meta"].get("supplier_id")
            if sup_id:
                add_edge(node["id"], sup_id, "from", "supply")

    # customer → product (buys) — aggregated from order line items
    for ord_ in orders:
        cid = _nid(_ga(ord_, ["customer_id", "customer"]))
        for item in (_ga(ord_, ["items", "line_items", "lines", "products"]) or []):
            if not isinstance(item, dict):
                continue
            item_sku = _nid(_ga(item, ["sku", "product_sku", "finished_sku"]))
            if item_sku:
                add_edge(cid, item_sku, "buys", "buys")

    # ── Phase 5: degree scores ─────────────────────────────────────────────
    degree: dict[str, int] = {}
    for e in edges_list:
        degree[e["from"]] = degree.get(e["from"], 0) + 1
        degree[e["to"]]   = degree.get(e["to"],   0) + 1

    for node in nodes_map.values():
        deg = degree.get(node["id"], 0)
        node["degree"] = deg
        node["priority"] = deg + (20 if node.get("below_min") else 0)

    # ── Phase 6: statistics ────────────────────────────────────────────────
    nodes_by_type: dict[str, int] = {}
    for n in nodes_map.values():
        nodes_by_type[n["type"]] = nodes_by_type.get(n["type"], 0) + 1

    edges_by_type: dict[str, int] = {}
    for e in edges_list:
        edges_by_type[e["type"]] = edges_by_type.get(e["type"], 0) + 1

    print(f"[graph] {len(nodes_map)} nodes | {len(edges_list)} edges")
    print(f"[graph] nodes_by_type: {nodes_by_type}")
    print(f"[graph] edges_by_type: {edges_by_type}")
    print(f"[graph] endpoint_records: {record_counts}")

    return {
        "nodes": list(nodes_map.values()),
        "edges": edges_list,
        "stats": {
            "node_count":   len(nodes_map),
            "edge_count":   len(edges_list),
            "nodes_by_type": nodes_by_type,
            "edges_by_type": edges_by_type,
            "endpoint_record_counts": record_counts,
            "below_min":    sum(1 for n in nodes_map.values() if n.get("below_min")),
            **nodes_by_type,
            "edges": len(edges_list),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/graph-data")
async def graph_data() -> JSONResponse:
    cached = _GRAPH_CACHE.get("data")
    if cached and time.time() - _GRAPH_CACHE.get("ts", 0) < _GRAPH_CACHE_TTL:
        return JSONResponse(content=cached)
    try:
        data = await _build_graph()
    except Exception as exc:
        import traceback
        print(f"[graph] ERROR: {exc}\n{traceback.format_exc()}")
        data = {
            "nodes": [], "edges": [], "stats": {},
            "error": f"Graph build failed: {exc}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    _GRAPH_CACHE["data"] = data
    _GRAPH_CACHE["ts"] = time.time()
    return JSONResponse(content=data)
