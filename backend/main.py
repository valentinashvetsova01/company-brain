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
    base_url = os.environ.get("MOCK_API_BASE_URL", "").rstrip("/")
    token = os.environ.get("MOCK_API_TOKEN", "")

    if not base_url or not token:
        return {
            "nodes": [], "edges": [], "stats": {},
            "error": "API credentials not configured — deploy backend/.env",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    headers = {"Authorization": f"Bearer {token}"}
    nodes_map: dict[str, dict] = {}   # id → node dict
    edges_list: list[dict] = []
    seen_edges: set[tuple] = set()
    skipped: dict[str, int] = {}

    def add_node(id_: str, type_: str, label: str, meta: dict,
                 below_min: bool = False) -> None:
        if not id_ or id_ in nodes_map:
            return
        nodes_map[id_] = {
            "id": id_, "type": type_, "label": label,
            "meta": meta, "below_min": below_min,
        }

    def add_edge(frm: str | None, to: str | None,
                 label: str, etype: str) -> None:
        if not frm or not to:
            skipped[f"null:{etype}"] = skipped.get(f"null:{etype}", 0) + 1
            return
        if frm not in nodes_map:
            skipped[f"no_from:{etype}"] = skipped.get(f"no_from:{etype}", 0) + 1
            return
        if to not in nodes_map:
            skipped[f"no_to:{etype}"] = skipped.get(f"no_to:{etype}", 0) + 1
            return
        key = (frm, to, label)
        if key in seen_edges:
            return
        seen_edges.add(key)
        edges_list.append({"from": frm, "to": to, "label": label, "type": etype})

    # ── Phase 1: fetch all endpoints in parallel ──────────────────────────
    async with httpx.AsyncClient() as client:
        raw_results = await asyncio.gather(
            _fetch_all(client, f"{base_url}/crm/customers",         headers),
            _fetch_all(client, f"{base_url}/crm/opportunities",     headers),
            _fetch_all(client, f"{base_url}/crm/orders",            headers),
            _fetch_all(client, f"{base_url}/crm/invoices",          headers),
            _fetch_all(client, f"{base_url}/erp/production-orders", headers),
            _fetch_all(client, f"{base_url}/erp/inventory",         headers, {"type": "finished_good"}),
            _fetch_all(client, f"{base_url}/erp/inventory",         headers, {"type": "raw_material"}),
            _fetch_all(client, f"{base_url}/erp/suppliers",         headers),
            _fetch_all(client, f"{base_url}/erp/shipments",         headers),
            _fetch_all(client, f"{base_url}/calls",                 headers),
            return_exceptions=True,
        )
        (customers, opportunities, orders, invoices, lots,
         fg_items, rm_items, suppliers, shipments, calls) = [
            r if not isinstance(r, Exception) else [] for r in raw_results
        ]
        record_counts = {
            "customers": len(customers), "opportunities": len(opportunities),
            "orders": len(orders), "invoices": len(invoices),
            "production_orders": len(lots), "finished_goods": len(fg_items),
            "raw_materials": len(rm_items), "suppliers": len(suppliers),
            "shipments": len(shipments), "calls": len(calls),
        }

        # ── Phase 2: create all primary nodes ──────────────────────────────

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

        for opp in opportunities:
            oid = _nid(_ga(opp, ["id", "opportunity_id"]))
            if not oid:
                continue
            cust_id = _nid(_ga(opp, ["customer_id"]))
            stage = opp.get("stage", "")
            add_node(oid, "opportunity", _short_label(oid, stage, 14),
                     {"stage": stage, "value": opp.get("value"),
                      "owner": opp.get("owner"), "customer_id": cust_id})

        for ord_ in orders:
            ord_id = _nid(_ga(ord_, ["id", "order_id"]))
            if not ord_id:
                continue
            status = ord_.get("status", "")
            cust_id = _nid(_ga(ord_, ["customer_id"]))
            add_node(ord_id, "order", _short_label(ord_id, status, 14),
                     {"status": status, "customer_id": cust_id,
                      "date": _ga(ord_, ["date", "created_at", "order_date"]),
                      "items": ord_.get("items") or ord_.get("line_items") or []})

        for lot in lots:
            lot_id = _nid(_ga(lot, ["id", "lot_id", "production_order_id"]))
            if not lot_id:
                continue
            sku = _nid(_ga(lot, ["sku", "product_sku", "finished_product_sku",
                                  "finished_sku"]))
            status = lot.get("status", "")
            ord_id = _nid(_ga(lot, ["order_id", "crm_order_id"]))
            cust_id = _nid(_ga(lot, ["customer_id"]))
            add_node(lot_id, "lot", _short_label(lot_id, sku or status, 16),
                     {"sku": sku, "status": status, "order_id": ord_id,
                      "customer_id": cust_id, "quantity": lot.get("quantity")})

        for call in calls:
            call_id = _nid(_ga(call, ["id", "call_id"]))
            if not call_id:
                continue
            outcome = call.get("outcome", "")
            ctype = call.get("type", "")
            cust_id = _nid(_ga(call, ["customer_id"]))
            lot_id = _nid(_ga(call, ["lot_id", "production_lot_id", "lot"]))
            add_node(call_id, "call", _short_label(call_id, outcome or ctype, 14),
                     {"type": ctype, "outcome": outcome, "customer_id": cust_id,
                      "lot_id": lot_id,
                      "date": _ga(call, ["date", "created_at"])})

        # ── Phase 3: BOM for ALL finished goods (parallel, rate-limited) ─────
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

    # ── Phase 4: build all edges ──────────────────────────────────────────

    # BOM: product → raw_material
    for result in bom_results:
        if isinstance(result, Exception):
            continue
        sku, bom_data = result
        for comp in (bom_data or []):
            raw_sku = _nid(_ga(comp, ["raw_material_sku", "raw_sku",
                                       "component_sku", "material_sku", "sku"]))
            if raw_sku and raw_sku.startswith("RAW-"):
                add_edge(sku, raw_sku, "uses", "bom")

    # Supply: raw_material → supplier
    for node in list(nodes_map.values()):
        if node["type"] == "raw_material":
            sup_id = node["meta"].get("supplier_id")
            if sup_id:
                add_edge(node["id"], sup_id, "from", "supply")

    # Opportunity → customer
    for node in list(nodes_map.values()):
        if node["type"] == "opportunity":
            add_edge(node["id"], node["meta"].get("customer_id"), "for", "crm_opp")

    # Order → customer; order → product (line items)
    for node in list(nodes_map.values()):
        if node["type"] == "order":
            add_edge(node["id"], node["meta"].get("customer_id"), "by", "crm_order")
            for item in (node["meta"].get("items") or []):
                if not isinstance(item, dict):
                    continue
                item_sku = _nid(_ga(item, ["sku", "product_sku", "finished_sku"]))
                if item_sku:
                    add_edge(node["id"], item_sku, "includes", "crm_order")

    # Lot → product, lot → order, lot → customer
    for node in list(nodes_map.values()):
        if node["type"] == "lot":
            add_edge(node["id"], node["meta"].get("sku"),         "produces", "erp_lot")
            add_edge(node["id"], node["meta"].get("order_id"),    "for",      "erp_lot")
            add_edge(node["id"], node["meta"].get("customer_id"), "for",      "erp_lot")

    # Call → customer, call → lot
    for node in list(nodes_map.values()):
        if node["type"] == "call":
            add_edge(node["id"], node["meta"].get("customer_id"), "with", "calls")
            add_edge(node["id"], node["meta"].get("lot_id"),      "re",   "calls")

    # Invoice nodes + edges (conditionally added when connected to known entities)
    for inv in invoices:
        inv_id = _nid(_ga(inv, ["id", "invoice_id"]))
        if not inv_id:
            continue
        cust_id = _nid(_ga(inv, ["customer_id"]))
        ord_id  = _nid(_ga(inv, ["order_id"]))
        cust_ok = cust_id and cust_id in nodes_map
        ord_ok  = ord_id  and ord_id  in nodes_map
        if cust_ok or ord_ok:
            status = inv.get("status", "")
            add_node(inv_id, "invoice", _short_label(inv_id, status, 10),
                     {"status": status, "customer_id": cust_id, "order_id": ord_id,
                      "amount": _ga(inv, ["amount", "total", "value"])})
            if cust_ok:
                add_edge(inv_id, cust_id, "to",  "crm_inv")
            if ord_ok:
                add_edge(inv_id, ord_id,  "for", "crm_inv")

    # Shipment nodes + edges (conditionally added)
    for ship in shipments:
        ship_id = _nid(_ga(ship, ["id", "shipment_id"]))
        if not ship_id:
            continue
        cust_id = _nid(_ga(ship, ["customer_id"]))
        ord_id  = _nid(_ga(ship, ["order_id"]))
        cust_ok = cust_id and cust_id in nodes_map
        ord_ok  = ord_id  and ord_id  in nodes_map
        if cust_ok or ord_ok:
            status = ship.get("status", "")
            add_node(ship_id, "shipment", _short_label(ship_id, status, 12),
                     {"status": status, "customer_id": cust_id, "order_id": ord_id})
            if cust_ok:
                add_edge(ship_id, cust_id, "to",       "erp_ship")
            if ord_ok:
                add_edge(ship_id, ord_id,  "fulfills", "erp_ship")
            for item in (ship.get("items") or ship.get("products") or []):
                if not isinstance(item, dict):
                    continue
                item_sku = _nid(_ga(item, ["sku", "product_sku"]))
                if item_sku:
                    add_edge(ship_id, item_sku, "delivers", "erp_ship")

    # KB documents → SKUs / customers
    kb_dir = Path(__file__).parent / "data" / "kb"
    all_node_ids = set(nodes_map.keys())
    for doc_path in sorted(kb_dir.glob("*.md")):
        doc_id = doc_path.stem
        try:
            content = doc_path.read_text(encoding="utf-8")
        except Exception:
            continue
        title_m = re.search(r"^#\s+(.+)", content, re.MULTILINE)
        title = (title_m.group(1).strip() if title_m else doc_id)[:40]
        add_node(doc_id, "kb_doc", f"{doc_id}\n{title[:22]}",
                 {"title": title, "id": doc_id})
        for target in (set(_SKU_RE.findall(content)) | set(_CUST_RE.findall(content))) & all_node_ids:
            add_edge(doc_id, target, "covers", "kb")

    # ── Phase 5: degree + priority scores (used by frontend readable mode) ──
    degree: dict[str, int] = {}
    for e in edges_list:
        degree[e["from"]] = degree.get(e["from"], 0) + 1
        degree[e["to"]]   = degree.get(e["to"],   0) + 1

    type_base: dict[str, int] = {
        "customer": 12, "product": 10, "raw_material": 8, "supplier": 9,
        "lot": 5, "order": 4, "opportunity": 4, "call": 3,
        "invoice": 2, "shipment": 2, "kb_doc": 6,
    }
    for node in nodes_map.values():
        deg = degree.get(node["id"], 0)
        node["degree"] = deg
        node["priority"] = (
            type_base.get(node["type"], 1)
            + deg * 2
            + (20 if node.get("below_min") else 0)
        )

    # ── Phase 6: statistics and server-side diagnostics ──────────────────
    nodes_by_type: dict[str, int] = {}
    for n in nodes_map.values():
        t = n["type"]
        nodes_by_type[t] = nodes_by_type.get(t, 0) + 1

    edges_by_type: dict[str, int] = {}
    for e in edges_list:
        t = e["type"]
        edges_by_type[t] = edges_by_type.get(t, 0) + 1

    print(f"[graph] {len(nodes_map)} nodes | {len(edges_list)} edges")
    print(f"[graph] nodes_by_type: {nodes_by_type}")
    print(f"[graph] edges_by_type: {edges_by_type}")
    print(f"[graph] endpoint_records: {record_counts}")
    if skipped:
        top = sorted(skipped.items(), key=lambda x: -x[1])[:10]
        print(f"[graph] skipped (top10): {top}")

    return {
        "nodes": list(nodes_map.values()),
        "edges": edges_list,
        "stats": {
            "node_count":   len(nodes_map),
            "edge_count":   len(edges_list),
            "nodes_by_type": nodes_by_type,
            "edges_by_type": edges_by_type,
            "endpoint_record_counts": record_counts,
            "skipped_by_reason": dict(sorted(skipped.items(), key=lambda x: -x[1])),
            "below_min":    sum(1 for n in nodes_map.values() if n.get("below_min")),
            # flat aliases for UI backward compat
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
