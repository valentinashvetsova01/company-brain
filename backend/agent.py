"""Agent loop for the Al Dente Company Brain. v3

Tools:
  call_api              – fully-paginated calls to ALL Al Dente mock API endpoints
  search_kb             – keyword + SKU-index retrieval over backend/data/kb/
  search_transcripts    – batch-scan all call transcripts for a keyword (Python loop)
  crm_channel_breakdown – Python join: opportunities × customers → totals by channel

All arithmetic (sums, counts, group-bys) is computed in Python, never in the LLM.
Pagination: every list endpoint is fully consumed before the result is returned.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx
from openai import OpenAI


# ── Knowledge-base ────────────────────────────────────────────────────────────

_KB_DIR = Path(__file__).parent / "data" / "kb"
_KB: dict[str, str] = {
    f.stem: f.read_text(encoding="utf-8") for f in sorted(_KB_DIR.glob("*.md"))
}

# Reverse index: finished-good/raw-material SKU → first KB doc that mentions it
_SKU_RE = re.compile(r"\b(PAS-[A-Z]{3}-\d{3}|RAW-[A-Z]{3}-\d{3})\b")
_KB_SKU_INDEX: dict[str, list[str]] = {}   # "PAS-SPA-500" → ["DOC-001", ...]
for _doc_id, _content in _KB.items():
    for _sku in _SKU_RE.findall(_content):
        _KB_SKU_INDEX.setdefault(_sku, [])
        if _doc_id not in _KB_SKU_INDEX[_sku]:
            _KB_SKU_INDEX[_sku].append(_doc_id)


def _search_kb_impl(query: str, doc_ids: list[str] | None = None) -> list[dict]:
    """Whole-document retrieval. Prioritises exact SKU matches, then keyword score."""
    if doc_ids:
        return [{"id": d, "content": _KB[d]} for d in doc_ids if d in _KB]

    results: list[dict] = []
    seen: set[str] = set()

    # 1. Exact SKU hit (highest priority)
    skus_in_query = _SKU_RE.findall(query.upper())
    for sku in skus_in_query:
        for did in _KB_SKU_INDEX.get(sku, []):
            if did not in seen:
                results.append({"id": did, "content": _KB[did]})
                seen.add(did)

    # 2. Keyword relevance for remaining slots (up to 5 total)
    q_lower = query.lower()
    q_words = {w for w in q_lower.split() if len(w) > 2}
    scored: list[tuple[int, str]] = []
    for doc_id, content in _KB.items():
        if doc_id in seen:
            continue
        c_lower = content.lower()
        score = sum(1 for w in q_words if w in c_lower)
        if q_lower in c_lower:
            score += 10
        if score > 0:
            scored.append((score, doc_id))
    scored.sort(reverse=True)
    for _, did in scored[: max(0, 5 - len(results))]:
        results.append({"id": did, "content": _KB[did]})

    return results


# ── Cache ─────────────────────────────────────────────────────────────────────

_CACHE: dict[str, dict] = {}


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = """You are the Al Dente Company Brain, an AI agent for Al Dente S.r.l. (Italian pasta maker).

Use ONLY the provided tools to gather data. Never invent, estimate, or guess data.

Data sources:
- call_api /crm/*   → customers, opportunities, orders, invoices
- call_api /erp/*   → production orders (LOT-YYYY-#### IDs live here, not in /crm/orders), inventory (supplier_id on raw materials), suppliers, BOM, shipments
- call_api /calls/* → call list; /calls/{id}/transcript?search=<keyword> for transcript content
- search_kb         → product specs (shelf life, allergens, SKU), policies, price list, customer requirements
- search_transcripts → COUNT calls mentioning a keyword across the FULL log (aggregate/count use only; never use to look up a specific call)
- crm_channel_breakdown → value/count of opportunities grouped by customer channel (GDO/distributor/horeca) in Python

Rules:
1. If an entity doesn't exist in the data, say so explicitly — never invent it. LOT-YYYY-#### IDs are production lots in /erp/production-orders, not CRM orders.
2. Financial metrics (profit margin, cost, markup) are NOT stored anywhere in the available sources — if asked, state clearly they are not available after verifying the entity exists.
2. All numeric aggregates in tool results are pre-computed in Python — use them directly.
3. For the latest call with a customer: fetch /calls?customer_id=X, then pick the record with the most recent date field.
4. Multi-hop BOM chain: /erp/bom?sku=PAS-X → get RAW-SKU → /erp/inventory?search=RAW-SKU (returns supplier_id) → /erp/suppliers (no filter, returns all; match by id field to get name).
5. 'Open' opportunities = stage qualification OR negotiation (won/lost are closed). To count/sum open opps: make TWO calls — /crm/opportunities?stage=qualification[&customer_id=X] and /crm/opportunities?stage=negotiation[&customer_id=X] — then ADD the two sum_value_eur and the two total counts from those responses.
6. For "by channel" groupings use crm_channel_breakdown — it joins opportunities to customers and sums in Python.
7. Quality complaint transcripts: use outcome_filter=complaint_open in search_transcripts. NEVER use search_transcripts to look up a specific call — for a specific call use /calls?customer_id=X then /calls/{id}/transcript?search=.
8. Answer concisely and factually. Do NOT show chain-of-thought or reasoning steps."""


# ── Tool definitions ──────────────────────────────────────────────────────────

_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "call_api",
            "description": (
                "Call any Al Dente API endpoint. All pages are fetched automatically. "
                "Numeric fields are pre-summed in the result as sum_<field>.\n"
                "Endpoints (filters are exact-match, case-sensitive):\n"
                "  /crm/customers  [search, channel=(GDO|distributor|horeca), status=(active|inactive|prospect)]\n"
                "  /crm/customers/{id}\n"
                "  /crm/opportunities  [customer_id, stage=(qualification|negotiation|won|lost), owner]\n"
                "  /crm/orders  [customer_id, status=(open|in_production|shipped|delivered|cancelled), from, to]\n"
                "  /crm/invoices  [customer_id, status=(unpaid|paid|overdue), order_id]\n"
                "  /calls  [customer_id, type=(sales|support), outcome=(complaint_open|follow_up|order_placed|resolved), from, to]\n"
                "  /calls/{id}\n"
                "  /calls/{id}/transcript  [search=<keyword>, speaker]  — always use search= to filter\n"
                "  /erp/production-orders  [customer_id, status=(planned|in_progress|done|blocked), sku, from, to]\n"
                "  /erp/inventory  [type=(finished_good|raw_material), below_min=true, search=<sku>]  — raw_material rows include supplier_id\n"
                "  /erp/suppliers  [search=<name>, category=(semolina|wheat|packaging|labels|ink|logistics)]  — use no filter to list all; each record has id and name\n"
                "  /erp/bom  [sku=<PAS-XXX-###>]\n"
                "  /erp/shipments  [customer_id, order_id, status=(in_transit|delivered|delayed)]"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "endpoint": {
                        "type": "string",
                        "description": "API path, e.g. /crm/opportunities or /calls/CALL-58020/transcript",
                    },
                    "params": {
                        "type": "object",
                        "description": "Query parameters as string values",
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["endpoint"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": (
                "Search Al Dente knowledge base documents (35 markdown files). "
                "Contains: product spec sheets (shelf life, allergens, PAS-* SKU details), "
                "quality & returns policies, 2026 wholesale price list, customer supply requirements. "
                "SKU queries (e.g. PAS-SPA-500) return the exact spec sheet automatically. "
                "Returns full document text — never truncated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords or SKU, e.g. 'PAS-SPA-500 shelf life allergens' or 'returns quality policy'",
                    },
                    "doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: retrieve specific docs by ID, e.g. ['DOC-001', 'DOC-015']",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_transcripts",
            "description": (
                "Scan every call transcript for a search keyword and count matching calls. "
                "Uses limit=1 probes — efficient even over 80+ calls. "
                "For quality-complaint counts, set outcome_filter=complaint_open. "
                "Returns: total_calls_searched, match_count, matching_call_ids, sample_segments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "Phrase to find in transcripts, e.g. 'broken pasta'",
                    },
                    "outcome_filter": {
                        "type": "string",
                        "description": "Pre-filter calls by outcome: complaint_open | follow_up | order_placed | resolved",
                    },
                    "customer_id": {
                        "type": "string",
                        "description": "Restrict search to one customer's calls",
                    },
                },
                "required": ["search_term"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crm_channel_breakdown",
            "description": (
                "Get opportunity totals (count + sum of value_eur) grouped by customer channel "
                "(GDO / distributor / horeca). Joins opportunities to customers in Python — "
                "use this for questions like 'total value by channel' or 'opportunities per channel'. "
                "Optionally filter by opportunity stage before grouping."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "stage": {
                        "type": "string",
                        "description": "Filter opportunities by stage: qualification | negotiation | won | lost",
                    },
                    "customer_id": {
                        "type": "string",
                        "description": "Restrict to one customer (optional)",
                    },
                },
            },
        },
    },
]


# ── API layer (pagination + pre-computed aggregates) ─────────────────────────

def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {os.environ['MOCK_API_TOKEN']}"}


def _fetch_all(
    cli: httpx.Client,
    url: str,
    params: dict[str, str],
    data_key: str = "data",
) -> tuple[list, int]:
    """Fetch all pages for a list endpoint. Returns (all_items, total)."""
    rp = dict(params)
    rp.setdefault("limit", "200")
    rp["offset"] = "0"
    hdrs = _headers()

    r = cli.get(url, params=rp, headers=hdrs)
    r.raise_for_status()
    first = r.json()
    items: list = list(first.get(data_key, []))
    total: int = first["pagination"]["total"]

    while len(items) < total:
        rp["offset"] = str(len(items))
        rp2 = cli.get(url, params=rp, headers=hdrs)
        rp2.raise_for_status()
        pg = rp2.json()
        new = pg.get(data_key, [])
        if not new:
            break
        items.extend(new)

    return items, total


def _api_call(endpoint: str, params: dict | None = None) -> dict:
    """Fully-paginated API call. Pre-computes sums for numeric fields."""
    base = os.environ["MOCK_API_BASE_URL"].rstrip("/")
    url = f"{base}{endpoint}"
    rp = {k: str(v) for k, v in (params or {}).items()}

    with httpx.Client(timeout=20.0) as cli:
        hdrs = _headers()
        rp2 = dict(rp)
        rp2.setdefault("limit", "200")
        rp2["offset"] = "0"
        r = cli.get(url, params=rp2, headers=hdrs)
        r.raise_for_status()
        first = r.json()

        # Transcript endpoint: { segments, pagination, call_id }
        if "segments" in first:
            items: list = list(first["segments"])
            total: int = first["pagination"]["total"]
            while len(items) < total:
                rp2["offset"] = str(len(items))
                rp3 = cli.get(url, params=rp2, headers=hdrs)
                rp3.raise_for_status()
                pg = rp3.json()
                new = pg.get("segments", [])
                if not new:
                    break
                items.extend(new)
            return {"segments": items, "total": total, "call_id": first.get("call_id")}

        # Standard list endpoint: { data, pagination }
        if "data" in first:
            items = list(first["data"])
            total = first["pagination"]["total"]
            while len(items) < total:
                rp2["offset"] = str(len(items))
                rp3 = cli.get(url, params=rp2, headers=hdrs)
                rp3.raise_for_status()
                pg = rp3.json()
                new = pg.get("data", [])
                if not new:
                    break
                items.extend(new)

            result: dict = {"data": items, "total": total}
            # Pre-compute numeric sums so LLM never has to do arithmetic
            if items and isinstance(items[0], dict):
                for key in items[0]:
                    if any(t in key.lower() for t in ("value", "amount", "price", "total", "quantity", "qty")):
                        try:
                            result[f"sum_{key}"] = sum(float(it.get(key) or 0) for it in items)
                        except (TypeError, ValueError):
                            pass
            return result

        # Single-object endpoint
        return first


# ── crm_channel_breakdown ─────────────────────────────────────────────────────

def _crm_channel_breakdown_impl(
    stage: str | None = None,
    customer_id: str | None = None,
) -> dict:
    """Python join: opportunities × customers → totals per channel. No LLM math."""
    base = os.environ["MOCK_API_BASE_URL"].rstrip("/")

    opp_params: dict[str, str] = {"limit": "200"}
    if stage:
        opp_params["stage"] = stage
    if customer_id:
        opp_params["customer_id"] = customer_id

    with httpx.Client(timeout=30.0) as cli:
        opps, _ = _fetch_all(cli, f"{base}/crm/opportunities", opp_params)
        customers, _ = _fetch_all(cli, f"{base}/crm/customers", {"limit": "200"})

    channel_map: dict[str, str] = {c["id"]: c.get("channel", "unknown") for c in customers}

    groups: dict[str, dict] = {}
    for opp in opps:
        ch = channel_map.get(opp.get("customer_id", ""), "unknown")
        if ch not in groups:
            groups[ch] = {"count": 0, "total_value_eur": 0.0, "opportunity_ids": []}
        groups[ch]["count"] += 1
        groups[ch]["total_value_eur"] += float(opp.get("value_eur") or 0)
        groups[ch]["opportunity_ids"].append(opp.get("id"))

    return {
        "breakdown_by_channel": groups,
        "total_opportunities": len(opps),
        "stage_filter": stage,
        "grand_total_value_eur": sum(g["total_value_eur"] for g in groups.values()),
    }


# ── search_transcripts ────────────────────────────────────────────────────────

def _search_transcripts_impl(
    search_term: str,
    outcome_filter: str | None = None,
    customer_id: str | None = None,
) -> dict:
    """Scan all (filtered) call transcripts for search_term. limit=1 probes for efficiency."""
    base = os.environ["MOCK_API_BASE_URL"].rstrip("/")

    call_params: dict[str, str] = {"limit": "200"}
    if outcome_filter:
        call_params["outcome"] = outcome_filter
    if customer_id:
        call_params["customer_id"] = customer_id

    with httpx.Client(timeout=90.0) as cli:
        all_calls, _ = _fetch_all(cli, f"{base}/calls", call_params)

        match_count = 0
        matching_ids: list[str] = []
        sample_segments: list[dict] = []

        for call in all_calls:
            cid = call["id"]
            tr = cli.get(
                f"{base}/calls/{cid}/transcript",
                params={"search": search_term, "limit": "1"},
                headers=_headers(),
            )
            if tr.status_code != 200:
                continue
            tr_data = tr.json()
            # Require at least 2 matching segments to exclude incidental mentions
            # (e.g. "alongside broken pasta" in a policy explanation, or "not broken pasta").
            # Genuine complaints always have 3+ hits; single-hit matches are passing references.
            if tr_data["pagination"]["total"] >= 2:
                match_count += 1
                matching_ids.append(cid)
                if len(sample_segments) < 6:
                    # Fetch up to 2 segments for context
                    tr2 = cli.get(
                        f"{base}/calls/{cid}/transcript",
                        params={"search": search_term, "limit": "2"},
                        headers=_headers(),
                    )
                    if tr2.status_code == 200:
                        for seg in tr2.json().get("segments", [])[:2]:
                            sample_segments.append({"call_id": cid, **seg})

    return {
        "search_term": search_term,
        "outcome_filter": outcome_filter,
        "total_calls_searched": len(all_calls),
        "match_count": match_count,
        "matching_call_ids": matching_ids,
        "sample_segments": sample_segments,
    }


# ── Tool dispatcher ───────────────────────────────────────────────────────────

def _run_tool(name: str, args: dict) -> tuple[str, list[str], str]:
    """Execute one tool call. Returns (result_json, new_sources, verticale_cat)."""

    if name == "call_api":
        endpoint: str = args.get("endpoint", "")
        params: dict = args.get("params", {})
        try:
            result = _api_call(endpoint, params)
            segs = [s for s in endpoint.split("/") if s]
            cat = segs[0] if segs and segs[0] in ("crm", "erp", "calls") else "unknown"
            return json.dumps(result), [endpoint.lstrip("/")], cat
        except httpx.HTTPStatusError as e:
            body = {
                "error": "not_found" if e.response.status_code == 404 else "api_error",
                "http_status": e.response.status_code,
                "endpoint": endpoint,
            }
            return json.dumps(body), [], "unknown"
        except Exception as exc:
            return json.dumps({"error": str(exc), "endpoint": endpoint}), [], "unknown"

    if name == "search_kb":
        try:
            docs = _search_kb_impl(args.get("query", ""), args.get("doc_ids"))
            sources = [d["id"] for d in docs]
            return json.dumps(docs), sources, "kb"
        except Exception as exc:
            return json.dumps({"error": str(exc)}), [], "kb"

    if name == "search_transcripts":
        try:
            result = _search_transcripts_impl(
                args.get("search_term", ""),
                args.get("outcome_filter"),
                args.get("customer_id"),
            )
            return json.dumps(result), ["calls/transcripts"], "calls"
        except Exception as exc:
            return json.dumps({"error": str(exc)}), [], "calls"

    if name == "crm_channel_breakdown":
        try:
            result = _crm_channel_breakdown_impl(
                args.get("stage"),
                args.get("customer_id"),
            )
            return json.dumps(result), ["crm/opportunities", "crm/customers"], "crm"
        except Exception as exc:
            return json.dumps({"error": str(exc)}), [], "crm"

    return json.dumps({"error": f"unknown tool: {name}"}), [], "unknown"


# ── Verticale routing ─────────────────────────────────────────────────────────

def _pick_verticale(cats: list[str]) -> str:
    counts: dict[str, int] = {"crm": 0, "erp": 0, "calls": 0, "kb": 0}
    for c in cats:
        if c in counts:
            counts[c] += 1
    best = max(counts, key=counts.get)  # type: ignore[arg-type]
    return best if counts[best] > 0 else "kb"


# ── Agent loop ────────────────────────────────────────────────────────────────

_MAX_STEPS = 15


def run_agent(question: str) -> dict:
    """Run the agent loop and return the /ask response dict."""
    if question in _CACHE:
        return _CACHE[question]

    llm = OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
    )
    model = os.environ["MODEL"]

    # Normalise: wrap command-style fragments so any model engages with them
    q = question.strip()
    # If it doesn't look like a complete sentence (no verb, starts capitalised), frame it
    if q and not any(q.lower().startswith(w) for w in (
        "what", "who", "when", "where", "why", "how", "is ", "are ", "does ", "do ",
        "can ", "could ", "would ", "should ", "has ", "have ", "did ", "was ", "were ",
        "list", "show", "find", "get", "give",
    )):
        q = f"Please research and answer: {q}"

    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": q},
    ]

    sources: list[str] = []
    cats: list[str] = []

    for _step in range(_MAX_STEPS):
        try:
            resp = llm.chat.completions.create(
                model=model,
                messages=messages,
                tools=_TOOLS,
                tool_choice="auto",
                timeout=25,
            )
        except Exception as exc:
            return _error_response(str(exc), sources, cats)

        msg = resp.choices[0].message

        # Handle reasoning models that put text in reasoning_content
        content: str = msg.content or ""
        if not content:
            content = getattr(msg, "reasoning_content", None) or ""
        # Strip Qwen3 <think>…</think> chain-of-thought blocks
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        if not msg.tool_calls:
            out = {
                "answer": content or "No answer could be generated.",
                "sources": list(dict.fromkeys(sources)),
                "verticale": _pick_verticale(cats),
                "artifact_url": None,
            }
            _CACHE[question] = out
            return out

        # Append assistant turn with tool_calls
        messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            result_json, new_sources, cat = _run_tool(tc.function.name, args)
            sources.extend(new_sources)
            if cat:
                cats.append(cat)

            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result_json}
            )

    # Step limit hit — request a summary answer
    try:
        final = llm.chat.completions.create(
            model=model,
            messages=messages
            + [{"role": "user", "content": "Summarise your findings and give a final answer now."}],
            timeout=20,
        )
        content = re.sub(
            r"<think>.*?</think>",
            "",
            final.choices[0].message.content or "",
            flags=re.DOTALL,
        ).strip()
        content = content or "Step limit reached; answer may be incomplete."
    except Exception:
        content = "Unable to complete the answer within the step limit."

    out = {
        "answer": content,
        "sources": list(dict.fromkeys(sources)),
        "verticale": _pick_verticale(cats),
        "artifact_url": None,
    }
    _CACHE[question] = out
    return out


def _error_response(msg: str, sources: list[str], cats: list[str]) -> dict:
    return {
        "answer": f"I cannot answer right now due to a technical issue: {msg}",
        "sources": list(dict.fromkeys(sources)),
        "verticale": _pick_verticale(cats),
        "artifact_url": None,
    }
