#!/usr/bin/env python3
"""Build replay traces from the agentic-proxy Neon Postgres database.

Reads anonymized Claude Code request data from the Neon DB and generates
trace JSON files compatible with trace_replay_tester.py.

Usage:
    DATABASE_URL="postgres://..." python3 build_traces_from_neon.py \
        --output-dir traces_neon/ \
        --min-requests 10 \
        --anonymize

Requirements:
    pip install psycopg2-binary
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("Error: psycopg2 is required. Install with: pip install psycopg2-binary")
    sys.exit(1)


def connect_db(database_url: str):
    """Connect to the Neon Postgres database."""
    conn = psycopg2.connect(database_url)
    conn.set_session(readonly=True)
    return conn


def fetch_sessions(conn, min_requests: int, session_ids: list = None, max_traces: int = None):
    """Fetch session IDs that meet the criteria."""
    cur = conn.cursor()

    if session_ids:
        placeholders = ",".join(["%s"] * len(session_ids))
        cur.execute(f"""
            SELECT session_id, COUNT(*) as req_count
            FROM requests
            WHERE session_id IN ({placeholders})
              AND is_anonymized = true
              AND endpoint = '/v1/messages'
              AND response_status_code = 200
            GROUP BY session_id
            HAVING COUNT(*) >= %s
            ORDER BY MIN(timestamp)
        """, session_ids + [min_requests])
    else:
        cur.execute("""
            SELECT session_id, COUNT(*) as req_count
            FROM requests
            WHERE is_anonymized = true
              AND endpoint = '/v1/messages'
              AND response_status_code = 200
            GROUP BY session_id
            HAVING COUNT(*) >= %s
            ORDER BY MIN(timestamp)
        """, (min_requests,))

    sessions = cur.fetchall()
    cur.close()

    if max_traces and len(sessions) > max_traces:
        sessions = sessions[:max_traces]

    return sessions


def fetch_session_requests(conn, session_id: str):
    """Fetch all requests for a session, ordered by timestamp."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT
            id,
            timestamp,
            model,
            input_tokens,
            output_tokens,
            cache_read_input_tokens,
            cache_write_tokens,
            duration_ms,
            is_streaming,
            hash_ids,
            ttft_ms,
            tpot_ms,
            response_body->'body'->>'stop_reason' as stop_reason
        FROM requests
        WHERE session_id = %s
          AND is_anonymized = true
          AND endpoint = '/v1/messages'
          AND response_status_code = 200
        ORDER BY timestamp
    """, (session_id,))
    rows = cur.fetchall()
    cur.close()
    return rows


def build_hash_id_map(all_sessions_requests: list) -> dict:
    """Build a global hex-string → sequential-integer mapping for hash_ids.

    Processes all sessions to ensure the same hex hash gets the same integer
    ID across traces, preserving cross-session cache sharing semantics.
    """
    hash_map = {}
    next_id = 1

    for requests in all_sessions_requests:
        for req in requests:
            if req["hash_ids"]:
                for hex_hash in req["hash_ids"]:
                    if hex_hash not in hash_map:
                        hash_map[hex_hash] = next_id
                        next_id += 1

    return hash_map


def infer_input_types(prev_stop_reason: str, is_first: bool) -> list:
    """Infer input_types from the previous request's stop_reason."""
    if is_first:
        return ["text"]
    if prev_stop_reason == "tool_use":
        return ["tool_result"]
    if prev_stop_reason == "end_turn":
        return ["text"]
    return ["text"]


def build_trace(session_id: str, requests: list, hash_map: dict,
                block_size: int, trace_number: int, anonymize: bool,
                tool_tokens: int = 0, system_tokens: int = 0) -> dict:
    """Build a trace dict from a list of DB request rows."""

    if not requests:
        return None

    # Determine conversation start time
    first_ts = requests[0]["timestamp"]
    if first_ts.tzinfo is None:
        first_ts = first_ts.replace(tzinfo=timezone.utc)

    # Collect models used
    models = sorted(set(r["model"] for r in requests if r["model"]))

    # Build request records
    trace_requests = []
    prev_stop_reason = None
    prev_end_time = None

    for i, req in enumerate(requests):
        ts = req["timestamp"]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        # Relative timestamp from conversation start
        t = round((ts - first_ts).total_seconds(), 1)

        # Total input tokens (new + cached + cache-write)
        input_tokens = (
            (req["input_tokens"] or 0) +
            (req["cache_read_input_tokens"] or 0) +
            (req["cache_write_tokens"] or 0)
        )
        output_tokens = req["output_tokens"] or 0

        # Convert hash_ids from hex strings to integers
        int_hash_ids = []
        if req["hash_ids"]:
            for hex_hash in req["hash_ids"]:
                int_hash_ids.append(hash_map[hex_hash])

        # Stop reason
        stop = req["stop_reason"] or ""
        if stop not in ("tool_use", "end_turn", "max_tokens", "stop_sequence"):
            stop = ""

        # API time (duration_ms → seconds)
        api_time = round(req["duration_ms"] / 1000.0, 2) if req["duration_ms"] else None

        # Think time: gap between previous response completion and this request
        think_time = None
        if i == 0:
            think_time = 0.0
        elif prev_end_time is not None:
            think_time = max(0.0, round((ts - prev_end_time).total_seconds(), 2))

        # Track when this request's response completed
        if req["duration_ms"]:
            prev_end_time = ts + timedelta(milliseconds=req["duration_ms"])
        else:
            prev_end_time = ts

        # Input types
        input_types = infer_input_types(prev_stop_reason, is_first=(i == 0))
        prev_stop_reason = req["stop_reason"]

        # Build request record
        record = {
            "t": t,
            "type": "s",
            "model": req["model"] or "",
            "in": input_tokens,
            "out": output_tokens,
            "hash_ids": int_hash_ids,
            "input_types": input_types,
            "output_types": [],
            "stop": stop,
        }
        if api_time is not None:
            record["api_time"] = api_time
        if think_time is not None:
            record["think_time"] = think_time

        trace_requests.append(record)

    # Build trace
    trace_id = f"trace_{trace_number:04d}" if anonymize else session_id
    trace = {
        "id": trace_id,
        "models": models,
        "block_size": block_size,
        "hash_id_scope": "global",
        "tool_tokens": tool_tokens,
        "system_tokens": system_tokens,
        "requests": trace_requests,
    }

    return trace


def write_trace(trace: dict, output_dir: Path):
    """Write a trace to a JSON file."""
    filepath = output_dir / f"{trace['id']}.json"
    with open(filepath, "w") as f:
        json.dump(trace, f, indent=2)
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Build replay traces from the agentic-proxy Neon database"
    )
    parser.add_argument("--output-dir", type=str, default="./traces/",
                        help="Output directory for trace JSON files (default: ./traces/)")
    parser.add_argument("--min-requests", type=int, default=10,
                        help="Minimum requests per session to include (default: 10)")
    parser.add_argument("--block-size", type=int, default=64,
                        help="Block size for trace metadata (default: 64)")
    parser.add_argument("--anonymize", action="store_true", default=False,
                        help="Replace session IDs with trace_NNNN")
    parser.add_argument("--session-ids", type=str, nargs="*", default=None,
                        help="Export only specific session IDs")
    parser.add_argument("--max-traces", type=int, default=None,
                        help="Maximum number of traces to export")
    parser.add_argument("--tool-tokens", type=int, default=0,
                        help="Tool definition token count for trace metadata (default: 0)")
    parser.add_argument("--system-tokens", type=int, default=0,
                        help="System prompt token count for trace metadata (default: 0)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Parallel workers for DB fetch + trace build/write (default: 8)")
    args = parser.parse_args()

    # Connect to DB
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("Error: DATABASE_URL environment variable is required")
        print("Example: DATABASE_URL='postgres://user:pass@host/db' python3 build_traces_from_neon.py")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to database...")
    conn = connect_db(database_url)

    # Fetch qualifying sessions
    print(f"Finding sessions with >= {args.min_requests} requests...")
    sessions = fetch_sessions(conn, args.min_requests, args.session_ids, args.max_traces)
    print(f"Found {len(sessions)} sessions")

    if not sessions:
        print("No sessions match criteria. Exiting.")
        conn.close()
        return

    conn.close()

    # Fetch all requests in parallel — each worker gets its own connection
    print(f"Fetching requests with {args.num_workers} workers...")
    all_requests = [None] * len(sessions)

    def _fetch(idx_and_session):
        idx, (session_id, _) = idx_and_session
        worker_conn = connect_db(database_url)
        try:
            return idx, session_id, fetch_session_requests(worker_conn, session_id)
        finally:
            worker_conn.close()

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = [pool.submit(_fetch, item) for item in enumerate(sessions)]
        for done, fut in enumerate(as_completed(futures), 1):
            idx, session_id, requests = fut.result()
            all_requests[idx] = requests
            print(f"  [{done}/{len(sessions)}] {session_id[:12]}... : {len(requests)} requests")

    # Build global hash_id mapping — must be serial since it assigns sequential IDs
    print(f"Building global hash_id mapping...")
    hash_map = build_hash_id_map(all_requests)
    print(f"  {len(hash_map)} unique hash_ids across all sessions")

    # Build and write traces in parallel
    print(f"Building traces with {args.num_workers} workers...")
    total_requests = 0

    def _build_and_write(i_sess_reqs):
        i, (session_id, _), requests = i_sess_reqs
        trace = build_trace(
            session_id=session_id,
            requests=requests,
            hash_map=hash_map,
            block_size=args.block_size,
            trace_number=i + 1,
            anonymize=args.anonymize,
            tool_tokens=args.tool_tokens,
            system_tokens=args.system_tokens,
        )
        if not trace:
            return None
        filepath = write_trace(trace, output_dir)
        return trace["id"], len(trace["requests"]), filepath

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        items = [(i, sess, reqs) for i, (sess, reqs) in enumerate(zip(sessions, all_requests))]
        for result in pool.map(_build_and_write, items):
            if result is None:
                continue
            trace_id, n_reqs, filepath = result
            total_requests += n_reqs
            print(f"  {trace_id}: {n_reqs} requests → {filepath}")

    # Summary
    print(f"\nDone!")
    print(f"  Traces: {len(sessions)}")
    print(f"  Total requests: {total_requests}")
    print(f"  Unique hash_ids: {len(hash_map)}")
    print(f"  Output: {output_dir}/")


if __name__ == "__main__":
    main()
