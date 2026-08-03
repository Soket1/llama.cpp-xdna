#!/usr/bin/env python3
"""Convert XRT native_trace.csv (VTF format) to Chrome Trace JSON.

XRT's XDP "native_xrt_trace" plugin emits a CSV with sections:
  HEADER / STRUCTURE / MAPPING / EVENTS / DEPENDENCIES

EVENTS row format:
  <event_id>,<paired_start_event_or_0>,<timestamp_ms>,<row_id>,<event_type>,<api_call_id>

api_call_id indexes into MAPPING (e.g. 10 -> xrt::run::run).
paired_start_event is 0 for "start" rows and references the matching start row for "end" rows.
row_id tells us which track to plot on:
  1 = Native XRT API Calls, 2 = Reads (D2H DMA), 3 = Writes (H2D DMA)

Output: Chrome Trace Event Format (JSON object with "traceEvents" array).
Open in chrome://tracing or https://ui.perfetto.dev/.

Usage:
  python xrt_trace_to_chrome.py native_trace.csv > trace.json
  python xrt_trace_to_chrome.py                   # reads ./native_trace.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROW_TO_TRACK = {
    1: ("host", "XRT API (host thread)"),
    2: ("dma_r", "DMA reads (D2H)"),
    3: ("dma_w", "DMA writes (H2D)"),
}


def parse_vtf(path: Path):
    section = None
    mapping: dict[int, str] = {}
    events: list[tuple[int, int, float, int, str, int]] = []

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line in ("HEADER", "STRUCTURE", "MAPPING", "EVENTS", "DEPENDENCIES"):
            section = line
            continue
        if section == "MAPPING":
            # "10,xrt::run::run"
            key, _, name = line.partition(",")
            try:
                mapping[int(key)] = name
            except ValueError:
                pass
        elif section == "EVENTS":
            parts = line.split(",")
            if len(parts) < 6:
                continue
            try:
                eid = int(parts[0])
                pair = int(parts[1])
                ts_ms = float(parts[2])
                row = int(parts[3])
                etype = parts[4]
                api_id = int(parts[5])
            except ValueError:
                continue
            events.append((eid, pair, ts_ms, row, etype, api_id))
    return mapping, events


def to_chrome(mapping, events):
    # Index events by id for pair lookup
    by_id = {e[0]: e for e in events}
    chrome = []

    # tid per logical row, so Reads/Writes/API Calls stack as sibling tracks
    for eid, pair, ts_ms, row, etype, api_id in events:
        if pair != 0:
            # This is an END row; emit a completed (X) event using the paired START.
            start = by_id.get(pair)
            if start is None:
                continue
            s_ts = start[2]
            dur_us = max(0.0, (ts_ms - s_ts) * 1000.0)
            track_id, track_name = ROW_TO_TRACK.get(row, (f"row{row}", f"row {row}"))
            chrome.append({
                "name": mapping.get(api_id, f"api_{api_id}"),
                "cat": track_name,
                "ph": "X",
                "ts": s_ts * 1000.0,  # ms -> us
                "dur": dur_us,
                "pid": 1,
                "tid": row,
                "args": {"event_id": eid, "start_id": pair, "api_id": api_id},
            })
    # Name the tracks (one metadata event per track).
    seen_tids = {e["tid"] for e in chrome}
    for tid in sorted(seen_tids):
        name = ROW_TO_TRACK.get(tid, (None, f"row {tid}"))[1]
        chrome.append({
            "name": "thread_name", "ph": "M", "pid": 1, "tid": tid,
            "args": {"name": name},
        })
    chrome.append({
        "name": "process_name", "ph": "M", "pid": 1, "tid": 0,
        "args": {"name": "XRT native trace"},
    })
    return chrome


def main(argv):
    path = Path(argv[1]) if len(argv) > 1 else Path("native_trace.csv")
    if not path.exists():
        sys.stderr.write(f"error: {path} not found\n")
        return 1
    mapping, events = parse_vtf(path)
    if not events:
        sys.stderr.write("warning: no EVENTS parsed; is this a VTF trace?\n")
    trace = to_chrome(mapping, events)
    sys.stderr.write(
        f"parsed {len(events)} events, {len(mapping)} api names, "
        f"{sum(1 for e in trace if e.get('ph') == 'X')} chrome spans\n"
    )
    json.dump({"traceEvents": trace, "displayTimeUnit": "ms"}, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
