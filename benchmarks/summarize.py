#!/usr/bin/env python3
"""Summarize `vllm bench serve` results - every metric in the saved JSONs.

  summarize.py RESULT_DIR     -> five tables (run summary, TTFT, TPOT, ITL, E2E latency),
                                 plus summary.csv (all table columns, one row per run)
                                 and summary_raw.csv (every scalar field in the JSONs)
  summarize.py DIR_A DIR_B    -> side-by-side comparison, ratios are B/A
                                 (e.g. A = main, B = feature)
"""
import csv
import glob
import json
import math
import os
import re
import sys

NAN = float("nan")


# ---------------------------------------------------------------- loading
def load(d):
    rows = {}
    for f in glob.glob(os.path.join(d, "json", "i*_o*_c*.json")):
        m = re.match(r"i(\d+)_o(\d+)_c(\d+)\.json$", os.path.basename(f))
        if not m:
            continue
        try:
            with open(f) as fh:
                j = json.load(fh)
        except Exception:
            continue
        rows[tuple(map(int, m.groups()))] = j
    return rows


def pick(j, keys):
    """First key from `keys` that has a value in the JSON, else None."""
    for k in keys:
        v = j.get(k)
        if v is not None:
            return v
    return None


def fmt(v, nd=2):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return "-" if math.isnan(v) else f"{v:.{nd}f}"
    return str(v)


def table(headers, rows):
    cols = list(zip(*([headers] + rows))) if rows else [[h] for h in headers]
    w = [max(len(str(x)) for x in col) for col in cols]
    sep = "+" + "+".join("-" * (x + 2) for x in w) + "+"
    line = lambda r: "| " + " | ".join(str(x).rjust(w[k]) for k, x in enumerate(r)) + " |"
    return "\n".join([sep, line(headers), sep] + [line(r) for r in rows] + [sep])


# ---------------------------------------------------------------- column definitions
# (header, [candidate JSON keys], decimals). Several candidate keys are listed where
# vllm versions name the field differently; the first one present is used.
RUN_COLS = [
    ("Done",            ["completed"], 0),
    ("Fail",            ["failed"], 0),
    ("Duration (s)",    ["duration"], 2),
    ("In tokens",       ["total_input_tokens"], 0),
    ("Out tokens",      ["total_output_tokens"], 0),
    ("Req/s",           ["request_throughput"], 2),
    ("Out tok/s",       ["output_throughput"], 2),
    ("Peak out tok/s",  ["max_output_tokens_per_s", "peak_output_throughput", "max_output_throughput"], 2),
    ("Tot tok/s",       ["total_token_throughput"], 2),
    ("Peak conc reqs",  ["max_concurrent_requests", "peak_concurrent_requests"], 0),
]
LAT = [("Time to First Token (ms)", "ttft"),
       ("Time per Output Token, excl. 1st token (ms)", "tpot"),
       ("Inter-token Latency (ms)", "itl"),
       ("End-to-end Latency (ms)", "e2el")]


def lat_cols(rows, m):
    """Mean, Median, and every pNN percentile present in the JSONs (p50, p99, ...)."""
    pk = set()
    for j in rows.values():
        for k in j:
            if re.fullmatch(rf"p[\d_.]+_{m}_ms", k):
                pk.add(k)
    num = lambda k: float(k.split("_")[0][1:])
    cols = [("Mean", [f"mean_{m}_ms"], 2), ("Median", [f"median_{m}_ms"], 2)]
    for k in sorted(pk, key=num):
        cols.append((k.split("_")[0].upper(), [k], 2))
    return cols


def build(rows, cols):
    keys = sorted(rows)
    body, raw = [], []
    for k in keys:
        j = rows[k]
        vals = [pick(j, c[1]) for c in cols]
        body.append([f"{k[0]}/{k[1]}", pick(j, ["max_concurrency"]) or k[2]] +
                    [fmt(v, c[2]) for v, c in zip(vals, cols)])
        raw.append(vals)
    return body, raw


# ---------------------------------------------------------------- single-dir report
def single(d):
    rows = load(d)
    if not rows:
        print(f"No results found in {d}/json")
        return
    print(f"Results: {d}")
    print("Latencies in ms, throughput in tokens/s (or req/s)\n")

    sections = [("Run summary", RUN_COLS)] + [(t, lat_cols(rows, m)) for t, m in LAT]
    csv_headers, csv_cols = ["In/Out", "Conc"], []
    keys = sorted(rows)
    for title, cols in sections:
        body, raw = build(rows, cols)
        headers = ["In/Out", "Conc"] + [c[0] for c in cols]
        print(title)
        print(table(headers, body))
        # tell the user if a column could not be found in any JSON
        for ci, c in enumerate(cols):
            if all(r[ci] is None for r in raw):
                print(f"  note: no value for '{c[0]}' in the JSONs (looked for: {', '.join(c[1])})"
                      " - see summary_raw.csv for the exact field names")
        print()
        for ci, c in enumerate(cols):
            csv_headers.append(f"{title.split(' (')[0]}: {c[0]}")
            csv_cols.append([r[ci] for r in raw])

    with open(os.path.join(d, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(csv_headers)
        for ri, k in enumerate(keys):
            w.writerow([f"{k[0]}/{k[1]}", k[2]] + ["" if col[ri] is None else col[ri] for col in csv_cols])

    # every scalar field of every run, nothing dropped
    scalars = {}
    for k in keys:
        scalars[k] = {f: v for f, v in rows[k].items()
                      if isinstance(v, (int, float, str, bool)) or v is None}
    fields = sorted({f for s in scalars.values() for f in s})
    with open(os.path.join(d, "summary_raw.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["in_out", "conc"] + fields)
        for k in keys:
            w.writerow([f"{k[0]}/{k[1]}", k[2]] + [scalars[k].get(f, "") for f in fields])
    print(f"Saved: {os.path.join(d, 'summary.csv')}  and  {os.path.join(d, 'summary_raw.csv')}")


# ---------------------------------------------------------------- two-dir comparison (unchanged)
def ratio(b, a):
    if a is None or b is None or math.isnan(a) or math.isnan(b) or a == 0:
        return "-"
    return f"{b / a:.2f}x"


def compare(da, db):
    ra, rb = load(da), load(db)
    na, nb = os.path.basename(os.path.normpath(da)), os.path.basename(os.path.normpath(db))
    keys = sorted(set(ra) & set(rb))
    if not keys:
        print("No common (in, out, conc) points between the two result dirs")
        return
    headers = ["In/Out", "Conc",
               "TTFT A", "TTFT B", "B/A",
               "TPOT A", "TPOT B", "B/A",
               "ITLp99 A", "ITLp99 B", "B/A",
               "tok/s A", "tok/s B", "B/A"]
    g = lambda j, k: j.get(k, NAN)
    body = []
    for k in keys:
        a, b = ra[k], rb[k]
        body.append([f"{k[0]}/{k[1]}", k[2],
                     fmt(g(a, "mean_ttft_ms"), 1), fmt(g(b, "mean_ttft_ms"), 1), ratio(g(b, "mean_ttft_ms"), g(a, "mean_ttft_ms")),
                     fmt(g(a, "mean_tpot_ms"), 1), fmt(g(b, "mean_tpot_ms"), 1), ratio(g(b, "mean_tpot_ms"), g(a, "mean_tpot_ms")),
                     fmt(g(a, "p99_itl_ms"), 1), fmt(g(b, "p99_itl_ms"), 1), ratio(g(b, "p99_itl_ms"), g(a, "p99_itl_ms")),
                     fmt(g(a, "output_throughput"), 1), fmt(g(b, "output_throughput"), 1),
                     ratio(g(b, "output_throughput"), g(a, "output_throughput"))])
    print(f"A = {na}\nB = {nb}")
    print("TTFT / TPOT / ITL in ms (mean unless noted), tok/s = output throughput. Ratios are B/A.")
    print(table(headers, body))
    out_csv = os.path.join(db, "compare_vs_" + na + ".csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        w.writerows(body)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        single(sys.argv[1])
    elif len(sys.argv) == 3:
        compare(sys.argv[1], sys.argv[2])
    else:
        print(__doc__)
        sys.exit(1)
