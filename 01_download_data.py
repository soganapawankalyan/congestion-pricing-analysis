#!/usr/bin/env python3
"""
Project 3A - NYC congestion pricing: download compact, aggregated data from NY Open Data.

Uses only the Python standard library (no installs needed).
The raw tables are huge (13.8M and 6.9M rows), so the grouping is done on the server
with SoQL and only daily / 10-minute summaries come back (a few hundred thousand rows).

Writes to data/raw/:
  bt_daily.csv        B&T crossings per day x facility x direction x vehicle category
                      (dataset ebfx-2m7v, "MTA Bridges and Tunnels Hourly Crossings: Beginning 2019")
  crz_daily.csv       Zone entries per day x detection group x vehicle class
                      (dataset t6yz-b64h, "MTA Congestion Relief Zone Vehicle Entries: Beginning 2025")
  crz_10min.csv       Zone entries per day x 10-minute slot x vehicle class (all entry points summed)
  _summary.txt        Coverage report: dates, facilities, categories, missing days. Paste this back.

Run:  python3 01_download_data.py          (takes a few minutes; safe to re-run)
"""
import csv, datetime as dt, io, os, sys, time, urllib.error, urllib.parse, urllib.request
from collections import defaultdict

BASE = os.environ.get("SODA_BASE", "https://data.ny.gov/resource/")
PAGE = int(os.environ.get("SODA_PAGE", "50000"))
OUT = "data/raw"
BT, CRZ = "ebfx-2m7v", "t6yz-b64h"
HEADERS = {"User-Agent": "portfolio-analysis/1.0", "Accept": "text/csv"}


def get(dataset, params, tries=4):
    url = f"{BASE}{dataset}.csv?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote, safe="(),*")
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:400]
            if e.code in (400, 404):       # our query is wrong: retrying will not help
                sys.exit(f"\nHTTP {e.code} from the server. Paste this message back:\n  {url}\n  {body}")
            print(f"  HTTP {e.code}, retrying ({k+1}/{tries})...")
        except Exception as e:             # network blip
            print(f"  {type(e).__name__}: {e}, retrying ({k+1}/{tries})...")
        time.sleep(3 * (k + 1))
    sys.exit("Giving up after repeated errors. Check your internet connection and re-run.")


def fetch_chunk(dataset, params, writer, state):
    """Page through one query, writing rows. state = {'header': None, 'n': 0}."""
    offset = 0
    while True:
        text = get(dataset, dict(params, **{"$limit": PAGE, "$offset": offset}))
        rows = list(csv.reader(io.StringIO(text)))
        if not rows:
            break
        hdr, data = rows[0], rows[1:]
        if state["header"] is None:
            state["header"] = hdr
            writer.writerow(hdr)
        elif hdr != state["header"]:
            sys.exit(f"Column mismatch between pages: {hdr} vs {state['header']}")
        writer.writerows(data)
        state["n"] += len(data)
        if len(data) < PAGE:
            break
        offset += PAGE
        time.sleep(0.4)


def pull(name, dataset, select, group, order, date_col, years, extra_where=""):
    path = f"{OUT}/{name}.csv"
    state = {"header": None, "n": 0}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for y in years:
            where = f"{date_col} >= '{y}-01-01T00:00:00' AND {date_col} < '{y+1}-01-01T00:00:00'" + extra_where
            before = state["n"]
            fetch_chunk(dataset, {"$select": select, "$where": where, "$group": group, "$order": order}, w, state)
            print(f"  {name} {y}: {state['n'] - before:,} rows")
    print(f"  -> {path}: {state['n']:,} rows total")
    if state["n"] == 0:
        sys.exit(f"No rows came back for {name}.")


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarize():
    out = []
    p = out.append
    bt = read(f"{OUT}/bt_daily.csv")
    p(f"== B&T daily summary: {len(bt):,} rows")
    dates = sorted({r["date"][:10] for r in bt})
    p(f"date range {dates[0]} to {dates[-1]}; {len(dates):,} distinct days")
    d0, d1 = dt.date.fromisoformat(dates[0]), dt.date.fromisoformat(dates[-1])
    have = set(dates)
    missing = [str(d0 + dt.timedelta(n)) for n in range((d1 - d0).days + 1) if str(d0 + dt.timedelta(n)) not in have]
    p(f"calendar days with no data at all: {len(missing)}" + (f" (first few: {missing[:8]})" if missing else ""))
    fac = defaultdict(lambda: {"rows": 0, "tot": 0.0, "min": "9", "max": "0", "dirs": set()})
    cats = defaultdict(float)
    for r in bt:
        f = fac[r["facility"]]
        f["rows"] += 1; f["tot"] += float(r["traffic"] or 0)
        d = r["date"][:10]; f["min"] = min(f["min"], d); f["max"] = max(f["max"], d); f["dirs"].add(r["direction"])
        cats[r["vehicle_class_category"]] += float(r["traffic"] or 0)
    p("facility | first day | last day | total vehicles | directions")
    for k, v in sorted(fac.items()):
        p(f"  {k} | {v['min']} | {v['max']} | {v['tot']:,.0f} | {sorted(v['dirs'])}")
    p("vehicle categories (total vehicles): " + "; ".join(f"{k}={v:,.0f}" for k, v in sorted(cats.items())))

    cz = read(f"{OUT}/crz_daily.csv")
    p(f"\n== CRZ daily summary: {len(cz):,} rows")
    cd = sorted({r["toll_date"][:10] for r in cz})
    p(f"date range {cd[0]} to {cd[-1]}; {len(cd):,} distinct days")
    grp = defaultdict(lambda: [0.0, 0.0, "", set()])
    cls = defaultdict(float)
    for r in cz:
        g = grp[r["detection_group"]]
        g[0] += float(r["crz_entries"] or 0); g[1] += float(r["excluded_entries"] or 0); g[2] = r["detection_region"]
        cls[r["vehicle_class"]] += float(r["crz_entries"] or 0)
    p("detection group | region | zone entries | excluded-roadway entries")
    for k, v in sorted(grp.items()):
        p(f"  {k} | {v[2]} | {v[0]:,.0f} | {v[1]:,.0f}")
    p("vehicle classes (zone entries): " + "; ".join(f"{k}={v:,.0f}" for k, v in sorted(cls.items())))

    tm = read(f"{OUT}/crz_10min.csv")
    p(f"\n== CRZ 10-minute summary: {len(tm):,} rows")
    tp = defaultdict(lambda: [0, 0.0])
    for r in tm:
        t = tp[(r["day_of_week"] in ("Saturday", "Sunday"), r["time_period"])]
        t[0] += 1; t[1] += float(r["crz_entries"] or 0)
    p("time_period labels by (is_weekend, label): rows, entries")
    for k, v in sorted(tp.items()):
        p(f"  {k} | {v[0]:,} | {v[1]:,.0f}")
    text = "\n".join(out)
    open(f"{OUT}/_summary.txt", "w").write(text + "\n")
    print("\n" + text)


def main():
    os.makedirs(OUT, exist_ok=True)
    now = dt.date.today().year
    print("1/3 B&T daily crossings (all years)...")
    pull("bt_daily", BT,
         "date,facility,direction,vehicle_class_category,sum(traffic_count) as traffic",
         "date,facility,direction,vehicle_class_category",
         "date,facility,direction,vehicle_class_category", "date", range(2019, now + 1))
    print("2/3 Zone entries by entry point...")
    pull("crz_daily", CRZ,
         "toll_date,detection_group,detection_region,vehicle_class,sum(crz_entries) as crz_entries,sum(excluded_roadway_entries) as excluded_entries",
         "toll_date,detection_group,detection_region,vehicle_class",
         "toll_date,detection_group,vehicle_class", "toll_date", range(2025, now + 1))
    print("3/3 Zone entries by 10-minute slot...")
    pull("crz_10min", CRZ,
         "toll_date,day_of_week,hour_of_day,minute_of_hour,time_period,vehicle_class,sum(crz_entries) as crz_entries",
         "toll_date,day_of_week,hour_of_day,minute_of_hour,time_period,vehicle_class",
         "toll_date,hour_of_day,minute_of_hour,vehicle_class", "toll_date", range(2025, now + 1))
    summarize()
    print("\nDone. Paste the text above (or data/raw/_summary.txt) back to Claude.")


if __name__ == "__main__":
    main()
