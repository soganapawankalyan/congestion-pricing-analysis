#!/usr/bin/env python3
"""
Project 3A - NYC congestion pricing analysis.   Needs: pandas, numpy, matplotlib.

Reads data/raw/*.csv written by 01_download_data.py and answers three questions.

Q1  How much traffic left?   (B&T tunnels that feed the zone vs far-away crossings)
    Method: a "gap" estimator. For a treated crossing, compare its weekly log traffic with an index of
    donor crossings that do not feed the zone. The expected gap for each week of the year comes from the
    baseline years (default Jan 2023 - Jan 2025). Effect = actual gap - expected gap, shown as an event study.
    Inference: PLACEBO tests, not textbook standard errors (there are only two treated crossings).
      - placebo in space: pretend each donor crossing was treated; how big are "effects" by chance?
      - placebo in time:  pretend the toll started a year early; does the method invent an effect?

Q2  Did drivers shift their timing to dodge the peak toll?   (10-minute zone entries, Tue-Thu)
    Method: jump at each price boundary. Entries are smooth in time except where the price changes, so a
    local fit on each side of the boundary (on a log scale, which handles steep rush-hour ramps) measures
    the response. 90% ranges come from resampling days; a robustness range varies the bandwidth and
    polynomial order; a placebo-boundary test shows how big a jump appears at fake boundaries elsewhere in
    the day. Boundaries are read from the data's own time_period labels.

Q3  Did traffic divert?   Same estimator as Q1 on crossings that approach the zone but are not tolled
    on entry, plus the share of entries using untolled roads, plus a reconciliation of the two datasets.

Robustness (Q1): Tue-Thu weeks only with holiday periods removed, other baselines, leave-one-donor-out.

Outputs: results/*.csv, results/summary.json, figures/*.png      Run:  python3 02_analysis.py
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

START = pd.Timestamp("2025-01-05")            # first Sunday of the toll
BASE_START = pd.Timestamp("2023-01-01")       # baseline = Jan 2023 .. day before START (avoids COVID years)
TREATED = {"Queens-Midtown Tunnel (into Manhattan)": ("Queens Midtown Tunnel", "Westbound to Manhattan"),
           "Hugh L. Carey Tunnel (into Manhattan)": ("Hugh L. Carey Tunnel", "Northbound to Manhattan")}
OUTBOUND = {"Queens-Midtown Tunnel (out of Manhattan)": ("Queens Midtown Tunnel", "Eastbound to Queens"),
            "Hugh L. Carey Tunnel (out of Manhattan)": ("Hugh L. Carey Tunnel", "Southbound to Brooklyn")}
DIVERSION = {"Henry Hudson Bridge (into Manhattan)": ("Henry Hudson Bridge", "Southbound to Manhattan"),
             "RFK Bridge from Bronx (into Manhattan/Queens)": ("Robert F. Kennedy Bridge Bronx", "Southbound to Manhattan or Queens")}
DONOR_FACILITIES = ["Cross Bay Bridge", "Marine Parkway Bridge", "Verrazzano - Narrows Bridge",
                    "Throgs Neck Bridge", "Bronx - Whitestone Bridge"]
# reference palette (validated in the earlier dashboards)
C1, C2, C3, GRAY, INK, INK2, GRID, BG = "#2a78d6", "#eb6834", "#1baf7a", "#898781", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


# ----------------------------------------------------------------------------- helpers
def need(df, cols, name):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        sys.exit(f"{name} is missing columns {miss}. Re-run 01_download_data.py. Found: {list(df.columns)}")


def style(ax, title=None):
    ax.set_facecolor(BG)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")


def weekly_levels(bt):
    """Weekly (Sunday-Saturday) total vehicles per facility x direction. Keeps complete weeks only."""
    day = bt.groupby(["facility", "direction", "date"], as_index=False).traffic.sum()
    day["week"] = day["date"].dt.to_period("W-SAT").dt.start_time
    w = day.groupby(["facility", "direction", "week"]).agg(traffic=("traffic", "sum"), days=("date", "nunique")).reset_index()
    w = w[w.days == 7]
    return {(f, d): g.set_index("week").traffic.sort_index() for (f, d), g in w.groupby(["facility", "direction"])}


def gap_effect(y, donors, start=START, base_start=BASE_START):
    """Weekly effect (log points) of a series y relative to its donor-based counterfactual.
    Baseline weeks are scored out-of-sample: a baseline week in year Y is compared with the same
    week-of-year in the *other* baseline years, so the pre-period noise shown is honest."""
    idx = y.index.intersection(donors.dropna().index)
    ly, ld = np.log(y.loc[idx]), np.log(donors.loc[idx])
    base = (ly.index >= base_start) & (ly.index < start)
    if base.sum() < 52:
        sys.exit("Fewer than 52 baseline weeks: check the date range of data/raw/bt_daily.csv")
    index = (ld - ld[base].mean()).mean(axis=1)          # donor index relative to baseline average
    g = ly - index
    woy = pd.Series(np.minimum((g.index.day_of_year - 1) // 7, 51), index=g.index)
    yr = pd.Series(g.index.year, index=g.index)
    bidx = g.index[base]
    eff = pd.Series(np.nan, index=g.index)
    for t in g.index:
        same = bidx[(woy[bidx] == woy[t])]
        if base[g.index.get_loc(t)]:
            same = same[yr[same] != yr[t]]             # out-of-sample for baseline weeks
        if len(same):
            eff[t] = g[t] - g[same].mean()
    return eff


def summarize_effect(eff, start=START):
    post = eff[eff.index >= start].dropna()
    y1 = post[post.index < start + pd.Timedelta(weeks=52)]
    y2 = post[post.index >= start + pd.Timedelta(weeks=52)]
    pre = eff[eff.index < start].dropna()
    pct = lambda s: (np.exp(s.mean()) - 1) * 100 if len(s) else np.nan
    return dict(first_year_pct=pct(y1), after_first_year_pct=pct(y2), all_post_pct=pct(post),
                post_weeks=int(len(post)), pre_noise_sd_pct=float(pre.std() * 100) if len(pre) > 3 else np.nan)


def pivot(series_dict, keys):
    return pd.concat({k: series_dict[k] for k in keys}, axis=1, sort=True)


# ----------------------------------------------------------------------------- Q1 / Q3 volume effects
def run_volume(bt, out):
    need(bt, ["date", "facility", "direction", "vehicle_class_category", "traffic"], "bt_daily.csv")
    bt["date"] = pd.to_datetime(bt["date"].str[:10])
    bt["traffic"] = pd.to_numeric(bt["traffic"], errors="coerce").fillna(0)
    W = weekly_levels(bt)
    for label, (f, d) in {**TREATED, **OUTBOUND, **DIVERSION}.items():
        if (f, d) not in W:
            sys.exit(f"Could not find {f!r} / {d!r} in the data. Facilities+directions present:\n" +
                     "\n".join(f"  {a!r} / {b!r}" for a, b in sorted(W)))
    donor_keys = [k for k in W if k[0] in DONOR_FACILITIES]
    if len(donor_keys) < 6:
        sys.exit(f"Expected donor crossings for {DONOR_FACILITIES}, found {donor_keys}")
    D = pivot(W, donor_keys)

    units = {}
    units["Both tunnels into Manhattan (combined)"] = sum(W[v] for v in TREATED.values()).dropna()
    for label, key in {**TREATED, **OUTBOUND, **DIVERSION}.items():
        units[label] = W[key]

    rows, effs = [], {}
    for label, y in units.items():
        eff = gap_effect(y, D)
        effs[label] = eff
        rows.append(dict(unit=label, **summarize_effect(eff)))
    res = pd.DataFrame(rows)

    # placebo in space: each donor crossing treated as if it were the tolled one (donors exclude its own facility)
    plc, plc_eff = [], {}
    for k in donor_keys:
        others = [j for j in donor_keys if j[0] != k[0]]
        e = gap_effect(W[k], D[others])
        plc_eff[k] = e
        s = summarize_effect(e)
        plc.append(dict(unit=f"{k[0]} | {k[1]}", **s))
    plc = pd.DataFrame(plc)
    ref = res.set_index("unit").loc["Both tunnels into Manhattan (combined)", "first_year_pct"]
    p_space = (np.sum(np.abs(plc.first_year_pct) >= abs(ref)) + 1) / (len(plc) + 1)

    # placebo in time: pretend the toll began 2024-01-07 with baseline Jul 2022 - Jan 2024
    ps, pb = pd.Timestamp("2024-01-07"), pd.Timestamp("2022-07-03")
    e_t = gap_effect(units["Both tunnels into Manhattan (combined)"], D, start=ps, base_start=pb)
    in_time = e_t[(e_t.index >= ps) & (e_t.index < START)].dropna()
    time_pct = (np.exp(in_time.mean()) - 1) * 100 if len(in_time) else np.nan

    res.to_csv(f"{out}/results/volume_effects.csv", index=False)
    plc.to_csv(f"{out}/results/placebo_donor_effects.csv", index=False)
    eff_tbl = pd.DataFrame({k: (np.exp(v) - 1) * 100 for k, v in effs.items()})
    eff_tbl.index.name = "week_start"
    eff_tbl.to_csv(f"{out}/results/weekly_effects_pct.csv")
    pe_tbl = pd.DataFrame({f"{k[0]} | {k[1]}": (np.exp(e) - 1) * 100 for k, e in plc_eff.items()})
    pe_tbl.index.name = "week_start"
    pe_tbl.to_csv(f"{out}/results/placebo_weekly_effects_pct.csv")
    return dict(res=res, plc=plc, effs=effs, plc_eff=plc_eff, p_space=float(p_space),
                time_pct=float(time_pct), donor_keys=donor_keys, W=W)


def fig_event_study(v, out):
    e = v["effs"]["Both tunnels into Manhattan (combined)"]
    pe = pd.concat(v["plc_eff"], axis=1, sort=True)
    lo, hi = pe.quantile(0.1, axis=1), pe.quantile(0.9, axis=1)
    pct = lambda s: (np.exp(s.rolling(4, center=True, min_periods=2).mean()) - 1) * 100
    x = e.index[e.index >= START - pd.Timedelta(weeks=60)]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160, facecolor=BG)
    style(ax, "Traffic into Manhattan through the two tunnels, vs. what far-away crossings imply")
    ax.fill_between(x, (np.exp(lo.reindex(x)) - 1) * 100, (np.exp(hi.reindex(x)) - 1) * 100, color=GRAY, alpha=0.25, lw=0,
                    label="Range for crossings that were never tolled (10th-90th pct)")
    ax.plot(x, pct(e).reindex(x), color=C1, lw=2.2, label="Queens-Midtown + Carey tunnels, inbound (4-week average)")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(START, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(START + pd.Timedelta(days=6), ax.get_ylim()[1] * 0.92, "Toll starts\nJan 5, 2025", color=INK, fontsize=8, va="top")
    ax.set_ylabel("Difference from expected traffic (%)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")
    fig.text(0.01, 0.01, "Each week is compared with the same week of the year in 2023-24, relative to far-away crossings (Verrazzano, Throgs Neck, Whitestone, Cross Bay, Marine Parkway).", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f"{out}/figures/event_study.png", facecolor=BG); plt.close(fig)


def fig_effect_summary(v, out):
    res = v["res"].set_index("unit")
    order = list(TREATED) + ["Both tunnels into Manhattan (combined)"] + list(OUTBOUND) + list(DIVERSION)
    order = [o for o in order if o in res.index][::-1]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160, facecolor=BG)
    style(ax, "First-year change by crossing, against the spread seen at never-tolled crossings")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.grid(axis="y", visible=False)
    p = v["plc"].first_year_pct
    ax.axvspan(p.min(), p.max(), color=GRAY, alpha=0.2, lw=0, label="Range at never-tolled crossings (placebo)")
    for i, u in enumerate(order):
        val = res.loc[u, "first_year_pct"]
        col = C1 if "Both" in u or "into Manhattan" in u and "Bridge" not in u else (C2 if "Bridge" in u else C3)
        ax.scatter([val], [i], s=70, color=col, zorder=3, edgecolor=BG, linewidth=1.5)
        ax.text(val + (0.6 if val >= 0 else -0.6), i, f"{val:+.1f}%", va="center", ha="left" if val >= 0 else "right", fontsize=8.5, color=INK, fontweight="bold")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8.5, color=INK2)
    ax.set_xlabel("Average difference from expected traffic, Jan 2025 - Jan 2026 (%)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{out}/figures/effect_summary.png", facecolor=BG); plt.close(fig)


# ----------------------------------------------------------------------------- Q2 time-of-day response
def group_of(c):
    c = str(c)
    if c.startswith("1"): return "Cars, pickups, vans"
    if c.startswith("TLC"): return "Taxi / for-hire"
    if c[:1] in ("2", "3"): return "Trucks"
    return "Buses / motorcycles"


def rd_jump(y, b, bw=12, order=1):
    """Jump in entries per 10 minutes at slot b: local polynomial fit on each side, evaluated at the boundary.
    Only the `bw` slots next to the boundary are used, so a smooth rush-hour curve cannot fake a jump."""
    s = np.arange(len(y))
    L = s[(s >= max(0, b - bw)) & (s < b)]
    R = s[(s >= b) & (s < min(len(y), b + bw))]
    if len(L) < order + 2 or len(R) < order + 2:
        return None
    cl = np.polyfit(L - b, y[L], order)
    cr = np.polyfit(R - b, y[R], order)
    left_at_b, right_at_b = np.polyval(cl, 0), np.polyval(cr, 0)
    return left_at_b, right_at_b


def rd_log_jump(y, b, bw=12, order=1):
    """Same as rd_jump but fitted to log(entries): a steep, roughly exponential ramp is close to a straight line
    on a log scale, so curvature is not mistaken for a jump. Returns the percent jump at slot b."""
    s = np.arange(len(y))
    L = s[(s >= max(0, b - bw)) & (s < b)]
    R = s[(s >= b) & (s < min(len(y), b + bw))]
    if len(L) < order + 2 or len(R) < order + 2 or (y[L] <= 0).any() or (y[R] <= 0).any():
        return np.nan
    cl = np.polyfit(L - b, np.log(y[L]), order)
    cr = np.polyfit(R - b, np.log(y[R]), order)
    return 100 * (np.exp(np.polyval(cr, 0) - np.polyval(cl, 0)) - 1)


def run_timing(tm, out, boot=300, seed=11):
    need(tm, ["toll_date", "day_of_week", "hour_of_day", "minute_of_hour", "time_period", "vehicle_class", "crz_entries"], "crz_10min.csv")
    tm = tm.copy()
    tm["toll_date"] = pd.to_datetime(tm["toll_date"].str[:10])
    for c in ("hour_of_day", "minute_of_hour", "crz_entries"):
        tm[c] = pd.to_numeric(tm[c], errors="coerce")
    tm["slot"] = (tm.hour_of_day * 6 + tm.minute_of_hour // 10).astype(int)
    tm["grp"] = tm.vehicle_class.map(group_of)
    wk = tm[tm.day_of_week.isin(["Tuesday", "Wednesday", "Thursday"])]
    lab = wk.groupby("slot").time_period.agg(lambda s: s.mode().iat[0]).sort_index()
    bounds = [(int(s), lab.iloc[i - 1], lab.iloc[i]) for i, s in enumerate(lab.index) if i > 0 and lab.iloc[i] != lab.iloc[i - 1]]
    daily = wk.groupby(["toll_date", "slot", "grp"]).crz_entries.sum().unstack("grp", fill_value=0)
    daily["All vehicles"] = daily.sum(axis=1)
    dates = daily.index.get_level_values(0).unique()
    cube = {g: daily[g].unstack("slot", fill_value=0).reindex(columns=range(144), fill_value=0) for g in daily.columns}
    rng = np.random.default_rng(seed)
    boot_idx = [rng.integers(0, len(dates), len(dates)) for _ in range(boot)]
    prof = pd.DataFrame({g: cube[g].mean(axis=0) for g in cube})
    rows = []
    for g, mat in cube.items():
        arr = mat.values.astype(float)
        y = arr.mean(axis=0)
        for (b, frm, to) in bounds:
            main = rd_jump(y, b, 12, 1)
            if main is None:
                continue
            pct = lambda lr: 100 * (lr[1] - lr[0]) / lr[0] if lr and lr[0] > 0 else np.nan
            lin = pct(main)
            jump = rd_log_jump(y, b, 12, 1)
            bs = [rd_log_jump(arr[i].mean(axis=0), b, 12, 1) for i in boot_idx]
            rob = [rd_log_jump(y, b, bw, o) for bw in (6, 9, 12, 18) for o in (1, 2) if not (bw < 9 and o == 2)]
            # placebo boundaries: same estimator at slots far from any real price change
            far = [sl for sl in range(12, 133) if all(abs(sl - bb[0]) >= 24 for bb in bounds)]
            fake = np.array([rd_log_jump(y, sl, 12, 1) for sl in far], dtype=float)
            fake = np.abs(fake[~np.isnan(fake)])
            fp90 = float(np.percentile(fake, 90)) if len(fake) else np.nan
            rows.append(dict(group=g, boundary=f"{b//6:02d}:{(b%6)*10:02d}", change=f"{frm} -> {to}",
                             jump_pct=jump, ci90_lo=np.nanpercentile(bs, 5), ci90_hi=np.nanpercentile(bs, 95),
                             robust_min=np.nanmin(rob), robust_max=np.nanmax(rob),
                             fake_boundary_abs_p90=fp90, clears_fake_p90=bool(abs(jump) > fp90) if not np.isnan(fp90) else False,
                             same_sign_all_settings=bool(np.nanmin(rob) > 0 or np.nanmax(rob) < 0),
                             linear_jump_pct=lin,
                             entries_per_10min_just_before=main[0], entries_per_10min_just_after=main[1],
                             daily_entries_in_group=float(y.sum())))
    res = pd.DataFrame(rows)
    res.to_csv(f"{out}/results/timing_jumps.csv", index=False)
    prof.to_csv(f"{out}/results/timing_profile_tue_thu.csv")
    return dict(bounds=bounds, prof=prof, jumps=res, ndays=int(len(dates)))


def fig_timing(t, out):
    prof, bounds = t["prof"], t["bounds"]
    y = prof["All vehicles"].reindex(range(144), fill_value=0).values.astype(float)
    hrs = np.arange(144) / 6.0
    fig = plt.figure(figsize=(9, 7), dpi=160, facecolor=BG)
    gs = fig.add_gridspec(2, max(1, len(bounds)), height_ratios=[1.1, 1])
    ax = fig.add_subplot(gs[0, :])
    style(ax, "Vehicles entering the zone, by time of day (Tue-Thu average)")
    ax.plot(hrs, y, color=C1, lw=2)
    for (b, frm, to) in bounds:
        ax.axvline(b / 6, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(b / 6 + 0.12, ax.get_ylim()[1] * 0.97, f"{b//6}:{(b%6)*10:02d}\n{frm} to {to}", fontsize=7.5, color=INK, va="top")
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3)); ax.set_xticklabels([f"{h}:00" for h in range(0, 25, 3)])
    ax.set_ylabel("Entries per 10 minutes", color=INK2, fontsize=9)
    for k, (b, frm, to) in enumerate(bounds):
        a2 = fig.add_subplot(gs[1, k]); style(a2)
        lo, hi = max(0, b - 18), min(144, b + 18)
        a2.scatter(hrs[lo:hi], y[lo:hi], s=14, color=C1, zorder=3)
        for side, (s0, s1) in (("L", (max(0, b - 12), b)), ("R", (b, min(144, b + 12)))):
            ss = np.arange(s0, s1); co = np.polyfit(ss - b, np.log(np.maximum(y[ss], 1e-9)), 1)
            a2.plot(hrs[ss], np.exp(np.polyval(co, ss - b)), color=C2, lw=2)
        a2.axvline(b / 6, color=INK2, lw=1, ls=(0, (4, 3)))
        j = t["jumps"]; r = j[(j.group == "All vehicles") & (j.boundary == f"{b//6:02d}:{(b%6)*10:02d}")]
        ttl = f"{b//6}:{(b%6)*10:02d}: {frm} to {to}"
        if len(r):
            r = r.iloc[0]; ttl += f"\njump {r.jump_pct:+.0f}%  (90% range {r.ci90_lo:+.0f}% to {r.ci90_hi:+.0f}%)"
        a2.set_title(ttl, loc="left", fontsize=9, color=INK, fontweight="bold")
        a2.set_xlabel("Hour of day", color=INK2, fontsize=8)
    fig.text(0.01, 0.005, "Orange lines: fits to the two hours on each side of the price change (log scale). The gap between them at the dashed line is the response to the price.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1)); fig.savefig(f"{out}/figures/time_of_day.png", facecolor=BG); plt.close(fig)


# ----------------------------------------------------------------------------- Q1 robustness
def excluded_dates(d0, d1):
    """Days to drop for the 'clean' specification: +-3 days around US federal holidays and Dec 20 - Jan 4."""
    from pandas.tseries.holiday import USFederalHolidayCalendar
    bad = set()
    for h in USFederalHolidayCalendar().holidays(d0 - pd.Timedelta(days=10), d1 + pd.Timedelta(days=10)):
        for k in range(-3, 4):
            bad.add(h + pd.Timedelta(days=k))
    for yr in range(d0.year - 1, d1.year + 1):
        bad.update(pd.date_range(f"{yr}-12-20", f"{yr + 1}-01-04"))
    return pd.DatetimeIndex(sorted(bad))


def weekly_levels_clean(bt):
    """Tue-Wed-Thu totals per week, only for weeks where all three days are ordinary (no holiday period)."""
    day = bt.groupby(["facility", "direction", "date"], as_index=False).traffic.sum()
    bad = excluded_dates(day.date.min(), day.date.max())
    d = day[day.date.dt.day_of_week.isin([1, 2, 3]) & ~day.date.isin(bad)].copy()
    d["week"] = d["date"].dt.to_period("W-SAT").dt.start_time
    w = d.groupby(["facility", "direction", "week"]).agg(traffic=("traffic", "sum"), days=("date", "nunique")).reset_index()
    w = w[w.days == 3]
    return {(f, dd): g.set_index("week").traffic.sort_index() for (f, dd), g in w.groupby(["facility", "direction"])}


def volume_spec(W, donor_fac, base_start, label):
    donor_keys = [k for k in W if k[0] in donor_fac]
    D = pivot(W, donor_keys)
    comb = sum(W[v] for v in TREATED.values()).dropna()
    s = summarize_effect(gap_effect(comb, D, base_start=base_start))
    pl = []
    for k in donor_keys:
        others = [j for j in donor_keys if j[0] != k[0]]
        pl.append(summarize_effect(gap_effect(W[k], D[others], base_start=base_start))["first_year_pct"])
    pl = np.array(pl)
    p = (np.sum(np.abs(pl) >= abs(s["first_year_pct"])) + 1) / (len(pl) + 1)
    return dict(spec=label, first_year_pct=s["first_year_pct"], all_post_pct=s["all_post_pct"], pre_noise_sd_pct=s["pre_noise_sd_pct"],
                post_weeks=s["post_weeks"], placebo_min=float(pl.min()), placebo_max=float(pl.max()), placebo_sd=float(pl.std(ddof=1)),
                n_placebos=len(pl), p_value=float(p))


def run_robustness(bt, v, out):
    Wc = weekly_levels_clean(bt)
    allf = DONOR_FACILITIES
    jul22 = pd.Timestamp("2022-07-03")
    specs = [volume_spec(v["W"], allf, BASE_START, "Main: all weeks, 2023-24 baseline"),
             volume_spec(v["W"], allf, jul22, "All weeks, baseline from Jul 2022"),
             volume_spec(Wc, allf, BASE_START, "Tue-Thu, holiday periods removed"),
             volume_spec(Wc, allf, jul22, "Tue-Thu clean, baseline from Jul 2022")]
    for f in allf:
        specs.append(volume_spec(Wc, [x for x in allf if x != f], BASE_START, f"Tue-Thu clean, without donor {f}"))
    res = pd.DataFrame(specs)
    res.to_csv(f"{out}/results/robustness_volume.csv", index=False)
    return res


def fig_robustness(r, out):
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160, facecolor=BG)
    style(ax)
    fig.suptitle("Does the tunnel result survive other ways of measuring it?", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.grid(axis="y", visible=False)
    n = len(r)
    for i, row in enumerate(r.itertuples()):
        y = n - 1 - i
        ax.plot([row.placebo_min, row.placebo_max], [y, y], color=GRAY, lw=7, alpha=0.3, solid_capstyle="butt")
        ax.scatter([row.first_year_pct], [y], s=60, color=C1, zorder=3, edgecolor=BG, linewidth=1.3)
        ax.text(row.first_year_pct, y + 0.34, f"{row.first_year_pct:+.1f}%", ha="center", fontsize=7.5, color=INK, fontweight="bold")
    ax.set_yticks(range(n)); ax.set_yticklabels(list(r.spec)[::-1], fontsize=8, color=INK2)
    ax.set_ylim(-0.6, n - 0.2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Both tunnels inbound, first-year change vs expected (%)\nBlue dot: tunnels.  Grey bar: range at never-tolled crossings in the same setup", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{out}/figures/robustness.png", facecolor=BG); plt.close(fig)


# ----------------------------------------------------------------------------- reconciliation and excluded roads
def run_reconcile(bt, cz, W, out):
    need(cz, ["toll_date", "detection_group", "vehicle_class", "crz_entries", "excluded_entries"], "crz_daily.csv")
    cz = cz.copy()
    cz["toll_date"] = pd.to_datetime(cz["toll_date"].str[:10])
    cz["crz_entries"] = pd.to_numeric(cz["crz_entries"], errors="coerce").fillna(0)
    cz["excluded_entries"] = pd.to_numeric(cz["excluded_entries"], errors="coerce").fillna(0)
    cz["all_entries"] = cz.crz_entries + cz.excluded_entries
    rec = []
    for grp, (fac, d) in {"Hugh L. Carey Tunnel": ("Hugh L. Carey Tunnel", "Northbound to Manhattan"),
                          "Queens Midtown Tunnel": ("Queens Midtown Tunnel", "Westbound to Manhattan")}.items():
        a = cz[cz.detection_group == grp].groupby("toll_date").all_entries.sum()
        b = bt[(bt.facility == fac) & (bt.direction == d) & (bt.date >= START)].groupby("date").traffic.sum()
        j = pd.concat([a.rename("zone_file"), b.rename("bt_file")], axis=1, sort=True).dropna()
        j["ratio"] = j.zone_file / j.bt_file
        m = j.groupby(j.index.to_period("M")).agg(zone_file=("zone_file", "sum"), bt_file=("bt_file", "sum"))
        m["ratio"] = m.zone_file / m.bt_file
        m.insert(0, "tunnel", grp)
        rec.append(m.reset_index().rename(columns={"index": "month"}))
    rec = pd.concat(rec)
    rec.to_csv(f"{out}/results/reconciliation_zone_vs_bt.csv", index=False)
    cz["month"] = cz.toll_date.dt.to_period("M")
    mm = cz.groupby("month").agg(tolled=("crz_entries", "sum"), excluded=("excluded_entries", "sum"))
    mm["excluded_share_pct"] = 100 * mm.excluded / (mm.tolled + mm.excluded)
    mm["days"] = cz.groupby("month").toll_date.nunique()
    mm["tolled_per_day"] = mm.tolled / mm.days
    mm.reset_index().to_csv(f"{out}/results/zone_entries_monthly.csv", index=False)
    return dict(rec=rec, monthly=mm)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    for d in ("results", "figures"):
        os.makedirs(f"{a.out}/{d}", exist_ok=True)
    bt = pd.read_csv(f"{a.raw}/bt_daily.csv")
    cz = pd.read_csv(f"{a.raw}/crz_daily.csv")
    tm = pd.read_csv(f"{a.raw}/crz_10min.csv")

    v = run_volume(bt, a.out)
    fig_event_study(v, a.out); fig_effect_summary(v, a.out)
    t = run_timing(tm, a.out); fig_timing(t, a.out)
    rb = run_robustness(bt, v, a.out); fig_robustness(rb, a.out)
    bt["date"] = pd.to_datetime(bt["date"]) if not np.issubdtype(bt["date"].dtype, np.datetime64) else bt["date"]
    r = run_reconcile(bt, cz, v["W"], a.out)

    pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", lambda x: f"{x:,.2f}")
    print("\n=== Q1/Q3  Traffic change vs expected (percent), first year = Jan 5 2025 - Jan 3 2026 ===")
    print(v["res"].to_string(index=False))
    print(f"\nPlacebo in space (never-tolled crossings, same method): first-year effects range "
          f"{v['plc'].first_year_pct.min():+.1f}% to {v['plc'].first_year_pct.max():+.1f}%  (n={len(v['plc'])}).")
    print(f"Combined-tunnel effect rank-based p-value: {v['p_space']:.2f}  (smallest possible with {len(v['plc'])} placebos = {1/(len(v['plc'])+1):.2f})")
    print(f"Placebo in time (fake start Jan 7 2024): tunnels' average 'effect' = {v['time_pct']:+.1f}%  (should be near 0)")
    print("\n=== Q2  Price boundaries found in the data (Tue-Thu) ===")
    print(t["bounds"], f"\nTue-Thu days used: {t['ndays']}")
    jj = t["jumps"]
    print("(jump = change in entries per 10 min right at the price change; 'fake p90' = size of the biggest typical jump at made-up boundaries)")
    print(jj[["group", "boundary", "change", "jump_pct", "ci90_lo", "ci90_hi", "robust_min", "robust_max", "fake_boundary_abs_p90", "clears_fake_p90"]].to_string(index=False))
    print("\n=== Q1 robustness: both tunnels inbound, first-year change under other specifications ===")
    print(rb[["spec", "first_year_pct", "all_post_pct", "pre_noise_sd_pct", "placebo_min", "placebo_max", "p_value"]].to_string(index=False))
    print("\n=== Reconciliation: zone file (tolled + excluded) vs B&T tunnel counts, by month ===")
    print(r["rec"].groupby("tunnel").tail(4).to_string(index=False))
    print("\n=== Share of zone entries on untolled roads, by month (last 6) ===")
    print(r["monthly"].tail(6).to_string())

    summary = dict(volume=v["res"].to_dict("records"), placebo_space=v["plc"].to_dict("records"), p_space=v["p_space"], placebo_time_pct=v["time_pct"],
                   boundaries=[list(map(str, b)) for b in t["bounds"]], timing=t["jumps"].to_dict("records"), robustness=rb.to_dict("records"),
                   reconciliation=r["rec"].astype(str).to_dict("records"))
    json.dump(summary, open(f"{a.out}/results/summary.json", "w"), indent=1, default=str)
    print("\nWrote results/*.csv, results/summary.json and figures/*.png")


if __name__ == "__main__":
    main()
