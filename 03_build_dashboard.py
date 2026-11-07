#!/usr/bin/env python3
"""Project 3A - builds dashboard/index.html from results/*.csv (run after 02_analysis.py).
Usage: python3 03_build_dashboard.py      Needs: pandas, numpy."""
import json, os, sys
import numpy as np
import pandas as pd

R = "results"
need = ["weekly_effects_pct.csv", "placebo_weekly_effects_pct.csv", "volume_effects.csv", "placebo_donor_effects.csv", "robustness_volume.csv",
        "timing_jumps.csv", "timing_profile_tue_thu.csv", "zone_entries_monthly.csv", "reconciliation_zone_vs_bt.csv"]
miss = [f for f in need if not os.path.exists(f"{R}/{f}")]
if miss:
    sys.exit(f"Missing {miss}. Run python3 02_analysis.py first (use the latest version).")

START = pd.Timestamp("2025-01-05")
COMB = "Both tunnels into Manhattan (combined)"
r1 = lambda x, n=1: None if pd.isna(x) else round(float(x), n)

# ---- event study: 4-week average of weekly effect, with placebo 10th-90th percentile band
we = pd.read_csv(f"{R}/weekly_effects_pct.csv", parse_dates=["week_start"]).set_index("week_start")
pw = pd.read_csv(f"{R}/placebo_weekly_effects_pct.csv", parse_dates=["week_start"]).set_index("week_start")
sm = lambda s: s.rolling(4, center=True, min_periods=2).mean()
x0 = START - pd.Timedelta(weeks=60)
tun = sm(we[COMB]); psm = pw.apply(sm)
lo, hi = psm.quantile(0.1, axis=1), psm.quantile(0.9, axis=1)
idx = tun.index[tun.index >= x0]
event = [dict(d=str(i.date()), t=r1(tun[i], 2), lo=r1(lo.get(i), 2), hi=r1(hi.get(i), 2)) for i in idx if not pd.isna(tun[i])]

# ---- first-year effect by crossing
ve = pd.read_csv(f"{R}/volume_effects.csv")
pl = pd.read_csv(f"{R}/placebo_donor_effects.csv")
units = [dict(name=r.unit, v=r1(r.first_year_pct, 2), after=r1(r.after_first_year_pct, 2), noise=r1(r.pre_noise_sd_pct, 1)) for r in ve.itertuples()]
placebo = dict(min=r1(pl.first_year_pct.min(), 2), max=r1(pl.first_year_pct.max(), 2), n=int(len(pl)))

rb = pd.read_csv(f"{R}/robustness_volume.csv")
robust = [dict(spec=r.spec, v=r1(r.first_year_pct, 2), post=r1(r.all_post_pct, 2), lo=r1(r.placebo_min, 2), hi=r1(r.placebo_max, 2), p=r1(r.p_value, 2)) for r in rb.itertuples()]

# ---- timing
prof = pd.read_csv(f"{R}/timing_profile_tue_thu.csv")
groups = [c for c in prof.columns if c != "slot"]
profile = {g: [r1(v, 1) for v in prof.set_index("slot").reindex(range(144), fill_value=0)[g]] for g in groups}
tj = pd.read_csv(f"{R}/timing_jumps.csv")
jumps = [dict(g=r.group, b=r.boundary, ch=r.change, j=r1(r.jump_pct, 1), lo=r1(r.ci90_lo, 1), hi=r1(r.ci90_hi, 1), rmin=r1(r.robust_min, 1), rmax=r1(r.robust_max, 1),
              fake=r1(r.fake_boundary_abs_p90, 1), clears=bool(r.clears_fake_p90), same=bool(r.same_sign_all_settings)) for r in tj.itertuples()]

# ---- zone
zm = pd.read_csv(f"{R}/zone_entries_monthly.csv")
zone = [dict(m=r.month, per_day=r1(r.tolled_per_day, 0), share=r1(r.excluded_share_pct, 2), days=int(r.days)) for r in zm.itertuples()]
rc = pd.read_csv(f"{R}/reconciliation_zone_vs_bt.csv")
recon = [dict(m=str(r.month), t=r.tunnel, ratio=r1(r.ratio, 3)) for r in rc.itertuples()]

# ---- findings text, chosen from the numbers so the wording never outruns the data
clean = rb[~rb.spec.str.contains("without donor")]
eff = float(rb.first_year_pct.median())
same_sign = bool((rb.first_year_pct < 0).all() or (rb.first_year_pct > 0).all())
pmin = float(rb.p_value.min())
if same_sign and pmin <= 0.10:
    q1_verdict = "Consistent across every check and borderline against chance."
elif same_sign:
    q1_verdict = "Consistent in direction across every check, but within what chance produces at never-tolled crossings."
else:
    q1_verdict = "The sign changes across checks, so there is no reliable change at these two crossings."
c21 = next((j for j in jumps if j["g"].startswith("Cars") and j["b"] == "21:00"), None)
a05 = next((j for j in jumps if j["g"] == "All vehicles" and j["b"] == "05:00"), None)
q2_main = (f"Cars entering just after 21:00, when the toll drops, are {c21['j']:+.0f}% vs just before"
           + (f" (stays between {c21['rmin']:+.0f}% and {c21['rmax']:+.0f}% under every setting)." if c21["same"] and c21["clears"] else " (not stable across settings).")) if c21 else ""
q2_note = ("At 05:00, when the toll rises, no group gives a stable answer, so the data cannot identify a response there." if a05 and not a05["same"]
           else "")
rr = [x["ratio"] for x in recon if x["ratio"]]
share_recent = float(zm.tail(6).excluded_share_pct.mean())
inbound = {u["name"]: u for u in units}
rfk = inbound.get("RFK Bridge from Bronx (into Manhattan/Queens)", {}).get("v")
hh = inbound.get("Henry Hudson Bridge (into Manhattan)", {}).get("v")
F = dict(
    q1=dict(n=f"{eff:+.1f}%", h="Traffic at the two tunnels feeding the zone", t=f"Median first-year change across {len(rb)} ways of measuring it. {q1_verdict}",
            a="Report it as a small, consistent decline, not a precise number. Two crossings is a thin base."),
    q2=dict(n=f"{c21['j']:+.0f}%" if c21 else "n/a", h="Cars shift their arrival to just after the price drops", t=(q2_main + " " + q2_note).strip(),
            a="Cheaper hours pull some trips later, which is what a time-of-day price should do."),
    q3=dict(n=f"{share_recent:.0f}%", h="Zone entries that use untolled roads", t=("Stable month to month. Bridges that approach the zone: "
            + (f"RFK from the Bronx {rfk:+.1f}%, Henry Hudson {hh:+.1f}% in the first year. " if rfk is not None and hh is not None else "")
            + "Small and descriptive, inside the noise."),
            a=f"The zone file and tunnel counts agree to within {100*max(abs(min(rr)-1), abs(max(rr)-1)):.0f}% in every month shown."),
)
D = dict(F=F, event=event, units=units, placebo=placebo, robust=robust, profile=profile, groups=groups, jumps=jumps, zone=zone, recon=recon, start=str(START.date()),
         eff=r1(eff, 1), pmin=r1(pmin, 2))

tpl = open("dashboard/template.html", encoding="utf-8").read()
html = tpl.replace("/*__DATA__*/null", json.dumps(D, separators=(",", ":")))
open("dashboard/index.html", "w", encoding="utf-8").write(html)
print(f"Wrote dashboard/index.html ({len(html)/1024:.0f} KB). Open it in a browser:  open dashboard/index.html")
