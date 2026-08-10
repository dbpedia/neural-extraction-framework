"""Offline tests for the 2026-08-04 date-style literal fix (train data only,
no network, no LLM). Validates against the REAL learned conventions.

Run:  USE_TF=0 python3 test_date_style.py
"""
import os, sys
os.environ.setdefault("USE_TF", "0")

import text2kg_harness as h
from autonomous_pipeline_v13 import _normalize_literal

vf_b  = h.learn_value_formats("4_building")
qh_b  = h.learn_quote_conventions("4_building")
NRHP  = "addedToTheNationalRegisterOfHistoricPlaces"

results = []
def check(name, got, want):
    ok = got == want
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r}, want {want!r}")

print("== learned date styles (4_building train) ==")
# completionDate train is 2009-06-01 x18 vs "April 2014" x1 -> honest majority
# is iso; the month-year force-quote rule covers "December 2008" independently.
check("completionDate style", vf_b.get("completionDate", {}).get("date_style"), "iso")
check("buildingStartDate style", vf_b.get("buildingStartDate", {}).get("date_style"), "english")
check("NRHP-added style", vf_b.get(NRHP, {}).get("date_style"), "iso")

print("== emission: english-style predicates keep month dates, quoted ==")
check('"December 2008" (the 7-sentence class)',
      _normalize_literal("December 2008", "completionDate", "date",
                         qh_b.get("completionDate"), vf_b.get("completionDate")),
      ("December 2008", True))
check('"30 March 2007" kept verbatim',
      _normalize_literal("30 March 2007", "buildingStartDate", "date",
                         qh_b.get("buildingStartDate"), vf_b.get("buildingStartDate")),
      ("30 March 2007", True))

print("== emission: iso-style predicate converts + quotes (NRHP, 3 sentences) ==")
for raw in ("February 27, 1987", "27 February 1987", "February 27th, 1987"):
    check(f'NRHP "{raw}"',
          _normalize_literal(raw, NRHP, "", qh_b.get(NRHP), vf_b.get(NRHP)),
          ("1987-02-27", True))

print("== regressions must not move ==")
# Bare year under an english-style date predicate: untouched, unquoted.
check('bare year 1988 unchanged',
      _normalize_literal("1988", "completionDate", "date",
                         qh_b.get("completionDate"), vf_b.get("completionDate")),
      ("1988", False))
# Scientist-class: english date under an iso-style pid still ISO-converts,
# quoting still follows the learned hint (dynamic — no hardcoded pid).
vf_s = h.learn_value_formats("18_scientist")
qh_s = h.learn_quote_conventions("18_scientist")
iso_pids = [p for p, d in vf_s.items() if d.get("date_style") == "iso"]
check("scientist has iso-style date pids", bool(iso_pids), True)
if iso_pids:
    pid = iso_pids[0]
    got = _normalize_literal("January 1, 1726", pid, "date", qh_s.get(pid), vf_s.get(pid))
    check(f"scientist {pid} ISO conversion",
          got, ("1726-01-01", qh_s.get(pid) is True))
# Epoch guard: parenthesized month value must NOT be mangled or force-quoted
# into a bare verbatim return by the new branch.
vf_c = h.learn_value_formats("8_celestialbody")
got_v, _q = _normalize_literal("31 July 2016 (JD2457600.5)", "epoch", "",
                               None, vf_c.get("epoch"))
check("epoch value not mangled", got_v, "31 July 2016 (JD2457600.5)")

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
