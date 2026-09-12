#!/usr/bin/env bash
# Extract per-row results from every sheet log and print a compact table.
# Usage: bash extract_sheet.sh
set -euo pipefail
cd "$(dirname "$0")"

PYBIN="/Users/ary/.workbuddy/binaries/python/envs/torchjam/bin/python3"
exec "$PYBIN" - <<'PYEOF'
import glob, re, os, json

logs = sorted(glob.glob("results/sheet_*.log"))
rows = {}  # (variant, row) -> dict
for path in logs:
    name = os.path.basename(path)
    # v1 logs are sheet_<N>.log ; other variants are sheet_<variant>_<N>.log
    m = re.match(r"sheet_v(\d+)_\w+_(\d+)\.log", name)
    if m:
        variant, row = f"v{m.group(1)}", int(m.group(2))
        variant = {"v1": "v1_sdpa", "v2": "v2_compile", "v3": "v3_fused",
                   "v4": "v4_sdpa_compile"}.get(variant, variant)
    else:
        m = re.match(r"sheet_(\d+)\.log", name)
        if not m:
            continue
        variant, row = "v1_sdpa", int(m.group(1))
    text = open(path).read()
    entry = {"variant": variant, "row": row, "file": name}
    macc = re.search(r"summary: (PASS|FAIL) \| max_abs=([\d.eE+-]+)", text)
    entry["accuracy"] = macc.group(1) if macc else "CRASH"
    entry["max_abs"] = macc.group(2) if macc else "?"
    mbase = re.search(r"baseline : median=([\d.]+) ms", text)
    mopt = re.search(r"optimized: median=([\d.]+) ms", text)
    mspd = re.search(r"speedup  : ([\d.]+)x", text)
    entry["baseline_ms"] = float(mbase.group(1)) if mbase else None
    entry["optimized_ms"] = float(mopt.group(1)) if mopt else None
    entry["speedup"] = float(mspd.group(1)) if mspd else None
    rows[(variant, row)] = entry

# Sheet definition (batch, d_model, heads, seq, layers, ffn)
sheet = {
    1: (64, 128, 4, 128, 4, 128), 2: (1, 128, 4, 128, 4, 128),
    3: (4, 128, 4, 128, 4, 128), 4: (16, 128, 4, 128, 4, 128),
    5: (128, 128, 4, 128, 4, 128), 6: (10000, 128, 4, 128, 4, 128),
    7: (64, 32, 4, 128, 4, 32), 8: (64, 1024, 4, 128, 4, 1024),
    9: (64, 128, 1, 128, 4, 128), 10: (64, 128, 2, 128, 4, 128),
    11: (64, 128, 16, 128, 4, 128), 12: (64, 128, 4, 32, 4, 128),
    13: (64, 128, 4, 1024, 4, 128), 14: (32, 1024, 16, 100000, 2, 1024),
}
variants = ["v0_baseline", "v1_sdpa", "v2_compile", "v3_fused", "v4_sdpa_compile"]

print(f"{'Row':>3} | {'config (b,dm,h,s,l,ff)':<26} | " + " | ".join(f"{v:>6}" for v in variants))
print("-" * 120)
for row in range(1, 15):
    bs, dm, h, sl, ly, ff = sheet[row]
    cfg = f"{bs},{dm},{h},{sl},{ly},{ff}"
    cells = []
    for v in variants:
        e = rows.get((v, row))
        if e is None or e["speedup"] is None:
            cells.append("  FAIL")
        else:
            cells.append(f"{e['speedup']:6.2f}x")
    print(f"{row:>3} | {cfg:<26} | " + " | ".join(cells))

print("\n# ===== Best variant per row =====\n")
best = []
for row in range(1, 15):
    bs, dm, h, sl, ly, ff = sheet[row]
    cands = []
    for v in variants:
        e = rows.get((v, row))
        if e and e["speedup"] is not None:
            cands.append((e["speedup"], v, e["optimized_ms"], e["accuracy"]))
    if not cands:
        print(f"Row {row:>2}: NO variant ran")
        continue
    cands.sort(reverse=True)
    sp, v, oms, acc = cands[0]
    runner_up = f" | runner-up {cands[1][1]} {cands[1][0]:.2f}x" if len(cands) > 1 else ""
    print(f"Row {row:>2} (b={bs}, dm={dm}, h={h}, s={sl}, l={ly}, ff={ff}): "
          f"BEST = {v}  {sp:.2f}x  (opt {oms:.3f} ms, {acc}){runner_up}")
    best.append({"row": row, "config": [bs, dm, h, sl, ly, ff],
                 "best_variant": v, "best_speedup": sp})

with open("results/_best_by_row.json", "w") as f:
    json.dump(best, f, indent=2)
print("\nJSON written to results/_best_by_row.json")
PYEOF
