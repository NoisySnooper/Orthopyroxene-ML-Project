import os, re

docs = r"C:\Users\NQTa\Documents\MLCourse\Final Project\docs"
root = r"C:\Users\NQTa\Documents\MLCourse\Final Project"

print("="*70)
print("v10 AUDIT REPORT")
print("="*70)

v10_files = sorted(f for f in os.listdir(docs) if f.startswith("v10_"))
print(f"\n[1] v10 docs in docs/ ({len(v10_files)} total):")
for f in v10_files:
    sz = os.path.getsize(os.path.join(docs, f))
    print(f"   {f:52s} {sz/1024:6.1f} KB")

print(f"\n[2] Strategy docs (v10 scope update block present?):")
for f in ["stacking_strategy.md", "resampling_strategy.md", "optuna_strategy.md"]:
    p = os.path.join(docs, f)
    sz = os.path.getsize(p)
    with open(p, "r", encoding="utf-8") as fp:
        txt = fp.read()
    has = "## v10 scope update" in txt
    print(f"   {f:30s} {sz/1024:6.1f} KB  scope_update={has}")

print(f"\n[3] Top-level updates:")
for f in ["README.md", "PROJECT_OVERVIEW.md"]:
    p = os.path.join(root, f)
    sz = os.path.getsize(p)
    with open(p, "r", encoding="utf-8") as fp:
        txt = fp.read()
    print(f"   {f:30s} {sz/1024:6.1f} KB")
    print(f"      mentions manuscripts/opx_2026 : {'manuscripts/opx_2026' in txt}")
    print(f"      mentions manuscripts/cpx_2026 : {'manuscripts/cpx_2026' in txt}")
    print(f"      mentions 8 models             : {'8 models' in txt or '8-model' in txt}")
    print(f"      mentions v10_master_plan      : {'v10_master_plan' in txt}")
    print(f"      references stale v10_impl_plan: {'v10_implementation_plan' in txt}")

print(f"\n[4] Cross-reference validation (broken links within v10 docs):")
broken = []
for f in v10_files:
    p = os.path.join(docs, f)
    with open(p, "r", encoding="utf-8") as fp:
        txt = fp.read()
    refs = re.findall(r"v10_[a-z_]+\.md", txt)
    for r in set(refs):
        ref_path = os.path.join(docs, r)
        if not os.path.exists(ref_path):
            broken.append((f, r))
if not broken:
    print("   OK - no broken v10_*.md references")
else:
    for src, dst in broken:
        print(f"   BROKEN: {src} -> {dst}")

print(f"\n[5] Key decision consistency check:")
# Check master plan has canonical facts; check they appear in relevant subdocs
with open(os.path.join(docs, "v10_master_plan.md"), "r", encoding="utf-8") as fp:
    master = fp.read()

checks = [
    ("14 notebook roster",  "14 notebooks" in master or "14-notebook" in master.lower()),
    ("8 model roster",      "8 models" in master or "8-model" in master),
    ("4 ensemble methods",  "4" in master and "ensemble" in master),
    ("T01-T12 tests",       "T01" in master and "T12" in master),
    ("T13-T14 universal",   "T13" in master and "T14" in master),
    ("gate c beat agreda",  "gate" in master.lower() and "(c)" in master),
    ("manuscripts/opx_2026", "opx_2026" in master),
    ("manuscripts/cpx_2026", "cpx_2026" in master),
    ("Phase A cleanup",     "Phase A" in master),
    ("Phase K submission",  "Phase K" in master),
]
for label, ok in checks:
    print(f"   {'OK ' if ok else 'MISS'}  {label}")

print(f"\n[6] Data file availability (Phase A prereqs):")
data_checks = [
    ("natural/2024-12-SGFTFN_ORTHOPYROXENES.csv",
     r"C:\Users\NQTa\Documents\MLCourse\Final Project\data\natural\2024-12-SGFTFN_ORTHOPYROXENES.csv"),
    ("natural/natural_opx_cleaned.csv",
     r"C:\Users\NQTa\Documents\MLCourse\Final Project\data\natural\natural_opx_cleaned.csv"),
    ("data/external/thermobar_examples/Thermobar",
     r"C:\Users\NQTa\Documents\MLCourse\Final Project\data\external\thermobar_examples\Thermobar"),
    ("results/nb03_per_family_winners.json",
     r"C:\Users\NQTa\Documents\MLCourse\Final Project\results\nb03_per_family_winners.json"),
    ("results/nb03_optuna_best_params.json",
     r"C:\Users\NQTa\Documents\MLCourse\Final Project\results\nb03_optuna_best_params.json"),
    ("config.py",
     r"C:\Users\NQTa\Documents\MLCourse\Final Project\config.py"),
]
for label, p in data_checks:
    present = os.path.exists(p)
    if present:
        sz = os.path.getsize(p)
        print(f"   OK   {label:55s} ({sz/1024:.1f} KB)")
    else:
        print(f"   MISS {label}")

print(f"\n[7] Existing v9 deprecated doc:")
p = os.path.join(docs, "v10_implementation_plan.md")
if os.path.exists(p):
    sz = os.path.getsize(p)
    print(f"   v10_implementation_plan.md present ({sz/1024:.1f} KB)")
    print(f"   NOTE: superseded by v10_master_plan.md. Should be archived or deleted.")

print("\n" + "="*70)
print("AUDIT COMPLETE")
print("="*70)
