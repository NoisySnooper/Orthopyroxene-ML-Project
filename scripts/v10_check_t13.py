import os

docs = r"C:\Users\NQTa\Documents\MLCourse\Final Project\docs"
files = [
    "v10_nb03_test_protocol.md",
    "v10_universal_model_exploration.md",
    "v10_master_plan.md",
]

for f in files:
    p = os.path.join(docs, f)
    with open(p, "r", encoding="utf-8") as fp:
        txt = fp.read()
    has_t13 = "T13" in txt
    has_t14 = "T14" in txt
    print(f"\n=== {f} ===")
    print(f"  T13 present: {has_t13}")
    print(f"  T14 present: {has_t14}")
    if has_t13:
        idx = txt.find("T13")
        print(f"  T13 context: ...{txt[max(0,idx-30):idx+120]!r}...")
    if has_t14:
        idx = txt.find("T14")
        print(f"  T14 context: ...{txt[max(0,idx-30):idx+120]!r}...")
