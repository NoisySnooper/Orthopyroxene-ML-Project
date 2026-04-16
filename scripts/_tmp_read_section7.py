import os

p = r"C:\Users\NQTa\Documents\MLCourse\Final Project\docs\v10_master_plan.md"
with open(p, "r", encoding="utf-8") as f:
    txt = f.read()

# Find Section 7 header and Section 8 header
s7_start = txt.find("## 7. Test-first protocol summary")
s8_start = txt.find("## 8. ")
print(f"Section 7 starts at char {s7_start}")
print(f"Section 8 starts at char {s8_start}")
print(f"Section 7 length: {s8_start - s7_start} chars")
print("--- Section 7 content ---")
print(txt[s7_start:s8_start])
