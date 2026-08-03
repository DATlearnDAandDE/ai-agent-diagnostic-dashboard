import re

with open("analysis_dashboard.py", "r") as f:
    src = f.read()

# Fix tất cả transparent trong context Plotly (không phải CSS)
# bgcolor="transparent" -> bgcolor="rgba(0,0,0,0)"
src = src.replace('bgcolor="transparent"', 'bgcolor="rgba(0,0,0,0)"')
src = src.replace("bgcolor='transparent'", "bgcolor='rgba(0,0,0,0)'")

# paper_bgcolor đã fix, nhưng chắc chắn lại
src = src.replace('paper_bgcolor="transparent"', 'paper_bgcolor="rgba(0,0,0,0)"')
src = src.replace('plot_bgcolor="transparent"', 'plot_bgcolor="rgba(0,0,0,0)"')

with open("analysis_dashboard.py", "w") as f:
    f.write(src)

print("DONE")
print("Remaining 'transparent' in Python (non-CSS context):")
for i, line in enumerate(src.split("\n"), 1):
    if "transparent" in line and not line.strip().startswith(("#", "//", "*", "<!--")):
        print(f"  L{i}: {line.strip()[:80]}")
