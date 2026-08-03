import ast
src = open("analysis_dashboard.py").read()
try:
    ast.parse(src)
    print("Syntax OK")
except SyntaxError as e:
    print(f"SyntaxError at line {e.lineno}: {e.msg}")
    print(f"  {e.text}")
