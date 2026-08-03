import re

with open('analysis_dashboard.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Trừ 32px chiều cao cho các biểu đồ
code = re.sub(r'fig_p1\s*=\s*p1\(df,pt,W1L,H1\)', r'fig_p1  = p1(df,pt,W1L,H1-32)', code)
code = re.sub(r'fig_p2\s*=\s*p2\(pt,W1M,H1\)', r'fig_p2  = p2(pt,W1M,H1-32)', code)
code = re.sub(r'fig_p3\s*=\s*p3\(pt,W1S1,H1\)', r'fig_p3  = p3(pt,W1S1,H1-32)', code)
code = re.sub(r'fig_p4\s*=\s*p4\(pt,W1S2,H1\)', r'fig_p4  = p4(pt,W1S2,H1-32)', code)
code = re.sub(r'fig_p4b\s*=\s*p4b\(pt,W1S1\+W1S2,H1\)', r'fig_p4b = p4b(pt,W1S1+W1S2,H1-32)', code)

code = re.sub(r'fig_p5\s*=\s*p5\(df,W2L,H2\)', r'fig_p5  = p5(df,W2L,H2-32)', code)
code = re.sub(r'fig_p6\s*=\s*p6\(df,W2M,H2\)', r'fig_p6  = p6(df,W2M,H2-32)', code)
code = re.sub(r'fig_p7\s*=\s*p7\(df,W2L,H2\)', r'fig_p7  = p7(df,W2L,H2-32)', code)
code = re.sub(r'fig_p8\s*=\s*p8\(pt,W2M,H2\)', r'fig_p8  = p8(pt,W2M,H2-32)', code)

code = re.sub(r'fig_p9\s*=\s*p9\(df,W3A,H3\)', r'fig_p9  = p9(df,W3A,H3-32)', code)
code = re.sub(r'fig_p10\s*=\s*p10\(pt,W3B,H3\)', r'fig_p10 = p10(pt,W3B,H3-32)', code)
code = re.sub(r'fig_p11\s*=\s*p11\(df,W3B,H3\)', r'fig_p11 = p11(df,W3B,H3-32)', code)
code = re.sub(r'fig_p12\s*=\s*p12\(df,W3A,H3\)', r'fig_p12 = p12(df,W3A,H3-32)', code)

code = re.sub(r'fig_p13\s*=\s*p13\(pt,W4A,H4A\)', r'fig_p13 = p13(pt,W4A,H4A-32)', code)
code = re.sub(r'fig_p14\s*=\s*p14\(pt,W4B,H4A\)', r'fig_p14 = p14(pt,W4B,H4A-32)', code)
code = re.sub(r'fig_p15\s*=\s*p15\(pt,W4C,H4B\)', r'fig_p15 = p15(pt,W4C,H4B-32)', code)
code = re.sub(r'fig_p16\s*=\s*p16\(df,pt,W4A\+W4B-W4C,H4B\)', r'fig_p16 = p16(df,pt,W4A+W4B-W4C,H4B-32)', code)

code = re.sub(r'fig_p17\s*=\s*p17\(df,pt,W5A,H5\)', r'fig_p17 = p17(df,pt,W5A,H5-32)', code)
code = re.sub(r'fig_p18\s*=\s*p18\(pt,W5B,H5\)', r'fig_p18 = p18(pt,W5B,H5-32)', code)
code = re.sub(r'fig_p19\s*=\s*p19\(pt,W5C,H5\)', r'fig_p19 = p19(pt,W5C,H5-32)', code)

# 2. Sửa file để inject Streamlit JS và logic Component
js_injection = """
<script src="https://cdn.jsdelivr.net/npm/streamlit-component-lib@1.3.0/dist/streamlit.js"></script>
<script>
function sendState() {
  var state = {
    model: document.getElementById('filter-model').value,
    benchmark: document.getElementById('filter-benchmark').value,
    theme: document.documentElement.getAttribute('data-theme'),
    page: window._currentPage || 'pg1'
  };
  Streamlit.setComponentValue(state);
}
function filterModel(v) { sendState(); }
function filterBenchmark(v) { sendState(); }
</script>
"""
code = code.replace('<script>\n(function(){\n  window.switchPage', js_injection + '\n<script>\n(function(){\n  window.switchPage')

# 3. Thay đổi HTML để gắn selected state
# In f-strings we can use variables, so we define dict `state`
code = code.replace('<html lang="vi" data-theme="dark">', '<html lang="vi" data-theme="{state[\'theme\']}">')

# Replace options safely
def sel(key, val): return '{ "selected" if state["' + key + '"]=="' + val + '" else "" }'

# Models
m_tag = '<select class="fsel" id="filter-model" onchange="filterModel(this.value)">'
code = code.replace('<select class="fsel"><option>All Models</option>', f'{m_tag}<option value="All Models" {sel("model","All Models")}>All Models</option>')
code = code.replace('<option>minimax-m2.5</option>', f'<option value="minimax-m2.5" {sel("model","minimax-m2.5")}>minimax-m2.5</option>')
code = code.replace('<option>claude-sonnet</option>', f'<option value="claude-sonnet" {sel("model","claude-sonnet")}>claude-sonnet</option>')
code = code.replace('<option>deepseek-v3.1</option>', f'<option value="deepseek-v3.1" {sel("model","deepseek-v3.1")}>deepseek-v3.1</option>')
code = code.replace('<option>claude-opus</option>', f'<option value="claude-opus" {sel("model","claude-opus")}>claude-opus</option>')

# Benchmarks
b_tag = '<select class="fsel" id="filter-benchmark" onchange="filterBenchmark(this.value)">'
code = code.replace('<select class="fsel"><option>All Benchmarks</option>', f'{b_tag}<option value="All Benchmarks" {sel("benchmark","All Benchmarks")}>All Benchmarks</option>')
code = code.replace('<option>swebench</option>', f'<option value="swebench" {sel("benchmark","swebench")}>swebench</option>')
code = code.replace('<option>gaia</option>', f'<option value="gaia" {sel("benchmark","gaia")}>gaia</option>')
code = code.replace('<option>wildclaw</option>', f'<option value="wildclaw" {sel("benchmark","wildclaw")}>wildclaw</option>')

# Theme toggler
thm_lbl = '{ "Command" if state["theme"]=="dark" else "Boardroom" }'
code = code.replace('<span class="tchip" onclick="toggleTheme()">Command</span>', f'<span class="tchip" onclick="toggleTheme(); setTimeout(sendState, 50)">{thm_lbl}</span>')

# Khôi phục tab state
def act(pg): return '{ "active" if state["page"]=="' + pg + '" else "" }'
code = code.replace('<div class="tab active" data-page="pg1"', f'<div class="tab {act("pg1")}" data-page="pg1"')
code = code.replace('<div class="tab" data-page="pg2"', f'<div class="tab {act("pg2")}" data-page="pg2"')
code = code.replace('<div class="tab" data-page="pg3"', f'<div class="tab {act("pg3")}" data-page="pg3"')
code = code.replace('<div class="tab" data-page="pg4"', f'<div class="tab {act("pg4")}" data-page="pg4"')
code = code.replace('<div class="tab" data-page="pg5"', f'<div class="tab {act("pg5")}" data-page="pg5"')

# Inject active page into JS
code = code.replace("var fp=document.getElementById('pg1');", "var page_id = \"{state['page']}\";\n    var fp=document.getElementById(page_id);")
code = code.replace("if(i===0){p.style.display='flex';", "if(p.id===page_id){p.style.display='flex';")
code = code.replace("var ft=document.querySelector('.tab');", "var ft=document.querySelector('[data-page=\"'+page_id+'\"]');")
code = code.replace("window.switchPage=function(id){", "window.switchPage=function(id){\n    window._currentPage = id; setTimeout(sendState, 50);")

# 4. Sửa hàm main() để dùng filter
main_replacement = """
def render_dashboard_component(html_str, key="pbi"):
    import os
    import streamlit.components.v1 as components
    comp_dir = os.path.join(os.path.dirname(__file__), "pbi_component")
    os.makedirs(comp_dir, exist_ok=True)
    with open(os.path.join(comp_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_str)
    pbi_comp = components.declare_component("pbi_dashboard", path=comp_dir)
    return pbi_comp(key=key)

def main():
    if "pbi_state" not in st.session_state:
        st.session_state.pbi_state = {"model": "All Models", "benchmark": "All Benchmarks", "theme": "dark", "page": "pg1"}
    
    state = st.session_state.pbi_state
    df, pt, meta = load_data()
    
    if state["model"] != "All Models":
        df = df[df["mg"] == state["model"]]
        pt = pt[pt["model"] == state["model"]]
    if state["benchmark"] != "All Benchmarks":
        df = df[df["benchmark"] == state["benchmark"]]
        pt = pt[pt["benchmark"] == state["benchmark"]]
        
    if len(pt) > 0:
        meta["total_cost"] = pt["final_cost"].sum()
        meta["n_tasks"] = len(pt)
        meta["rr"] = pt["resolved"].mean() * 100
        meta["avg_cost_r"] = pt[pt["resolved"] == 1]["final_cost"].mean()
        if pd.isna(meta["avg_cost_r"]): meta["avg_cost_r"] = 0.0
        meta["wasted"] = pt["wasted"].sum()
        meta["spikes"] = int(df["is_spike"].sum())
        meta["n_models"] = pt["model"].nunique()
    else:
        for k in ["total_cost", "n_tasks", "rr", "avg_cost_r", "wasted", "spikes", "n_models"]:
            meta[k] = 0

    n = meta["n_tasks"]; today = meta["today"]
"""
code = code.replace("def main():\n    df, pt, meta = load_data()\n    n = meta[\"n_tasks\"]; today = meta[\"today\"]", main_replacement)

# Cập nhật component call
render_call = """
    new_state = render_dashboard_component(html)
    if new_state and new_state != st.session_state.pbi_state:
        st.session_state.pbi_state = new_state
        st.rerun()
"""
code = re.sub(r'    import streamlit\.components\.v1 as components\n.*?components\.html\(html, height=760, scrolling=False\)', render_call, code, flags=re.DOTALL)

with open('analysis_dashboard_fixed.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Created analysis_dashboard_fixed.py")
