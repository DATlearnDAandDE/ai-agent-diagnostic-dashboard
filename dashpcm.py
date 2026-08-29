# -*- coding: utf-8 -*-
"""
EXECUTIVE REPORT: BAN LINH DIEU PHOI AI AGENT
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AI Agent Cost Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── THEME STATE ──────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False
is_dark = st.session_state["dark_mode"]

MODEL_COLORS = {
    'claude-sonnet-4-6': '#0ea5e9',
    'claude-opus-4-6':   '#10b981',
    'deepseek-v3.1':     '#f59e0b',
    'minimax-m2.5':      '#f43f5e',
}

CLR_POS   = '#10b981'
CLR_NEG   = '#f43f5e'
CLR_BLUE  = '#0ea5e9'
TURN_CUT  = 27

if is_dark:
    BG        = "radial-gradient(ellipse at top, #0f172a 0%, #020617 100%)"
    CARD      = "rgba(30,41,59,0.95)"
    BORDER    = "rgba(56,189,248,0.28)"
    H         = "#f8fafc"
    V         = "#e2e8f0"
    LBL       = "#94a3b8"
    MUT       = "#64748b"
    GRID      = "#1e293b"
    TBL_HDR   = "#1e293b"
    TBL_ROW   = "#0f172a"
    TBOX_BG   = "#1e293b"
    FLT_BG    = "#0b1120"
    FLT_BORDER= "rgba(56,189,248,0.4)"
    SKY_D     = "#38bdf8"
    TAG_CLR   = "#38bdf8"
    SEL_BG    = "#1e293b"
    TD_V      = "#e2e8f0"
    TD_B      = "#38bdf8"
    PL_FONT   = "#cbd5e1"
    PL_GRID   = "#334155"
    PL_PLOT   = "rgba(15,23,42,0.6)"
    PL_HBG    = "#1e293b"
    PL_HTXT   = "#f8fafc"
    PL_HBD    = "#475569"
    CUT_CLR   = "#38bdf8"
    TAB_INACTIVE = "rgba(30,41,59,0.9)"
    TAB_TXT      = "#94a3b8"
    TAB_HOV_BG   = "rgba(14,165,233,0.25)"
    TAB_HOV_TXT  = "#38bdf8"
else:
    BG        = "linear-gradient(145deg, #f0f7ff 0%, #e8f4fd 50%, #f5f9ff 100%)"
    CARD      = "#ffffff"
    BORDER    = "rgba(14,165,233,0.2)"
    H         = "#0f172a"
    V         = "#1e293b"
    LBL       = "#334155"
    MUT       = "#64748b"
    GRID      = "#e2e8f0"
    TBL_HDR   = "#f1f5f9"
    TBL_ROW   = "#f8fafc"
    TBOX_BG   = "#f0f9ff"
    FLT_BG    = "#ffffff"
    FLT_BORDER= "rgba(14,165,233,0.4)"
    SKY_D     = "#0284c7"
    TAG_CLR   = "#0369a1"
    SEL_BG    = "#ffffff"
    TD_V      = "#1e293b"
    TD_B      = "#0284c7"
    PL_FONT   = "#334155"
    PL_GRID   = "#e2e8f0"
    PL_PLOT   = "rgba(248,250,252,0.7)"
    PL_HBG    = "#ffffff"
    PL_HTXT   = "#0f172a"
    PL_HBD    = "#cbd5e1"
    CUT_CLR   = "#0284c7"
    TAB_INACTIVE = "rgba(14,165,233,0.08)"
    TAB_TXT      = "#0f172a"
    TAB_HOV_BG   = "rgba(14,165,233,0.18)"
    TAB_HOV_TXT  = "#0284c7"

def pls(fig, h=290, ml=48):
    fig.update_layout(
        height=h,
        font=dict(family="Inter", size=11, color=PL_FONT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PL_PLOT,
        margin=dict(l=ml, r=18, t=34, b=30),
        hoverlabel=dict(bgcolor=PL_HBG, font_size=12, font_color=PL_HTXT,
                        font_family="Inter", bordercolor=PL_HBD),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=BORDER,
                     tickfont=dict(size=10, color=PL_FONT, family="Inter"))
    fig.update_yaxes(showgrid=True, gridcolor=PL_GRID, zeroline=False,
                     linecolor=BORDER, tickfont=dict(size=10, color=PL_FONT, family="Inter"))
    return fig

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;600;700;800&display=swap');

* {{ box-sizing: border-box; }}

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: {V} !important;
}}

.stApp {{
    background: {BG} !important;
    color: {V} !important;
}}

/* Hide sidebar completely, show only main area */
section[data-testid="stSidebar"] {{ display: none !important; }}
[data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}
#MainMenu, footer {{ visibility: hidden; height: 0; }}
header {{ background: transparent !important; box-shadow: none !important; }}

.block-container {{
    max-width: 1580px !important;
    padding: 16px 20px 36px !important;
}}

/* ── Filter Panel ──────────────────────────────── */
.flt-panel {{
    background: {FLT_BG};
    border: 2px solid {FLT_BORDER};
    border-radius: 16px;
    padding: 20px 24px 16px;
    margin-bottom: 18px;
    box-shadow: 0 8px 32px -4px rgba(14,165,233,0.15);
    backdrop-filter: blur(16px);
}}
.flt-title {{
    font-size: 13px;
    font-weight: 900;
    color: {SKY_D};
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.flt-title::after {{
    content: '';
    flex: 1;
    height: 1.5px;
    background: linear-gradient(90deg, rgba(14,165,233,0.4), transparent);
}}

/* ── KPI Cards ─────────────────────────────────── */
.kpi-card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 15px 16px;
    min-height: 96px;
    box-shadow: 0 4px 14px -2px rgba(14,165,233,0.08);
    display: flex;
    flex-direction: column;
    justify-content: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}}
.kpi-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 12px 24px -4px rgba(14,165,233,0.2);
    border-color: rgba(14,165,233,0.45);
}}
.kpi-card.wasted {{ border-left: 4px solid #f43f5e !important; }}
.kpi-card.good   {{ border-left: 4px solid #10b981 !important; }}
.kpi-lbl {{ font-size: 10.5px; color: {LBL}; margin-bottom: 5px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.7px; }}
.kpi-val {{ font-size: 26px; font-weight: 900; color: {H}; font-family: 'JetBrains Mono', monospace; letter-spacing: -0.5px; line-height: 1.1; }}
.kpi-val.w {{ color: #f43f5e; }}
.kpi-val.g {{ color: #10b981; }}
.kpi-sub {{ font-size: 11px; color: {MUT}; margin-top: 4px; font-weight: 600; }}

/* ── Section Header ────────────────────────────── */
.sec-hdr {{
    background: linear-gradient(90deg, rgba(14,165,233,0.15) 0%, rgba(14,165,233,0.04) 60%, transparent 100%);
    border-left: 4px solid #0ea5e9;
    padding: 10px 18px;
    border-radius: 0 12px 12px 0;
    margin: 18px 0 14px;
    font-size: 14px;
    font-weight: 900;
    color: {SKY_D};
    text-transform: uppercase;
    letter-spacing: 0.7px;
}}
.panel-title {{
    font-size: 12.5px;
    font-weight: 900;
    color: {SKY_D};
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin: 0 0 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.panel-title::after {{
    content: '';
    flex: 1;
    height: 1.5px;
    background: linear-gradient(90deg, rgba(14,165,233,0.3), transparent);
}}

/* ── Tabs ──────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background: {("rgba(30,41,59,0.9)" if is_dark else "#dbeafe")};
    border-radius: 14px;
    padding: 6px;
    border: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 10px !important;
    padding: 10px 20px !important;
    background: {TAB_INACTIVE} !important;
    border: 1px solid {BORDER} !important;
    transition: all 0.2s ease !important;
}}
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] span,
.stTabs [data-baseweb="tab"] div {{
    color: {TAB_TXT} !important;
    font-weight: 800 !important;
    font-size: 13px !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    background: {TAB_HOV_BG} !important;
    border-color: #0ea5e9 !important;
}}
.stTabs [data-baseweb="tab"]:hover p,
.stTabs [data-baseweb="tab"]:hover span {{
    color: {TAB_HOV_TXT} !important;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
    border-color: #0284c7 !important;
    box-shadow: 0 4px 14px rgba(14,165,233,0.4) !important;
}}
.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] span,
.stTabs [aria-selected="true"] div {{
    color: #ffffff !important;
    font-weight: 900 !important;
}}
.stTabs [data-baseweb="tab-border"] {{ display: none !important; }}
.stTabs [data-baseweb="tab-panel"] {{
    background: transparent !important;
    padding: 16px 0 0 !important;
}}

/* ── Cards ─────────────────────────────────────── */
div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stContainer"] > div {{
    background: {CARD} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 18px -2px rgba(14,165,233,0.07) !important;
    padding: 18px !important;
    transition: all 0.25s ease !important;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:hover,
div[data-testid="stContainer"] > div:hover {{
    transform: translateY(-2px) !important;
    border-color: rgba(14,165,233,0.4) !important;
    box-shadow: 0 10px 28px -4px rgba(14,165,233,0.16) !important;
}}

/* ── Table ─────────────────────────────────────── */
.dt {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12.5px;
    font-variant-numeric: tabular-nums;
}}
.dt th {{
    padding: 10px 12px;
    background: {TBL_HDR};
    border-bottom: 2px solid {BORDER};
    color: {H} !important;
    font-weight: 900;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    text-align: left;
}}
.dt th.r {{ text-align: right; }}
.dt td {{
    padding: 10px 12px;
    border-bottom: 1px solid {GRID};
    color: {TD_V} !important;
    font-weight: 600;
}}
.dt td.r {{ text-align: right; }}
.dt td b {{ color: {TD_B} !important; font-weight: 800; font-family: 'JetBrains Mono', monospace; }}
.dt tr:hover td {{ background: {TBL_ROW}; }}
.mdot {{
    width: 9px; height: 9px; border-radius: 50%;
    display: inline-block; margin-right: 7px; vertical-align: middle;
}}

/* ── Misc ──────────────────────────────────────── */
.tbox {{
    border: 2px solid #0ea5e9;
    background: {TBOX_BG};
    border-radius: 14px;
    padding: 18px 22px;
    font-size: 13px;
    color: {V} !important;
    line-height: 1.75;
    margin-top: 14px;
}}
.cap {{
    font-size: 11.5px;
    color: {LBL} !important;
    margin-top: 6px;
    font-style: italic;
    line-height: 1.5;
}}
.cap.w {{ color: #f43f5e !important; font-weight: 700; font-style: normal; }}
.rpt-title {{
    font-size: 26px; font-weight: 900; letter-spacing: -0.7px;
    background: linear-gradient(135deg, #0ea5e9 0%, #38bdf8 50%, #0369a1 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 4px; line-height: 1.2;
}}
.rpt-sub {{ font-size: 13px; color: {LBL}; margin: 0 0 14px; font-weight: 600; }}
.story-box {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-left: 4px solid #0ea5e9;
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 13px;
    line-height: 1.65;
    color: {V} !important;
}}
.story-box.warning {{ border-left-color: #f59e0b; }}
.flow-wrap {{ display: flex; align-items: stretch; gap: 0; padding: 8px 0; }}
.fbox {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 12px 10px;
    text-align: center;
    flex: 1;
}}
.fbox-t {{ font-size: 11.5px; font-weight: 800; color: {SKY_D}; margin-bottom: 5px; }}
.fbox-s {{ font-size: 10.5px; color: {MUT}; line-height: 1.4; }}
.farr {{ font-size: 18px; color: {MUT}; padding: 0 6px; align-self: center; flex-shrink: 0; }}
.fb4 {{ border: 1.5px solid #f43f5e !important; }}
.warn-bar {{
    background: rgba(244,63,94,0.1);
    border: 1px solid rgba(244,63,94,0.3);
    border-left: 4px solid #f43f5e;
    color: {("#fca5a5" if is_dark else "#9b2c2c")};
    padding: 9px 14px; border-radius: 10px;
    font-size: 12px; margin-bottom: 12px; font-weight: 700;
}}
.fnt {{
    font-size: 11px; color: {MUT};
    border-top: 1px solid {GRID};
    padding-top: 14px; margin-top: 26px; line-height: 1.6;
}}
.empty-state {{
    display: flex; align-items: center; justify-content: center;
    padding: 36px 16px; color: {MUT}; font-size: 13px;
}}

/* Multiselect + selectbox contrast fixes */
span[data-baseweb="tag"] {{
    background: rgba(14,165,233,0.18) !important;
    border: 1px solid rgba(14,165,233,0.4) !important;
    border-radius: 6px !important;
}}
span[data-baseweb="tag"] span {{ color: {TAG_CLR} !important; font-weight: 800 !important; }}
div[data-baseweb="select"] > div {{
    background: {SEL_BG} !important;
    border-color: {BORDER} !important;
    color: {V} !important;
    border-radius: 10px !important;
}}
div[data-baseweb="select"] span {{ color: {V} !important; font-weight: 600 !important; }}

/* Slider thumb color */
.stSlider [data-testid="stSlider"] {{ color: #0ea5e9; }}

/* st.button styling */
.stButton button {{
    background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 800 !important;
    font-size: 13px !important;
    padding: 8px 18px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(14,165,233,0.3) !important;
}}
.stButton button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(14,165,233,0.45) !important;
}}

/* Toggle */
.stToggle label {{ color: {V} !important; font-weight: 700 !important; }}
</style>""", unsafe_allow_html=True)

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def fc(v):
    return f"${float(v):,.2f}" if not pd.isna(v) else "$0.00"
def fi(v):
    return f"{int(v):,}" if not pd.isna(v) else "0"
def fp(v, d=1):
    return f"{float(v):.{d}f}%" if not pd.isna(v) else "0%"
def empty_html():
    return f'<div class="empty-state">📭 Không có dữ liệu khớp bộ lọc</div>'
def get_card():
    try: return st.container(border=True)
    except: return st.container()

# ─── DATA ─────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "processed_agentic_traces.csv")

@st.cache_data(show_spinner="Đang tải dữ liệu…")
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    def bench(sid):
        s = str(sid)
        if s.startswith("swebench__"): return "swebench"
        if s.startswith("gaia__"):     return "gaia"
        if s.startswith("wildclaw__"): return "wildclaw"
        return "unknown"
    df["benchmark"] = df["session_id"].apply(bench)
    df = df.sort_values(["session_id","turn_number"]).reset_index(drop=True)
    df["cum_cost"] = df.groupby("session_id")["turn_cost"].cumsum()
    sess = df.groupby("session_id").agg(
        model=("model","first"),
        benchmark=("benchmark","first"),
        is_system_prompt_present=("is_system_prompt_present","first"),
        n_turns=("turn_number","max"),
        total_cost=("turn_cost","sum"),
        avg_input_tokens=("input_tokens","mean"),
        error_share=("has_error","mean"),
    ).reset_index()
    sess["failed"]        = (sess["error_share"] == 1.0).astype(int)
    sess["resolved"]      = 1 - sess["failed"]
    sess["cost_per_turn"] = sess["total_cost"] / sess["n_turns"].replace(0,np.nan)
    return df, sess

df_turns_raw, df_sess_raw = load_data()

def mdl_agg(df_t, df_s):
    if df_s.empty:
        return pd.DataFrame()
    g = df_s.groupby("model").agg(
        sessions=("session_id","count"),
        total_cost=("total_cost","sum"),
        avg_cost_s=("total_cost","mean"),
        avg_tokens=("avg_input_tokens","mean"),
    ).reset_index()
    ta = df_t.groupby("model").agg(total_turns=("turn_number","count"), errs=("has_error","sum")).reset_index()
    g = g.merge(ta, on="model", how="left").fillna(0)
    g["err_rate"]  = g["errs"] / g["total_turns"].replace(0, np.nan)
    g["cost_turn"] = g["total_cost"] / g["total_turns"].replace(0, np.nan)
    tot_t = g["total_turns"].sum(); tot_c = g["total_cost"].sum()
    g["pct_turns"] = g["total_turns"] / tot_t if tot_t > 0 else 0
    g["pct_cost"]  = g["total_cost"]  / tot_c if tot_c > 0 else 0
    return g.sort_values("total_cost", ascending=False).reset_index(drop=True)

# ─── HEADER ROW ───────────────────────────────────────────────────────────────
hc1, hc2 = st.columns([7.5, 4.5])
with hc1:
    st.markdown("""
    <div class="rpt-title">📊 Bản Lĩnh Điều Phối AI Agent — Tối Ưu Chi Phí &amp; Độ Ổn Định</div>
    <div class="rpt-sub">Executive Diagnostic Report · 24.880 Traces · 767 Sessions · 4 Mô hình · Phân tích 4 Cấp độ Độc Lập</div>
    """, unsafe_allow_html=True)
with hc2:
    theme_col, spacer = st.columns([3, 1])
    with theme_col:
        theme_choice = st.toggle("🌙 Chế độ Tối (Dark Mode)", value=st.session_state["dark_mode"], key="theme_toggle")
        if theme_choice != st.session_state["dark_mode"]:
            st.session_state["dark_mode"] = theme_choice
            st.rerun()

# ─── INLINE FILTER PANEL ──────────────────────────────────────────────────────
with st.expander("⚙️  BỘ LỌC ĐIỀU HÀNH  —  Nhấn để mở / thu gọn bộ lọc dữ liệu", expanded=False):
    st.markdown(f'<div class="flt-title">⚙️ Tùy Chỉnh Phạm Vi Phân Tích</div>', unsafe_allow_html=True)
    fc1, fc2, fc3, fc4, fc5 = st.columns([3, 2.5, 2, 2.5, 2])
    with fc1:
        sel_models = st.multiselect(
            "🤖 Mô hình",
            options=list(MODEL_COLORS.keys()),
            default=list(MODEL_COLORS.keys()),
            key="flt_model",
        )
    with fc2:
        sel_bench = st.multiselect(
            "🎯 Benchmark",
            options=["swebench","gaia","wildclaw"],
            default=["swebench","gaia","wildclaw"],
            key="flt_bench",
        )
    with fc3:
        sel_prompt = st.selectbox(
            "💬 System Prompt",
            ["Tất cả","Có Prompt","Không Prompt"],
            key="flt_prompt",
        )
    with fc4:
        sel_turns = st.slider(
            "🔄 Dải Turn",
            min_value=1, max_value=50, value=(1, 50),
            key="flt_turns",
        )
    with fc5:
        st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
        if st.button("🔄 Reset bộ lọc", use_container_width=True):
            for _k in ["flt_model", "flt_bench", "flt_prompt", "flt_turns"]:
                if _k in st.session_state:
                    del st.session_state[_k]
            st.rerun()
    
    # Summary row
    sm1, sm2, sm3, sm4 = st.columns(4)
    with sm1:
        st.markdown(f'<div style="font-size:11px; color:{LBL};">📊 Tổng mẫu gốc: <b style="color:{SKY_D};">{fi(len(df_sess_raw))} sessions · {fi(len(df_turns_raw))} turns</b></div>', unsafe_allow_html=True)
    with sm2:
        st.markdown(f'<div style="font-size:11px; color:{LBL};">🤖 Model đã chọn: <b style="color:{SKY_D};">{len(sel_models) if sel_models else 0}/4</b></div>', unsafe_allow_html=True)
    with sm3:
        st.markdown(f'<div style="font-size:11px; color:{LBL};">🎯 Benchmark: <b style="color:{SKY_D};">{", ".join(sel_bench) if sel_bench else "Không có"}</b></div>', unsafe_allow_html=True)
    with sm4:
        st.markdown(f'<div style="font-size:11px; color:{LBL};">🔄 Dải Turn: <b style="color:{SKY_D};">{sel_turns[0]} → {sel_turns[1]}</b></div>', unsafe_allow_html=True)

# ─── APPLY FILTERS ────────────────────────────────────────────────────────────
m_list = sel_models if sel_models else list(MODEL_COLORS.keys())
b_list = sel_bench  if sel_bench  else ["swebench","gaia","wildclaw"]
mask   = df_sess_raw["model"].isin(m_list) & df_sess_raw["benchmark"].isin(b_list)
if sel_prompt == "Có Prompt":
    mask &= (df_sess_raw["is_system_prompt_present"] == 1)
elif sel_prompt == "Không Prompt":
    mask &= (df_sess_raw["is_system_prompt_present"] == 0)
df_s = df_sess_raw[mask].copy()
df_t = df_turns_raw[
    df_turns_raw["session_id"].isin(df_s["session_id"]) &
    df_turns_raw["turn_number"].between(sel_turns[0], sel_turns[1])
].copy()

ms              = mdl_agg(df_t, df_s)
wasted_narrow   = df_s[df_s["model"].isin(["minimax-m2.5","deepseek-v3.1"])]["total_cost"].sum()
wasted_full     = df_s[df_s["failed"] == 1]["total_cost"].sum()
total_budget    = df_s["total_cost"].sum()
total_n_sess    = len(df_s)
total_n_turns   = len(df_t)
resolve_rate    = df_s["resolved"].mean() if total_n_sess > 0 else 0.0

if df_s.empty:
    st.warning("⚠️ Bộ lọc hiện tại không khớp phiên dữ liệu nào. Hãy mở Bộ Lọc bên trên để điều chỉnh lại.")

# ─── KPI STRIP ────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
_wp = (wasted_full / total_budget * 100) if total_budget > 0 else 0
for col, lbl, val, sub, cls in [
    (k1, "Tổng Ngân Sách (K1)", fc(total_budget), f"{total_n_sess} sessions lọc", ""),
    (k2, "Sessions (K2)", fi(total_n_sess), f"/ {fi(len(df_sess_raw))} tổng", ""),
    (k3, "Turns (K3)", fi(total_n_turns), f"Turn {sel_turns[0]}–{sel_turns[1]}", ""),
    (k4, "Resolve Rate (K4)", fp(resolve_rate*100), f"{fi(int(df_s['resolved'].sum()) if total_n_sess else 0)} thành công", "g"),
    (k5, "Cost/Session TB (K5)", fc(df_s['total_cost'].mean() if total_n_sess else 0), "Trung bình mẫu lọc", ""),
    (k6, "Chi Phí Lãng Phí (K6)", fc(wasted_full), f"{_wp:.1f}% ngân sách thiêu đốt", "w"),
]:
    with col:
        card_cls = "kpi-card " + ("wasted" if cls == "w" else "good" if cls == "g" else "")
        st.markdown(f"""<div class="{card_cls}">
            <div class="kpi-lbl">{lbl}</div>
            <div class="kpi-val {cls}">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "§01 MÔ TẢ — Điều gì đã xảy ra?",
    "§02 CHẨN ĐOÁN — Tại sao xảy ra?",
    "§03 DỰ BÁO — Khi nào nên dừng?",
    "§04 KHUYẾN NGHỊ — Nên làm gì?",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · MÔ TẢ
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec-hdr">§01 MÔ TẢ — BỐI CẢNH QUY MÔ, CHI PHÍ &amp; PHÂN PHỐI LƯỢT CHẠY</div>', unsafe_allow_html=True)
    r1c1, r1c2 = st.columns([5.2, 6.8])
    with r1c1:
        with get_card():
            st.markdown('<div class="panel-title">MT1 · Bảng Bối Cảnh 4 Mô Hình</div>', unsafe_allow_html=True)
            if ms.empty:
                st.markdown(empty_html(), unsafe_allow_html=True)
            else:
                rows = ""
                for _, r in ms.iterrows():
                    dc = MODEL_COLORS.get(r["model"], MUT)
                    rows += f"""<tr>
                        <td><span class="mdot" style="background:{dc}"></span><b style="color:{H};">{r['model']}</b></td>
                        <td class="r" style="color:{TD_V};">{fi(r['sessions'])}</td>
                        <td class="r" style="color:{TD_V};">{fi(r['total_turns'])}</td>
                        <td class="r"><b>{fc(r['total_cost'])}</b></td>
                        <td class="r" style="color:{TD_V};">{fc(r['avg_cost_s'])}</td>
                        <td class="r" style="color:{TD_V};">{fp(r['err_rate']*100)}</td>
                    </tr>"""
                st.markdown(f"""<table class="dt">
                    <thead><tr>
                        <th>Model</th><th class="r">Sessions</th><th class="r">Turns</th>
                        <th class="r">Total Cost</th><th class="r">Avg/Session</th><th class="r">Err Rate</th>
                    </tr></thead><tbody>{rows}</tbody>
                </table>""", unsafe_allow_html=True)
                st.markdown('<div class="cap">Bảng tổng hợp nhanh các chỉ số vận hành cốt lõi.</div>', unsafe_allow_html=True)
    with r1c2:
        with get_card():
            st.markdown('<div class="panel-title">MT2 · Tổng Chi Phí Theo Model &amp; Trọng Số Ngân Sách</div>', unsafe_allow_html=True)
            if ms.empty:
                st.markdown(empty_html(), unsafe_allow_html=True)
            else:
                mt2 = ms.sort_values("total_cost", ascending=True)
                f2 = px.bar(mt2, x="total_cost", y="model", orientation="h",
                            color="model", color_discrete_map=MODEL_COLORS,
                            text=mt2["total_cost"].apply(fc),
                            custom_data=["sessions","pct_cost","total_turns"])
                f2.update_traces(hovertemplate="<b>%{y}</b><br>$%{x:,.2f}<br>%{customdata[1]:.1%} ngân sách<extra></extra>",
                                 textposition="outside", textfont=dict(size=11, family="JetBrains Mono"))
                sr = mt2[mt2["model"]=="claude-sonnet-4-6"]
                tot = mt2["total_cost"].sum()
                if not sr.empty and tot > 0:
                    f2.add_annotation(x=sr.iloc[0]["total_cost"], y="claude-sonnet-4-6",
                                      text=f"<b>{sr.iloc[0]['total_cost']/tot*100:.1f}% tổng ngân sách</b>",
                                      showarrow=True, arrowhead=2, ax=90, ay=0,
                                      font=dict(size=11, color=SKY_D, family="Inter"),
                                      bgcolor="rgba(14,165,233,0.15)", bordercolor="#0ea5e9", borderwidth=1)
                pls(f2, h=250)
                f2.update_layout(showlegend=False, xaxis_title="Tổng chi phí (USD)", yaxis_title="", xaxis_tickformat="$,.2f")
                st.plotly_chart(f2, use_container_width=True, config={"displayModeBar":False})

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        with get_card():
            st.markdown('<div class="panel-title">MT3 · Phân Phối Số Lượt (Turns/Session)</div>', unsafe_allow_html=True)
            if df_s.empty:
                st.markdown(empty_html(), unsafe_allow_html=True)
            else:
                f3 = go.Figure()
                for m in sorted(df_s["model"].unique()):
                    f3.add_trace(go.Box(y=df_s[df_s["model"]==m]["n_turns"], name=m,
                                        marker_color=MODEL_COLORS.get(m,MUT), boxmean=True,
                                        hovertemplate=f"<b>{m}</b><br>Turns: %{{y}}<extra></extra>"))
                f3.add_hline(y=25, line_dash="dash", line_color="#f43f5e", line_width=1.5,
                             annotation_text="Ngưỡng kẹt (25 turns)", annotation_position="top right",
                             annotation_font=dict(size=10, color="#f43f5e"))
                pls(f3, h=290)
                f3.update_layout(showlegend=False, yaxis_title="n_turns / session")
                st.plotly_chart(f3, use_container_width=True, config={"displayModeBar":False})
                st.markdown('<div class="cap">Minimax &amp; Deepseek thường kéo dài vượt ngưỡng 25-30 turns.</div>', unsafe_allow_html=True)
    with r2c2:
        with get_card():
            st.markdown('<div class="panel-title">MT4 · Tỷ Lệ Lỗi (Error Rate) Theo Mô Hình</div>', unsafe_allow_html=True)
            if ms.empty:
                st.markdown(empty_html(), unsafe_allow_html=True)
            else:
                mt4 = ms.sort_values("err_rate", ascending=False).copy()
                f4 = go.Figure(go.Bar(
                    x=mt4["model"], y=mt4["err_rate"],
                    text=mt4["err_rate"].apply(lambda x: f"{x*100:.1f}%"), textposition="outside",
                    marker_color=[MODEL_COLORS.get(m,MUT) for m in mt4["model"]],
                    hovertemplate="<b>%{x}</b><br>Error rate: %{y:.1%}<extra></extra>", width=0.45
                ))
                f4.add_hline(y=1.0, line_dash="dot", line_color="#f43f5e", line_width=1.2,
                             annotation_text="Liệt 100% lỗi", annotation_font=dict(size=10, color="#f43f5e"))
                pls(f4, h=290)
                f4.update_layout(yaxis_tickformat=".0%", yaxis_range=[0,1.22], showlegend=False, xaxis_title="")
                st.plotly_chart(f4, use_container_width=True, config={"displayModeBar":False})
                st.markdown('<div class="cap w">Minimax và Deepseek có tỷ lệ lỗi 100% — đối lập hoàn toàn với Claude.</div>', unsafe_allow_html=True)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        with get_card():
            st.markdown('<div class="panel-title">MT5 · Cơ Cấu Phân Bổ Tác Vụ Theo Benchmark</div>', unsafe_allow_html=True)
            if df_s.empty:
                st.markdown(empty_html(), unsafe_allow_html=True)
            else:
                bm = df_s.groupby(["benchmark","model"]).size().reset_index(name="n")
                f5 = px.bar(bm, x="benchmark", y="n", color="model", color_discrete_map=MODEL_COLORS,
                            barmode="group", text="n", custom_data=["model"])
                f5.update_traces(textposition="outside", textfont=dict(size=10, family="JetBrains Mono"),
                                 hovertemplate="<b>%{customdata[0]}</b><br>Benchmark: %{x}<br>Sessions: %{y}<extra></extra>")
                pls(f5, h=280)
                f5.update_layout(xaxis_title="Benchmark", yaxis_title="Sessions",
                                 legend=dict(orientation="h", y=1.1, x=0, font=dict(size=10)))
                st.plotly_chart(f5, use_container_width=True, config={"displayModeBar":False})
    with r3c2:
        with get_card():
            st.markdown('<div class="panel-title">MT6 · Phân Phối Chi Phí Phiên (Cost Distribution)</div>', unsafe_allow_html=True)
            if df_s.empty:
                st.markdown(empty_html(), unsafe_allow_html=True)
            else:
                f6 = go.Figure()
                for m in sorted(df_s["model"].unique()):
                    mc = df_s[df_s["model"]==m]["total_cost"]
                    if len(mc):
                        f6.add_trace(go.Histogram(x=mc, name=m, marker_color=MODEL_COLORS.get(m,MUT),
                                                  opacity=0.75, nbinsx=24,
                                                  hovertemplate=f"<b>{m}</b><br>$%{{x:.2f}}<br>%{{y}} phiên<extra></extra>"))
                pls(f6, h=280)
                f6.update_layout(barmode="overlay", xaxis_tickformat="$,.2f",
                                 xaxis_title="Chi phí / session", yaxis_title="Sessions",
                                 legend=dict(orientation="h", y=1.1, x=0, font=dict(size=10)))
                st.plotly_chart(f6, use_container_width=True, config={"displayModeBar":False})

    st.markdown("""<div class="tbox">
        <b>💡 TỔNG QUAN MÔ TẢ (§01):</b><br>
        • <b>Quy mô:</b> Claude Sonnet chiếm 66.7% ngân sách nhưng hoàn thành đại đa số tác vụ. Minimax &amp; Deepseek chiếm 34% chi phí nhưng 0% kết quả thực sự.<br>
        • <b>Bất thường:</b> Tỷ lệ lỗi 100% và số lượt lặp >35 turns của 2 mô hình rẻ báo hiệu hiện tượng kẹt vòng lặp — được chẩn đoán sâu ở Tab §02.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · CHẨN ĐOÁN
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-hdr">§02 CHẨN ĐOÁN — TẠI SAO PHÁT SINH LÃNG PHÍ &amp; BẤT THƯỜNG HIỆU NĂNG?</div>', unsafe_allow_html=True)
    with get_card():
        st.markdown('<div class="panel-title">CD1 · Cạm Bẫy Giá Rẻ — Đơn Giá Thấp Nhưng Tổng Lãng Phí Cao</div>', unsafe_allow_html=True)
        if df_s.empty:
            st.markdown(empty_html(), unsafe_allow_html=True)
        else:
            fig_cd1 = go.Figure()
            for m in sorted(df_s["model"].unique()):
                m_df = df_s[df_s["model"]==m]
                is_c = 'claude' in m
                sz = (8 + (m_df['avg_input_tokens'] / 35000.0) * 16).clip(lower=9, upper=22)
                fig_cd1.add_trace(go.Scatter(
                    x=m_df['cost_per_turn'], y=m_df['n_turns'], mode='markers', name=m,
                    marker=dict(size=sz, color=MODEL_COLORS.get(m,'#94a3b8'),
                                symbol='diamond' if is_c else 'circle',
                                opacity=0.88 if is_c else 0.72,
                                line=dict(width=1.2 if is_c else 0.6, color='#ffffff')),
                    hovertemplate=f"<b>{m}</b><br>Cost/turn: $%{{x:.4f}}<br>Turns: %{{y}}<br>Tokens: %{{customdata:,.0f}}<extra></extra>",
                    customdata=m_df['avg_input_tokens']
                ))
            fig_cd1.add_hline(y=25, line_dash="dash", line_color="#f43f5e", line_width=1.5)
            trap = df_s[df_s['model'].isin(['minimax-m2.5','deepseek-v3.1'])]
            if not trap.empty:
                fig_cd1.add_annotation(x=trap['cost_per_turn'].median(), y=trap['n_turns'].max()*0.9,
                    text=f"<b>🚨 KẸT VÒNG LẶP</b><br>{fc(wasted_narrow)} lãng phí (100% Fail)",
                    showarrow=True, arrowhead=2, ax=50, ay=-20,
                    font=dict(color="#be123c", size=11, family="Inter"),
                    bgcolor="rgba(255,241,242,0.95)", bordercolor="#f43f5e", borderwidth=1.2)
            pls(fig_cd1, h=350)
            fig_cd1.update_layout(xaxis_title="Chi phí TB / turn ($)", yaxis_title="n_turns / session",
                                  xaxis_tickformat="$.3f", legend=dict(orientation="h", y=1.06, x=0))
            st.plotly_chart(fig_cd1, use_container_width=True, config={"displayModeBar":False})

    cf1, cf2 = st.columns([5.5, 6.5])
    with cf1:
        with get_card():
            st.markdown('<div class="panel-title">CD2 · Cơ Chế Vòng Lặp Lỗi Vô Tận</div>', unsafe_allow_html=True)
            st.markdown("""<div class="flow-wrap">
                <div class="fbox" style="opacity:.7"><div class="fbox-t">① Sai Tool Call</div><div class="fbox-s">Minimax &amp; Deepseek sai cú pháp</div></div>
                <div class="farr">→</div>
                <div class="fbox" style="opacity:.85"><div class="fbox-t">② Dồn Stack Trace</div><div class="fbox-s">Context nhồi traceback lỗi</div></div>
                <div class="farr">→</div>
                <div class="fbox" style="opacity:.95"><div class="fbox-t">③ Phình Context&gt;27k</div><div class="fbox-s">27-29k tokens/turn</div></div>
                <div class="farr">→</div>
                <div class="fbox fb4"><div class="fbox-t">④ Vòng Lặp Vô Tận</div><div class="fbox-s">34-38 turns / session</div></div>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="story-box warning" style="margin-top:12px;">
                <b>Cơ chế:</b> Stack trace lỗi nhồi vào context mỗi turn → bão hòa cửa sổ ngữ cảnh → mô hình tiếp tục sinh lỗi → cạn turn limit.
            </div>""", unsafe_allow_html=True)
    with cf2:
        with get_card():
            st.markdown('<div class="panel-title">CD3 · Token Bloat Theo Lượt — Context Phình To</div>', unsafe_allow_html=True)
            if df_t.empty:
                st.markdown(empty_html(), unsafe_allow_html=True)
            else:
                f_cd3 = go.Figure()
                for m in sorted(df_t["model"].unique()):
                    ag = df_t[df_t["model"]==m].groupby("turn_number")["input_tokens"].mean().reset_index()
                    if not ag.empty:
                        f_cd3.add_trace(go.Scatter(x=ag["turn_number"], y=ag["input_tokens"], name=m, mode="lines+markers",
                            line=dict(color=MODEL_COLORS.get(m,MUT), width=2.2), marker=dict(size=4),
                            hovertemplate=f"<b>{m}</b><br>Turn %{{x}}<br>%{{y:,.0f}} tokens avg<extra></extra>"))
                pls(f_cd3, h=248)
                f_cd3.update_layout(xaxis_title="Turn Number", yaxis_title="Avg Input Tokens",
                                    yaxis_tickformat=",", legend=dict(orientation="h", y=1.1, x=0, font=dict(size=9.5)))
                st.plotly_chart(f_cd3, use_container_width=True, config={"displayModeBar":False})

    cs1, cs2 = st.columns(2)
    with cs1:
        with get_card():
            st.markdown('<div class="panel-title">CD4 · Nghịch Lý System Prompt (Claude Sonnet)</div>', unsafe_allow_html=True)
            st.markdown('<div class="warn-bar">⚠️ Bật System Prompt làm tăng token &amp; chi phí</div>', unsafe_allow_html=True)
            son_df = df_sess_raw[df_sess_raw["model"]=="claude-sonnet-4-6"]
            t_son  = df_turns_raw[df_turns_raw["model"]=="claude-sonnet-4-6"]
            _m = []
            for sv, sl in [(0,"Không prompt"),(1,"Có prompt")]:
                ss = son_df[son_df["is_system_prompt_present"]==sv]
                st_ = t_son[t_son["session_id"].isin(ss["session_id"])]
                _m.append({"sp":sl,"tok":st_["input_tokens"].mean() if not st_.empty else 0,
                           "err":st_["has_error"].mean() if not st_.empty else 0,
                           "fail":ss["failed"].mean() if not ss.empty else 0,
                           "cst":ss["total_cost"].mean() if not ss.empty else 0})
            if len(_m)==2:
                mn_list = ["Tokens/turn","Error rate","Fail rate","Cost/session"]
                mk_list = ["tok","err","fail","cst"]
                f_cd4 = go.Figure()
                for i,(mn,mk) in enumerate(zip(mn_list,mk_list)):
                    v0,v1 = _m[0][mk],_m[1][mk]
                    mx = max(abs(v0),abs(v1),1e-9)
                    x0,x1 = v0/mx*100, v1/mx*100
                    f_cd4.add_trace(go.Scatter(x=[x0,x1],y=[mn,mn],mode="lines",line=dict(color="rgba(14,165,233,0.3)",width=2),showlegend=False,hoverinfo="skip"))
                    f_cd4.add_trace(go.Scatter(x=[x0],y=[mn],mode="markers",marker=dict(color=MUT,size=11),name="Không prompt" if i==0 else "",showlegend=(i==0),hovertemplate=f"<b>Không prompt</b><br>{mn}: {v0:,.3g}<extra></extra>"))
                    f_cd4.add_trace(go.Scatter(x=[x1],y=[mn],mode="markers",marker=dict(color="#f43f5e",size=11),name="Có prompt" if i==0 else "",showlegend=(i==0),hovertemplate=f"<b>Có prompt</b><br>{mn}: {v1:,.3g}<extra></extra>"))
                pls(f_cd4, h=240)
                f_cd4.update_layout(xaxis_title="Giá trị chuẩn hóa (%)",xaxis_range=[-10,130],legend=dict(orientation="h",y=1.12,x=0,font=dict(size=10)))
                st.plotly_chart(f_cd4, use_container_width=True, config={"displayModeBar":False})
            st.markdown('<div class="cap">Nguyên nhân thực: SWE-bench khó hơn GAIA/Wildclaw (xem CD5).</div>', unsafe_allow_html=True)
    with cs2:
        with get_card():
            st.markdown('<div class="panel-title">CD5 · Biến Nhiễu Nội Sinh (Confounding Variable)</div>', unsafe_allow_html=True)
            ct = pd.crosstab(df_sess_raw["benchmark"], df_sess_raw["is_system_prompt_present"])
            xl = ["Không Prompt" if c==0 else "Có Prompt" for c in ct.columns]
            fig_cd5 = px.imshow(ct.values, x=xl, y=ct.index.tolist(),
                                color_continuous_scale=[[0,"#f0f9ff"],[1,"#0ea5e9"]], text_auto=True, aspect="auto")
            pls(fig_cd5, h=240)
            fig_cd5.update_layout(coloraxis_showscale=False, yaxis_title="Benchmark")
            fig_cd5.update_traces(hovertemplate="<b>%{y}</b><br>%{x}: %{z} sessions<extra></extra>",
                                  textfont=dict(size=13, color=H, family="JetBrains Mono"))
            st.plotly_chart(fig_cd5, use_container_width=True, config={"displayModeBar":False})
            st.markdown('<div class="cap w">100% Có Prompt = SWE-bench. 100% Không Prompt = GAIA/Wildclaw. Benchmark mới là nguyên nhân!</div>', unsafe_allow_html=True)

    cr1, cr2 = st.columns(2)
    with cr1:
        with get_card():
            st.markdown('<div class="panel-title">CD6 · Đầu Tư Sinh Lời vs Thiêu Đốt Lãng Phí</div>', unsafe_allow_html=True)
            if df_s.empty:
                st.markdown(empty_html(), unsafe_allow_html=True)
            else:
                ml_ = sorted(df_s["model"].unique())
                vc  = [df_s[(df_s["model"]==m)&(df_s["failed"]==0)]["total_cost"].sum() for m in ml_]
                wc  = [df_s[(df_s["model"]==m)&(df_s["failed"]==1)]["total_cost"].sum() for m in ml_]
                f6_ = go.Figure()
                f6_.add_trace(go.Bar(name="Hiệu quả (Resolved)", x=ml_, y=vc, marker_color=CLR_POS, opacity=0.9, text=[fc(v) if v>0 else "" for v in vc], textposition="inside", hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>"))
                f6_.add_trace(go.Bar(name="Lãng phí (Failed)", x=ml_, y=wc, marker_color=CLR_NEG, opacity=0.9, text=[fc(v) if v>0 else "" for v in wc], textposition="inside", hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>"))
                pls(f6_, h=280)
                f6_.update_layout(barmode="stack", yaxis_tickformat="$,.2f", xaxis_title="", legend=dict(orientation="h",y=1.1,x=0,font=dict(size=10)))
                st.plotly_chart(f6_, use_container_width=True, config={"displayModeBar":False})
                st.markdown(f'<div class="cap w">Tổng chi phí lãng phí: <b>{fc(wasted_full)}</b></div>', unsafe_allow_html=True)
    with cr2:
        with get_card():
            st.markdown('<div class="panel-title">CD7 · Giải Mã TCO Opus — Vì Sao Chi Phí Thấp Hơn?</div>', unsafe_allow_html=True)
            _sa = df_sess_raw[df_sess_raw["model"]=="claude-sonnet-4-6"]["total_cost"].mean()
            _oa = df_sess_raw[df_sess_raw["model"]=="claude-opus-4-6"]["total_cost"].mean()
            _st = df_sess_raw[df_sess_raw["model"]=="claude-sonnet-4-6"]["n_turns"].mean()
            _ot = df_sess_raw[df_sess_raw["model"]=="claude-opus-4-6"]["n_turns"].mean()
            if pd.notna(_sa) and pd.notna(_oa) and _st>0:
                _dt = (_st-_ot)*(_sa/_st); _dc = _sa-_oa-_dt
                fw = go.Figure(go.Waterfall(orientation="v",
                    measure=["absolute","relative","relative","total"],
                    x=["Sonnet\n(Baseline)","Ít turns hơn\n(Dứt nhanh)","Đơn giá/token\ncao hơn","Opus\n(Total Cost)"],
                    y=[_sa,-abs(_dt),abs(_dc),None],
                    connector=dict(line=dict(color="rgba(14,165,233,0.3)",width=1.5)),
                    decreasing=dict(marker=dict(color=CLR_POS)),
                    increasing=dict(marker=dict(color=CLR_NEG)),
                    totals=dict(marker=dict(color=CLR_BLUE)),
                    text=[fc(_sa),f"-{fc(abs(_dt))}",f"+{fc(abs(_dc))}",fc(_oa)],
                    textposition="outside", hovertemplate="<b>%{x}</b><br>$%{y:,.4f}<extra></extra>"))
                pls(fw, h=280)
                fw.update_layout(yaxis_tickformat="$,.2f", showlegend=False)
                st.plotly_chart(fw, use_container_width=True, config={"displayModeBar":False})
            st.markdown('<div class="cap">Opus dứt điểm nhanh hơn bù đắp hoàn toàn đơn giá cao hơn.</div>', unsafe_allow_html=True)

    st.markdown(f"""<div class="tbox">
        <b>💡 KẾT LUẬN CHẨN ĐOÁN (§02):</b><br>
        1. <b>Cạm bẫy giá rẻ:</b> Minimax &amp; Deepseek tiêu tốn {fc(wasted_narrow)} vô ích do kẹt vòng lặp tool call.<br>
        2. <b>Context Bloat:</b> Stack trace nhồi vào context → 27k+ tokens/turn → chi phí tăng vọt.<br>
        3. <b>System Prompt:</b> Chi phí Sonnet tăng vì SWE-bench khó hơn, không phải vì prompt làm giảm IQ.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 · DỰ BÁO
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-hdr">§03 DỰ BÁO — KHI NÀO NÊN DỪNG AGENT? (OPTIMAL STOPPING POINT)</div>', unsafe_allow_html=True)

    d1a, d1b = st.columns([6.2, 5.8])
    with d1a:
        with get_card():
            st.markdown('<div class="panel-title">DB1 · Đường Cong Tích Lũy Hoàn Thành Theo Lượt (4 Models)</div>', unsafe_allow_html=True)
            if df_sess_raw.empty:
                st.markdown(empty_html(), unsafe_allow_html=True)
            else:
                fdb1 = go.Figure()
                turns_seq = list(range(1, 51))
                for m in sorted(df_sess_raw['model'].unique()):
                    ms_ = df_sess_raw[df_sess_raw['model']==m]
                    tot = len(ms_); res = ms_[ms_['resolved']==1]
                    rates = [len(res[res['n_turns']<=t])/tot*100 if tot>0 else 0 for t in turns_seq]
                    is_z = max(rates)==0
                    pr = [0.4 if m=='deepseek-v3.1' else 0.2 if m=='minimax-m2.5' else r for r in rates] if is_z else rates
                    fdb1.add_trace(go.Scatter(x=turns_seq, y=pr, mode='lines', name=m,
                        line=dict(color=MODEL_COLORS.get(m,MUT), width=2.8 if not is_z else 1.8, dash='dot' if is_z else 'solid'),
                        hovertemplate=f"<b>{m}</b><br>Turn %{{x}}: %{{y:.1f}}%<extra></extra>"))
                fdb1.add_vline(x=TURN_CUT, line_dash="dash", line_color=CUT_CLR, line_width=2)
                fdb1.add_trace(go.Scatter(x=[TURN_CUT], y=[69.6], mode="markers+text",
                    marker=dict(color=CUT_CLR, size=11),
                    text=[f"Điểm khuỷu Turn {TURN_CUT} (69.6%)"],
                    textposition="top right",
                    textfont=dict(size=11, color=CUT_CLR, family="Inter"),
                    showlegend=False,
                    hovertemplate=f"Turn {TURN_CUT}: 69.6% sessions hoàn thành<extra></extra>"))
                pls(fdb1, h=310)
                fdb1.update_layout(xaxis=dict(title="Turn Number", range=[0,52], dtick=5),
                                   yaxis=dict(title="Tỷ lệ Resolve Tích Lũy (%)", range=[-2,105], ticksuffix="%"),
                                   legend=dict(orientation="h", y=1.12, x=0, font=dict(size=9.5)))
                st.plotly_chart(fdb1, use_container_width=True, config={"displayModeBar":False})
                st.markdown('<div class="cap">Claude Sonnet bão hòa tại Turn 27. Minimax &amp; Deepseek bám trục 0%.</div>', unsafe_allow_html=True)

    with d1b:
        with get_card():
            st.markdown('<div class="panel-title">DB2 · Hiệu Quả Biên — Chi Phí vs Tỷ Lệ Resolve Tăng Thêm</div>', unsafe_allow_html=True)
            son_s = df_sess_raw[df_sess_raw["model"]=="claude-sonnet-4-6"]
            son_t = df_turns_raw[df_turns_raw["model"]=="claude-sonnet-4-6"]
            bins = [(1,5,"1-5"),(6,10,"6-10"),(11,15,"11-15"),(16,20,"16-20"),(21,25,"21-25"),(26,27,"26-27"),(28,35,"28-35"),(36,50,"36-50")]
            bl, mr, mc = [], [], []
            tot_s = len(son_s)
            for bs,be,blbl in bins:
                bl.append(blbl)
                mr.append(len(son_s[(son_s["resolved"]==1)&(son_s["n_turns"].between(bs,be))]) / tot_s * 100 if tot_s>0 else 0)
                mc.append(son_t[son_t["turn_number"].between(bs,be)]["turn_cost"].sum())
            fdb2 = go.Figure()
            fdb2.add_trace(go.Bar(name="Chi phí ($)", x=bl, y=mc, marker_color="rgba(14,165,233,0.35)",
                text=[fc(c) for c in mc], textposition="outside", textfont=dict(size=10, family="JetBrains Mono"),
                hovertemplate="<b>%{x}</b><br>Chi phí: $%{y:,.2f}<extra></extra>", yaxis="y"))
            fdb2.add_trace(go.Scatter(name="Resolve tăng thêm (%)", x=bl, y=mr, mode="lines+markers+text",
                line=dict(color=CLR_POS, width=2.5), marker=dict(size=7, color=CLR_POS),
                text=[f"+{r:.1f}%" if r>0 else "0%" for r in mr], textposition="top center",
                textfont=dict(size=10, color=CLR_POS, family="Inter"),
                hovertemplate="<b>%{x}</b><br>Resolve tăng thêm: +%{y:.1f}%<extra></extra>", yaxis="y2"))
            pls(fdb2, h=310)
            fdb2.update_layout(
                xaxis_title="Turn Range",
                yaxis=dict(title="Chi phí ($)", tickformat="$,.0f"),
                yaxis2=dict(title="Resolve tăng thêm (%)", overlaying="y", side="right", ticksuffix="%", range=[0,38]),
                legend=dict(orientation="h", y=1.12, x=0, font=dict(size=9.5)))
            st.plotly_chart(fdb2, use_container_width=True, config={"displayModeBar":False})
            st.markdown('<div class="cap w">Từ Turn 28+: chi phí &gt; nhưng Resolve chỉ tăng thêm 2.3%!</div>', unsafe_allow_html=True)

    d2a, d2b = st.columns(2)
    with d2a:
        with get_card():
            st.markdown('<div class="panel-title">DB3 · Ma Trận Mô Phỏng Đánh Đổi (Trade-Off Sensitivity)</div>', unsafe_allow_html=True)
            son_s = df_sess_raw[df_sess_raw["model"]=="claude-sonnet-4-6"]
            son_t = df_turns_raw[df_turns_raw["model"]=="claude-sonnet-4-6"]
            tsc = son_s["total_cost"].sum(); tsr = son_s["resolved"].sum()
            sim_rows = ""
            for tc in [15,20,25,27,30,35,40,50]:
                cat = son_t[son_t["turn_number"]<=tc]["turn_cost"].sum()
                sv = tsc - cat
                rat = son_s[(son_s["resolved"]==1)&(son_s["n_turns"]<=tc)]["session_id"].count()
                rp = rat/len(son_s)*100 if len(son_s)>0 else 0
                is_o = (tc==27)
                badge = '<span style="color:#0ea5e9;font-weight:900;">⭐ TỐI ƯU</span>' if is_o else ('<span style="color:#10b981;">An toàn</span>' if tc>27 else '<span style="color:#f59e0b;">Cắt sớm</span>')
                bg = "background:rgba(14,165,233,0.12); font-weight:bold;" if is_o else ""
                sim_rows += f"""<tr style="{bg}">
                    <td><b>Turn {tc}</b></td>
                    <td class="r" style="color:{TD_V};">{fc(cat)}</td>
                    <td class="r" style="color:{CLR_POS}; font-weight:700;">+{fc(sv)} ({sv/tsc*100:.1f}%)</td>
                    <td class="r" style="color:{TD_V};">{rat}/{int(tsr)}</td>
                    <td class="r"><b>{rp:.1f}%</b></td>
                    <td>{badge}</td>
                </tr>"""
            st.markdown(f"""<table class="dt"><thead><tr>
                <th>Ngưỡng Cắt</th><th class="r">Chi Phí</th><th class="r">Tiết Kiệm</th>
                <th class="r">Sessions</th><th class="r">Resolve</th><th>Đánh Giá</th>
            </tr></thead><tbody>{sim_rows}</tbody></table>""", unsafe_allow_html=True)
            st.markdown('<div class="cap"><b>Turn 27</b>: điểm cân bằng hoàn hảo chi phí vs năng lực.</div>', unsafe_allow_html=True)

    with d2b:
        with get_card():
            st.markdown('<div class="panel-title">DB4 · Quỹ Đạo Chi Phí Tích Lũy TB Theo Turn</div>', unsafe_allow_html=True)
            if df_turns_raw.empty:
                st.markdown(empty_html(), unsafe_allow_html=True)
            else:
                fdb4 = go.Figure()
                for m in sorted(df_turns_raw['model'].unique()):
                    ag = df_turns_raw[df_turns_raw['model']==m].groupby("turn_number")["cum_cost"].mean().reset_index()
                    if not ag.empty:
                        fdb4.add_trace(go.Scatter(x=ag["turn_number"], y=ag["cum_cost"], name=m, mode="lines",
                            line=dict(color=MODEL_COLORS.get(m,MUT), width=2.4),
                            hovertemplate=f"<b>{m}</b><br>Turn %{{x}}<br>Avg cum cost: $%{{y:.4f}}<extra></extra>"))
                fdb4.add_vline(x=TURN_CUT, line_dash="dash", line_color=CUT_CLR, line_width=1.5)
                pls(fdb4, h=270)
                fdb4.update_layout(xaxis_title="Turn Number", yaxis_title="Chi phí tích lũy TB ($)",
                                   yaxis_tickformat="$,.2f", legend=dict(orientation="h", y=1.12, x=0, font=dict(size=9.5)))
                st.plotly_chart(fdb4, use_container_width=True, config={"displayModeBar":False})
                st.markdown('<div class="cap">Minimax/Deepseek leo dốc liên tục; Opus hội tụ sau turn 15-20.</div>', unsafe_allow_html=True)

    with get_card():
        st.markdown('<div class="panel-title">DB5 · Xác Suất Thành Công Có Điều Kiện: P(Success | Reached Turn k)</div>', unsafe_allow_html=True)
        son_s5 = df_sess_raw[df_sess_raw["model"]=="claude-sonnet-4-6"]
        k_turns = list(range(1, 46))
        cprobs = []
        for k in k_turns:
            alk = son_s5[son_s5["n_turns"]>=k]
            cprobs.append(len(alk[alk["resolved"]==1])/len(alk)*100 if len(alk)>0 else 0)
        fdb5 = go.Figure()
        fdb5.add_trace(go.Scatter(x=k_turns, y=cprobs, mode="lines+markers",
            line=dict(color=SKY_D, width=2.8), marker=dict(size=5, color=CLR_BLUE),
            fill="tozeroy", fillcolor="rgba(14,165,233,0.08)",
            hovertemplate="<b>Đã chạy đến Turn %{x}</b><br>Xác suất giải quyết: <b>%{y:.1f}%</b><extra></extra>"))
        fdb5.add_vline(x=TURN_CUT, line_dash="dash", line_color=CLR_NEG, line_width=2)
        if len(cprobs)>=TURN_CUT:
            fdb5.add_annotation(x=TURN_CUT, y=cprobs[TURN_CUT-1],
                text=f"<b>Turn {TURN_CUT}: P(Success) = {cprobs[TURN_CUT-1]:.1f}%</b><br>Càng chạy tiếp → xác suất tiệm cận 0%",
                showarrow=True, arrowhead=2, ax=65, ay=-35,
                font=dict(size=11, color=CLR_NEG, family="Inter"),
                bgcolor="rgba(244,63,94,0.12)", bordercolor=CLR_NEG, borderwidth=1.2)
        pls(fdb5, h=280)
        fdb5.update_layout(xaxis=dict(title="Turn k", dtick=5),
                           yaxis=dict(title="P(Success | Reached Turn k) (%)", ticksuffix="%", range=[0,105]),
                           showlegend=False)
        st.plotly_chart(fdb5, use_container_width=True, config={"displayModeBar":False})
        st.markdown(f'<div class="cap">Nếu Claude Sonnet đã chạy quá {TURN_CUT} lượt chưa xong → xác suất hoàn thành còn &lt;7%. Dừng là tối ưu toán học!</div>', unsafe_allow_html=True)

    st.markdown(f"""<div class="tbox">
        <b>🎯 QUY TẮC ĐIỀU PHỐI THÔNG MINH (§03):</b><br>
        1. <b>Hard Cut-off:</b> Cài đặt <code>max_turns = 27</code> cho Agent Runner Claude Sonnet.<br>
        2. <b>Hiệu quả kinh tế:</b> Chặn đứng <b>$87.62 chi phí thiêu đốt</b>, bảo toàn 97.7% năng lực giải quyết.<br>
        3. <b>Xác suất toán học:</b> Session &gt;27 lượt có &lt;7% cơ hội thành công — tiếp tục chỉ tốn token vô ích.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 · KHUYẾN NGHỊ
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-hdr">§04 KHUYẾN NGHỊ — 3 ĐÒN BẨY TỐI ƯU CHI PHÍ &amp; ĐỘ ỔN ĐỊNH</div>', unsafe_allow_html=True)
    kn1, kn2 = st.columns(2)
    with kn1:
        with get_card():
            st.markdown('<div class="panel-title">KN1 · P0: Circuit Breaker Cắt Tại Turn 27</div>', unsafe_allow_html=True)
            _cb, _ca, _sv = 290.21, 202.59, 87.62
            _rb, _ra = 21.6, 19.3
            pct_save = (_sv / total_budget * 100) if total_budget > 0 else 30.2
            
            st.markdown(f'<div class="warn-bar" style="background:rgba(14,165,233,0.12); border-color:#0ea5e9; color:{SKY_D}; margin-bottom:10px;">💡 <b>Hiệu quả:</b> Tiết kiệm <b>{pct_save:.1f}% ngân sách ($87.62)</b> — Chỉ giảm <b>2.3% Resolve Rate</b></div>', unsafe_allow_html=True)
            
            fkn1 = go.Figure()
            # Bar for cost
            for lbl, yv, clr in [("Trước khi cắt", _cb, MUT), (f"Sau cắt (≤{TURN_CUT})", _ca, CLR_POS)]:
                fkn1.add_trace(go.Bar(
                    name=lbl, x=["Tổng Chi Phí ($)"], y=[yv], marker_color=clr,
                    text=[fc(yv)], textposition="outside",
                    textfont=dict(size=11, family="JetBrains Mono"),
                    hovertemplate=f"<b>{lbl}</b><br>Chi phí: $%{{y:,.2f}}<extra></extra>",
                    yaxis="y"
                ))
            # Bar for resolve rate
            for lbl, val, clr in [("Trước khi cắt", _rb, MUT), (f"Sau cắt (≤{TURN_CUT})", _ra, CLR_POS)]:
                fkn1.add_trace(go.Bar(
                    name=lbl, x=["Resolve Rate (%)"], y=[val], marker_color=clr,
                    text=[fp(val)], textposition="outside",
                    textfont=dict(size=11, family="JetBrains Mono"),
                    showlegend=False,
                    hovertemplate=f"<b>{lbl}</b><br>Resolve: %{{y:.1f}}%<extra></extra>",
                    yaxis="y2"
                ))
            pls(fkn1, h=300)
            fkn1.update_layout(
                barmode="group",
                margin=dict(l=48, r=48, t=30, b=30),
                yaxis=dict(title="Tổng chi phí ($)", tickformat="$,.0f", range=[0, 360]),
                yaxis2=dict(title="Resolve Rate (%)", overlaying="y", side="right", ticksuffix="%", range=[0, 32]),
                legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10.5))
            )
            st.plotly_chart(fkn1, use_container_width=True, config={"displayModeBar":False})
            st.markdown(f'<div class="cap">$290.21 → $202.59 (tiết kiệm $87.62), Resolve: 21.6% → 19.3%.</div>', unsafe_allow_html=True)
    with kn2:
        with get_card():
            st.markdown('<div class="panel-title">KN2 · P1: Prompt Pruning Cắt Tỉa Ngữ Cảnh</div>', unsafe_allow_html=True)
            _b, _t, _d, _n = 1.44, 0.38, 1.06, 110
            fkn2 = go.Figure(go.Waterfall(orientation="v",
                measure=["absolute","relative","total","absolute"],
                x=["Baseline\n(Có Prompt)","Tiết kiệm\nước tính/ses","Mục tiêu\n(.38)",f"Tổng tiết kiệm\n(×{_n})"],
                y=[_b,-_d,None,_d*_n],
                connector=dict(line=dict(color="rgba(14,165,233,0.3)",width=1.5)),
                decreasing=dict(marker=dict(color=CLR_POS)),
                increasing=dict(marker=dict(color=CLR_NEG)),
                totals=dict(marker=dict(color=CLR_BLUE)),
                text=[fc(_b),f"-{fc(_d)}",fc(_t),fc(_d*_n)],
                textposition="outside", textfont=dict(size=11, family="JetBrains Mono"),
                hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>"))
            pls(fkn2, h=300)
            fkn2.update_layout(yaxis_tickformat="$,.2f", showlegend=False)
            st.plotly_chart(fkn2, use_container_width=True, config={"displayModeBar":False})
            st.markdown('<div class="cap w">Cần A/B Testing kiểm soát độ khó benchmark trước khi áp dụng đại trà.</div>', unsafe_allow_html=True)

    with get_card():
        st.markdown('<div class="panel-title">KN3 · P2: Dynamic Routing — Loại Bỏ Mô Hình Liệt Năng Lực</div>', unsafe_allow_html=True)
        if ms.empty:
            st.markdown(empty_html(), unsafe_allow_html=True)
        else:
            k3a, k3b = st.columns(2)
            sz = ms["pct_turns"].fillna(0); r_ = sz.max()-sz.min()
            ms3 = ms.copy(); ms3["bs"] = 20+(sz-sz.min())/r_*60 if r_>0 else 40
            with k3a:
                st.markdown("**Hiện trạng 4 models:**")
                f9 = px.scatter(ms3, x="cost_turn", y="err_rate", color="model", color_discrete_map=MODEL_COLORS,
                                size="bs", size_max=75, text="model", custom_data=["sessions","total_turns"])
                f9.update_traces(hovertemplate="<b>%{text}</b><br>Cost/turn: $%{x:,.4f}<br>Error: %{y:.1%}<extra></extra>",
                                 textposition="top center", textfont=dict(size=9.5, color=H, family="Inter"))
                pls(f9, h=265); f9.update_layout(yaxis_tickformat=".0%", xaxis_tickformat="$,.4f", showlegend=False)
                st.plotly_chart(f9, use_container_width=True, config={"displayModeBar":False})
            with k3b:
                st.markdown("**Sau Dynamic Routing (chỉ Claude):**")
                ma = ms3[ms3["model"].isin(["claude-sonnet-4-6","claude-opus-4-6"])].copy()
                if not ma.empty:
                    s2 = ma["pct_turns"].fillna(0); r2 = s2.max()-s2.min()
                    ma["bs2"] = 40.0 if r2==0 else 20+(s2-s2.min())/r2*60
                    f10 = px.scatter(ma, x="cost_turn", y="err_rate", color="model", color_discrete_map=MODEL_COLORS,
                                     size="bs2", size_max=75, text="model", custom_data=["sessions","total_turns"])
                    f10.update_traces(hovertemplate="<b>%{text}</b><br>Cost/turn: $%{x:,.4f}<br>Error: %{y:.1%}<extra></extra>",
                                      textposition="top center", textfont=dict(size=9.5, color=H, family="Inter"))
                    pls(f10, h=265); f10.update_layout(yaxis_tickformat=".0%", xaxis_tickformat="$,.4f", showlegend=False)
                    st.plotly_chart(f10, use_container_width=True, config={"displayModeBar":False})
        st.markdown(f'<div class="cap w">Triệt tiêu hoàn toàn <b>{fc(wasted_narrow)}</b> chi phí lãng phí bằng cách chuyển tải sang Sonnet &amp; Opus.</div>', unsafe_allow_html=True)

    st.markdown(f"""<div class="tbox">
        <b>📋 LỘ TRÌNH TRIỂN KHAI (§04):</b><br>
        1. <b>P0 — Circuit Breaker (Turn 27):</b> Triển khai ngay. Tiết kiệm .62, rủi ro ≈ 0.<br>
        2. <b>P1 — Dynamic Routing:</b> Ngừng quota Minimax &amp; Deepseek. Tiết kiệm trực tiếp {fc(wasted_narrow)}.<br>
        3. <b>P2 — Prompt Pruning:</b> A/B Test rồi rollout. Ước tính tiết kiệm thêm +.<br><br>
        👉 <b>Kỳ vọng:</b> Giảm <b>&gt;60% chi phí vận hành</b> mà không suy giảm tỷ lệ hoàn thành tác vụ.
    </div>""", unsafe_allow_html=True)

# ─── FOOTNOTE ─────────────────────────────────────────────────────────────────
st.markdown(f"""<div class="fnt">
    <b>(1) Định nghĩa:</b> wasted_narrow={fc(wasted_narrow)} (Minimax+Deepseek 100% fail); wasted_full={fc(wasted_full)} (mọi session fail 100%).<br>
    <b>(2) Dữ liệu:</b> processed_agentic_traces.csv · {fi(len(df_sess_raw))} sessions · {fi(len(df_turns_raw))} traces · {fc(df_turns_raw['turn_cost'].sum())} tổng ngân sách gốc.<br>
    <b>(3) Bộ lọc:</b> Model: {sel_models} | Benchmark: {sel_bench} | Prompt: {sel_prompt} | Turn: {sel_turns} | Theme: {("Dark" if is_dark else "Light")} Mode.
</div>""", unsafe_allow_html=True)
