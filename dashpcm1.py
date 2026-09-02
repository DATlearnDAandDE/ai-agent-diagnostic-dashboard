# -*- coding: utf-8 -*-
"""
EXECUTIVE REPORT: BAN LINH DIEU PHOI AI AGENT — TOI UU CHI PHI & DO ON DINH
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
    page_title="Báo cáo phân tích chí phí & hiệu năng của AI Agent",
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

# ─── SWE-BENCH FIXED SCOPE (Tab §02-§04) ───────────────────────────────────────
# Cố định Apples-to-Apples: chỉ SWE-bench, loại claude-opus-4-6 (Opus chỉ chạy ở WildClaw).
# df_swe_s / df_swe_t KHÔNG chịu ảnh hưởng bởi bộ lọc Model/Benchmark phía trên (chỉ Tab 01 dùng bộ lọc đó).
df_swe_s = df_sess_raw[
    (df_sess_raw["benchmark"] == "swebench") & (df_sess_raw["model"] != "claude-opus-4-6")
].copy()
df_swe_t = df_turns_raw[df_turns_raw["session_id"].isin(df_swe_s["session_id"])].copy()
SWE_MODELS = sorted(df_swe_s["model"].unique().tolist())

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
    st.markdown(f"""
    <div class="rpt-title">BÁO CÁO PHÂN TÍCH CHI PHÍ VÀ HIỆU NĂNG CỦA AI AGENT</div>
    <div class="rpt-sub">Executive Diagnostic Report · {fi(len(df_turns_raw))} Traces · {fi(len(df_sess_raw))} Sessions · {df_sess_raw['model'].nunique()} Mô hình · Phân tích 4 Cấp độ Độc Lập</div>
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
            "🤖 Mô hình (chỉ Tab 01)",
            options=list(MODEL_COLORS.keys()),
            default=list(MODEL_COLORS.keys()),
            key="flt_model",
            help="Chỉ áp dụng cho Tab 01 · Mô Tả. Tab 02-04 cố định trên SWE-bench (Sonnet · Minimax · DeepSeek) để so sánh công bằng.",
        )
    with fc2:
        sel_bench = st.multiselect(
            "🎯 Benchmark (chỉ Tab 01)",
            options=["swebench","gaia","wildclaw"],
            default=["swebench","gaia","wildclaw"],
            key="flt_bench",
            help="Chỉ áp dụng cho Tab 01 · Mô Tả. Tab 02-04 cố định trên SWE-bench.",
        )
    with fc3:
        sel_prompt = st.selectbox(
            "💬 System Prompt (chỉ Tab 01)",
            ["Tất cả","Có Prompt","Không Prompt"],
            key="flt_prompt",
        )
    with fc4:
        sel_turns = st.slider(
            "🔄 Dải Turn (mọi Tab)",
            min_value=1, max_value=50, value=(1, 50),
            key="flt_turns",
            help="Áp dụng cho tất cả các Tab, kể cả Tab 02-04 (dữ liệu SWE-bench cố định).",
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

# ─── SWE-BENCH SCOPE: dải Turn vẫn áp dụng, Model/Benchmark thì không ─────────
df_swe_t_f      = df_swe_t[df_swe_t["turn_number"].between(sel_turns[0], sel_turns[1])].copy()
ms_swe          = mdl_agg(df_swe_t, df_swe_s)
wasted_narrow_swe = df_swe_s[df_swe_s["model"].isin(["minimax-m2.5","deepseek-v3.1"])]["total_cost"].sum()
wasted_full_swe   = df_swe_s[df_swe_s["failed"] == 1]["total_cost"].sum()
swe_budget      = df_swe_s["total_cost"].sum()

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
    "01 MÔ TẢ",
    "02 CHẨN ĐOÁN",
    "03 DỰ BÁO",
    "04 KHUYẾN NGHỊ",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · MÔ TẢ
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec-hdr">§01. TỔNG QUAN — Hiện trạng Chi phí &amp; Năng lực</div>', unsafe_allow_html=True)
    r1c1, r1c2 = st.columns([5.2, 6.8])
    with r1c1:
        with get_card():
            st.markdown('<div class="panel-title">Biểu đồ 1.1: So sánh tổng quan 4 Trợ lý AI</div>', unsafe_allow_html=True)
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
            st.markdown('<div class="panel-title">Biểu đồ 1.2: Tiền chi tiêu cho từng Trợ lý AI</div>', unsafe_allow_html=True)
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
                st.plotly_chart(f2, key="chart_mt2", use_container_width=True, config={"displayModeBar":False})

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        with get_card():
            st.markdown('<div class="panel-title">Biểu đồ 1.3: Số bước xử lý trung bình cho một công việc</div>', unsafe_allow_html=True)
            if df_s.empty:
                st.markdown(empty_html(), unsafe_allow_html=True)
            else:
                f3 = go.Figure()
                for m in sorted(df_s["model"].unique()):
                    f3.add_trace(go.Box(y=df_s[df_s["model"]==m]["n_turns"], name=m,
                                        marker_color=MODEL_COLORS.get(m,MUT), boxmean=True,
                                        hovertemplate=f"<b>{m}</b><br>Turns: %{{y}}<extra></extra>"))
                pls(f3, h=290)
                f3.update_layout(showlegend=False, yaxis_title="n_turns / session")
                st.plotly_chart(f3, key="chart_mt3", use_container_width=True, config={"displayModeBar":False})
                st.markdown('<div class="cap">Minimax &amp; Deepseek thường kéo dài vượt ngưỡng 25-30 turns.</div>', unsafe_allow_html=True)
    with r2c2:
        with get_card():
            st.markdown('<div class="panel-title">Biểu đồ 1.4: Tỷ lệ làm hỏng việc của từng Trợ lý AI</div>', unsafe_allow_html=True)
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
                st.plotly_chart(f4, key="chart_mt4", use_container_width=True, config={"displayModeBar":False})
                st.markdown('<div class="cap w">Minimax và Deepseek có tỷ lệ lỗi 100% — đối lập hoàn toàn với Claude.</div>', unsafe_allow_html=True)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        with get_card():
            st.markdown('<div class="panel-title">Biểu đồ 1.5: Phân bổ độ khó của các công việc</div>', unsafe_allow_html=True)
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
                st.plotly_chart(f5, key="chart_mt5", use_container_width=True, config={"displayModeBar":False})
    with r3c2:
        with get_card():
            st.markdown('<div class="panel-title">Biểu đồ 1.6: Mức độ đắt/rẻ trung bình cho một công việc</div>', unsafe_allow_html=True)
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
                st.plotly_chart(f6, key="chart_mt6", use_container_width=True, config={"displayModeBar":False})

    st.markdown(f"""<div class="tbox">
        <b>💡 TỔNG QUAN MÔ TẢ (§01):</b><br>
        • <b>Quy mô:</b> Claude Sonnet chiếm {fp(ms[ms['model']=='claude-sonnet-4-6']['pct_cost'].sum()*100 if not ms.empty else 0)} ngân sách nhưng hoàn thành đại đa số tác vụ. Minimax &amp; Deepseek chiếm {fp(ms[ms['model'].isin(['minimax-m2.5','deepseek-v3.1'])]['pct_cost'].sum()*100 if not ms.empty else 0)} chi phí nhưng 0% kết quả thực sự.<br>
        • <b>Bất thường:</b> Tỷ lệ lỗi 100% và số lượt lặp >35 turns của 2 mô hình rẻ báo hiệu hiện tượng kẹt vòng lặp — được chẩn đoán sâu ở Tab §02.
    </div>""", unsafe_allow_html=True)

    _pct_sess_swe  = (len(df_swe_s) / len(df_sess_raw) * 100) if len(df_sess_raw) else 0.0
    _pct_turns_swe = (len(df_swe_t) / len(df_turns_raw) * 100) if len(df_turns_raw) else 0.0
    _pct_cost_swe  = (swe_budget / df_sess_raw['total_cost'].sum() * 100) if df_sess_raw['total_cost'].sum() > 0 else 0.0
    st.markdown(f"""<div class="warn-bar" style="background:rgba(14,165,233,0.1); border-color:#0ea5e9; color:{SKY_D}; margin-top:10px;">
        📌 <b>SWE-bench chiếm {_pct_sess_swe:.1f}% sessions, {_pct_turns_swe:.1f}% turns và {fc(swe_budget)} ({_pct_cost_swe:.1f}%) ngân sách</b> toàn hệ thống.
        Để đảm bảo tính chuẩn xác và so sánh sòng phẳng (Apples-to-Apples), toàn bộ §02, §03, §04 bên dưới đào sâu phân tích trên {fi(len(df_swe_s))} bài toán SWE-bench
        (Sonnet · Minimax · DeepSeek — loại Opus vì Opus chỉ chạy ở WildClaw), không phụ thuộc bộ lọc Model/Benchmark phía trên.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · CHẨN ĐOÁN
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-hdr">§02. CHẨN ĐOÁN — Tại sao lãng phí tiền bạc?</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="cap" style="margin:-6px 0 10px;">📌 Toàn bộ §02–§04 cố định trên <b>{fi(len(df_swe_s))} sessions SWE-bench</b> ({" · ".join(SWE_MODELS)}) — không phụ thuộc bộ lọc Model/Benchmark, chỉ dải Turn vẫn được áp dụng.</div>', unsafe_allow_html=True)
    with get_card():
        st.markdown('<div class="panel-title">Biểu đồ 2.1: Bẫy "Giá Rẻ" - Đơn giá rẻ nhưng làm tốn tiền vô ích</div>', unsafe_allow_html=True)
        if df_swe_s.empty:
            st.markdown(empty_html(), unsafe_allow_html=True)
        else:
            fig_cd1 = go.Figure()
            for m in SWE_MODELS:
                m_df = df_swe_s[df_swe_s["model"]==m]
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
            trap = df_swe_s[df_swe_s['model'].isin(['minimax-m2.5','deepseek-v3.1'])]
            if not trap.empty:
                fig_cd1.add_annotation(x=trap['cost_per_turn'].median(), y=trap['n_turns'].max()*0.9,
                    text=f"<b>🚨 KẸT VÒNG LẶP</b><br>{fc(wasted_narrow_swe)} lãng phí (100% Fail)",
                    showarrow=True, arrowhead=2, ax=50, ay=-20,
                    font=dict(color="#be123c", size=11, family="Inter"),
                    bgcolor="rgba(255,241,242,0.95)", bordercolor="#f43f5e", borderwidth=1.2)
            pls(fig_cd1, h=350)
            fig_cd1.update_layout(xaxis_title="Chi phí TB / turn ($)", yaxis_title="n_turns / session",
                                  xaxis_tickformat="$.3f", legend=dict(orientation="h", y=1.06, x=0))
            st.plotly_chart(fig_cd1, key="chart_cd1", use_container_width=True, config={"displayModeBar":False})

    cf1, cf2 = st.columns([5.5, 6.5])
    with cf1:
        with get_card():
            st.markdown('<div class="panel-title">Biểu đồ 2.2: Lỗi mắc kẹt - Trợ lý AI bị kẹt trong vòng lặp vô tận</div>', unsafe_allow_html=True)
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
            st.markdown('<div class="panel-title">Biểu đồ 2.3: Lượng dữ liệu AI phải đọc tăng vọt qua từng bước</div>', unsafe_allow_html=True)
            if df_swe_t_f.empty:
                st.markdown(empty_html(), unsafe_allow_html=True)
            else:
                f_cd3 = go.Figure()
                for m in SWE_MODELS:
                    ag = df_swe_t_f[df_swe_t_f["model"]==m].groupby("turn_number")["input_tokens"].mean().reset_index()
                    if not ag.empty:
                        f_cd3.add_trace(go.Scatter(x=ag["turn_number"], y=ag["input_tokens"], name=m, mode="lines+markers",
                            line=dict(color=MODEL_COLORS.get(m,MUT), width=2.2), marker=dict(size=4),
                            hovertemplate=f"<b>{m}</b><br>Turn %{{x}}<br>%{{y:,.0f}} tokens avg<extra></extra>"))
                pls(f_cd3, h=248)
                f_cd3.update_layout(xaxis_title="Turn Number", yaxis_title="Avg Input Tokens",
                                    yaxis_tickformat=",", legend=dict(orientation="h", y=1.1, x=0, font=dict(size=9.5)))
                st.plotly_chart(f_cd3, key="chart_cd3", use_container_width=True, config={"displayModeBar":False})

    with get_card():
        st.markdown('<div class="panel-title">Biểu đồ 2.4: Hành vi Tự phản tư khi gặp lỗi (Self-Reflection via Output Length)</div>', unsafe_allow_html=True)
        if df_swe_t.empty:
            st.markdown(empty_html(), unsafe_allow_html=True)
        else:
            refl = df_swe_t.groupby(["model","has_error"])["output_length"].mean().reset_index()
            fig_refl = go.Figure()
            for ev, lbl, clr in [(0, "Không lỗi (has_error=0)", CLR_POS), (1, "Có lỗi (has_error=1)", CLR_NEG)]:
                sub = refl[refl["has_error"]==ev].copy()
                sub["model"] = pd.Categorical(sub["model"], categories=SWE_MODELS, ordered=True)
                sub = sub.sort_values("model")
                fig_refl.add_trace(go.Bar(
                    x=sub["model"], y=sub["output_length"], name=lbl, marker_color=clr,
                    text=[f"{v:.0f}" for v in sub["output_length"]], textposition="outside",
                    textfont=dict(size=11, family="JetBrains Mono"),
                    hovertemplate=f"<b>%{{x}}</b><br>{lbl}: %{{y:.0f}} tokens<extra></extra>"
                ))

            cap_bits = []
            son_r = refl[refl["model"]=="claude-sonnet-4-6"]
            if set(son_r["has_error"]) == {0,1}:
                a = son_r[son_r["has_error"]==0]["output_length"].values[0]
                b = son_r[son_r["has_error"]==1]["output_length"].values[0]
                delta_pct = (b-a)/a*100 if a>0 else 0
                fig_refl.add_annotation(x="claude-sonnet-4-6", y=b, text=f"<b>+{delta_pct:.1f}%</b> khi gặp lỗi",
                    showarrow=True, arrowhead=2, ax=0, ay=-38,
                    font=dict(size=11, color=CLR_POS, family="Inter"),
                    bgcolor="rgba(16,185,129,0.12)", bordercolor=CLR_POS, borderwidth=1)
                cap_bits.append(f"Khi gặp lỗi, Sonnet tăng output từ {a:.0f} lên {b:.0f} tokens (+{delta_pct:.1f}%) — dấu hiệu tự phản tư: dừng lại phân tích stack trace, thử cách khác.")
            for m, mlbl in [("minimax-m2.5","Minimax"), ("deepseek-v3.1","DeepSeek")]:
                sub_m = refl[refl["model"]==m]
                states = set(sub_m["has_error"].tolist())
                if states == {1}:
                    v = sub_m[sub_m["has_error"]==1]["output_length"].values[0]
                    cap_bits.append(f"{mlbl} gần như không có turn nào thoát lỗi trong SWE-bench — giữ output ngắn (~{v:.0f} tokens) và lặp lại lệnh hỏng thay vì tự sửa.")

            pls(fig_refl, h=320)
            fig_refl.update_layout(barmode="group", xaxis_title="", yaxis_title="Output Length trung bình (tokens)",
                                    legend=dict(orientation="h", y=1.1, x=0, font=dict(size=10)))
            st.plotly_chart(fig_refl, key="chart_cd4_reflect", use_container_width=True, config={"displayModeBar":False})
            st.markdown(f'<div class="cap w">{" ".join(cap_bits)}</div>', unsafe_allow_html=True)

    cr1, cr2 = st.columns(2)
    with cr1:
        with get_card():
            st.markdown('<div class="panel-title">Biểu đồ 2.6: Tỷ lệ Tiền sinh lời vs Tiền bị lãng phí (SWE-bench)</div>', unsafe_allow_html=True)
            if df_swe_s.empty:
                st.markdown(empty_html(), unsafe_allow_html=True)
            else:
                ml_ = SWE_MODELS
                vc  = [df_swe_s[(df_swe_s["model"]==m)&(df_swe_s["failed"]==0)]["total_cost"].sum() for m in ml_]
                wc  = [df_swe_s[(df_swe_s["model"]==m)&(df_swe_s["failed"]==1)]["total_cost"].sum() for m in ml_]
                f6_ = go.Figure()
                f6_.add_trace(go.Bar(name="Hiệu quả (Resolved)", x=ml_, y=vc, marker_color=CLR_POS, opacity=0.9, text=[fc(v) if v>0 else "" for v in vc], textposition="inside", hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>"))
                f6_.add_trace(go.Bar(name="Lãng phí (Failed)", x=ml_, y=wc, marker_color=CLR_NEG, opacity=0.9, text=[fc(v) if v>0 else "" for v in wc], textposition="inside", hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>"))
                pls(f6_, h=280)
                f6_.update_layout(barmode="stack", yaxis_tickformat="$,.2f", xaxis_title="", legend=dict(orientation="h",y=1.1,x=0,font=dict(size=10)))
                st.plotly_chart(f6_, key="chart_cd6", use_container_width=True, config={"displayModeBar":False})
                _swe_waste_pct = (wasted_full_swe/swe_budget*100) if swe_budget>0 else 0
                st.markdown(f'<div class="cap w">Tổng chi phí lãng phí trên SWE-bench: <b>{fc(wasted_full_swe)}</b> ({_swe_waste_pct:.1f}% ngân sách SWE-bench).</div>', unsafe_allow_html=True)

    with cr2:
        with get_card():
            st.markdown('<div class="panel-title">Biểu đồ 2.7: Phân rã Chi phí Thiêu đốt trên SWE-bench</div>', unsafe_allow_html=True)
            _minimax_cost = df_swe_s[df_swe_s["model"]=="minimax-m2.5"]["total_cost"].sum()
            _deepseek_cost = df_swe_s[df_swe_s["model"]=="deepseek-v3.1"]["total_cost"].sum()
            _sonnet_swe = df_swe_s[df_swe_s["model"]=="claude-sonnet-4-6"]
            _sonnet_failed_cost = _sonnet_swe[_sonnet_swe["failed"]==1]["total_cost"].sum()
            _net_productive = swe_budget - _minimax_cost - _deepseek_cost - _sonnet_failed_cost

            fw = go.Figure(go.Waterfall(
                orientation="v",
                measure=["absolute", "relative", "relative", "relative", "total"],
                x=["Ngân Sách SWE-bench", "Lãng phí Minimax", "Lãng phí DeepSeek", "Lãng phí Sonnet Failed", "Chi phí Thực sự Sinh lời"],
                y=[swe_budget, -_minimax_cost, -_deepseek_cost, -_sonnet_failed_cost, None],
                connector=dict(line=dict(color="rgba(14,165,233,0.3)", width=1.5)),
                decreasing=dict(marker=dict(color=CLR_NEG)),
                increasing=dict(marker=dict(color=CLR_POS)),
                totals=dict(marker=dict(color=CLR_BLUE)),
                text=[fc(swe_budget), f"-{fc(_minimax_cost)}", f"-{fc(_deepseek_cost)}", f"-{fc(_sonnet_failed_cost)}", fc(_net_productive)],
                textposition="outside", textfont=dict(size=10.5, family="JetBrains Mono"),
                hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>"
            ))
            pls(fw, h=280)
            fw.update_layout(yaxis_tickformat="$,.2f", showlegend=False)
            st.plotly_chart(fw, key="chart_cd7", use_container_width=True, config={"displayModeBar":False})
            st.markdown(f'<div class="cap w">Chỉ <b>{fc(_net_productive)}</b> ({(_net_productive/swe_budget*100) if swe_budget>0 else 0:.1f}%) trong {fc(swe_budget)} thực sự mang lại kết quả — phần còn lại bị thiêu đốt vào vòng lặp lỗi.</div>', unsafe_allow_html=True)

    st.markdown(f"""<div class="tbox">
        <b>💡 KẾT LUẬN CHẨN ĐOÁN (§02 · SWE-bench, {fi(len(df_swe_s))} sessions):</b><br>
        1. <b>Cạm bẫy giá rẻ:</b> Minimax &amp; Deepseek tiêu tốn {fc(wasted_narrow_swe)} vô ích do kẹt vòng lặp tool call.<br>
        2. <b>Context Bloat:</b> Stack trace nhồi vào context mỗi turn → tokens/turn tăng vọt qua thời gian (Biểu đồ 2.3).<br>
        3. <b>Tự phản tư (Self-Reflection):</b> Sonnet tăng output khi gặp lỗi để phân tích & thử cách mới; Minimax/DeepSeek gần như không thoát khỏi trạng thái lỗi, lặp lại lệnh hỏng.<br>
        4. <b>Tổng thiệt hại:</b> {fc(wasted_full_swe)} ({(wasted_full_swe/swe_budget*100) if swe_budget>0 else 0:.1f}% ngân sách SWE-bench) bị thiêu đốt, chỉ {fc(swe_budget-wasted_full_swe)} thực sự mang lại kết quả.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 · DỰ BÁO
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-hdr">§03. DỰ BÁO — Điểm dừng nào tiết kiệm nhất?</div>', unsafe_allow_html=True)

    d1a, d1b = st.columns([6.2, 5.8])
    with d1a:
        with get_card():
            st.markdown('<div class="panel-title">Biểu đồ 3.1: Tỷ lệ hoàn thành công việc sau mỗi bước xử lý (SWE-bench)</div>', unsafe_allow_html=True)
            if df_swe_s.empty:
                st.markdown(empty_html(), unsafe_allow_html=True)
            else:
                fdb1 = go.Figure()
                turns_seq = list(range(1, 51))
                son_rate_at_cut = 0.0
                
                for m in SWE_MODELS:
                    ms_ = df_swe_s[df_swe_s['model']==m]
                    tot = len(ms_); res = ms_[ms_['resolved']==1]
                    rates = [len(res[res['n_turns']<=t])/tot*100 if tot>0 else 0 for t in turns_seq]
                    if m == 'claude-sonnet-4-6' and len(rates) >= TURN_CUT:
                        son_rate_at_cut = rates[TURN_CUT-1]
                    is_z = max(rates)==0
                    pr = [0.4 if m=='deepseek-v3.1' else 0.2 if m=='minimax-m2.5' else r for r in rates] if is_z else rates
                    fdb1.add_trace(go.Scatter(x=turns_seq, y=pr, mode='lines', name=m,
                        line=dict(color=MODEL_COLORS.get(m,MUT), width=2.8 if not is_z else 1.8, dash='dot' if is_z else 'solid'),
                        hovertemplate=f"<b>{m}</b><br>Turn %{{x}}: %{{y:.1f}}%<extra></extra>"))
                
                fdb1.add_vline(x=TURN_CUT, line_dash="dash", line_color=CUT_CLR, line_width=2)
                fdb1.add_trace(go.Scatter(x=[TURN_CUT], y=[son_rate_at_cut], mode="markers+text",
                    marker=dict(color=CUT_CLR, size=11),
                    text=[f"Điểm khuỷu Turn {TURN_CUT} ({son_rate_at_cut:.1f}%)"],
                    textposition="top right",
                    textfont=dict(size=11, color=CUT_CLR, family="Inter"),
                    showlegend=False,
                    hovertemplate=f"Turn {TURN_CUT}: {son_rate_at_cut:.1f}% sessions hoàn thành<extra></extra>"))
                pls(fdb1, h=310)
                fdb1.update_layout(xaxis=dict(title="Turn Number", range=[0,52], dtick=5),
                                   yaxis=dict(title="Tỷ lệ Resolve Tích Lũy (%)", range=[-2,105], ticksuffix="%"),
                                   legend=dict(orientation="h", y=1.12, x=0, font=dict(size=9.5)))
                st.plotly_chart(fdb1, key="chart_db1", use_container_width=True, config={"displayModeBar":False})
                st.markdown(f'<div class="cap">Claude Sonnet bão hòa tại Turn {TURN_CUT} ({son_rate_at_cut:.1f}%). Minimax &amp; Deepseek bám trục 0%.</div>', unsafe_allow_html=True)

    with d1b:
        with get_card():
            st.markdown('<div class="panel-title">Biểu đồ 3.2: Chi thêm tiền liệu có mang lại kết quả tốt hơn? (Sonnet · SWE-bench)</div>', unsafe_allow_html=True)
            son_s = df_swe_s[df_swe_s["model"]=="claude-sonnet-4-6"]
            son_t = df_swe_t[df_swe_t["model"]=="claude-sonnet-4-6"]
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
            st.plotly_chart(fdb2, key="chart_db2", use_container_width=True, config={"displayModeBar":False})
            st.markdown('<div class="cap w">Từ Turn 28+: chi phí tiếp tục tiêu hao nhưng Resolve chỉ tăng thêm rất ít!</div>', unsafe_allow_html=True)

    with get_card():
        st.markdown('<div class="panel-title">Biểu đồ 3.4: Đánh đổi giữa Ngân sách và Kết quả thực tế (Sonnet · SWE-bench)</div>', unsafe_allow_html=True)
        son_s4 = df_swe_s[df_swe_s["model"]=="claude-sonnet-4-6"]
        son_t4 = df_swe_t[df_swe_t["model"]=="claude-sonnet-4-6"]
        if son_s4.empty or son_t4.empty:
            st.markdown(empty_html(), unsafe_allow_html=True)
        else:
            k_range = list(range(1, 51))
            sim_costs = []
            sim_rates = []
            tot_s4 = len(son_s4)
            
            for k in k_range:
                c = son_t4[son_t4["turn_number"] <= k]["turn_cost"].sum()
                sim_costs.append(c)
                r = len(son_s4[(son_s4["resolved"] == 1) & (son_s4["n_turns"] <= k)])
                sim_rates.append(r / tot_s4 * 100 if tot_s4 > 0 else 0)
            
            fdb4 = go.Figure()
            fdb4.add_trace(go.Bar(x=k_range, y=sim_costs, name="Tổng chi phí ($)", 
                                  marker_color="rgba(14,165,233,0.3)",
                                  hovertemplate="<b>Max Turns: %{x}</b><br>Tổng chi phí: $%{y:,.2f}<extra></extra>",
                                  yaxis="y"))
            fdb4.add_trace(go.Scatter(x=k_range, y=sim_rates, name="Tỷ lệ hoàn thành (%)", mode="lines",
                                      line=dict(color=CLR_POS, width=3),
                                      hovertemplate="<b>Max Turns: %{x}</b><br>Tỷ lệ hoàn thành: %{y:.1f}%<extra></extra>",
                                      yaxis="y2"))
            fdb4.add_vline(x=TURN_CUT, line_dash="dash", line_color=CUT_CLR, line_width=2)
            pls(fdb4, h=270)
            fdb4.update_layout(xaxis=dict(title="Ngưỡng cắt (Max Turns)", dtick=5),
                               yaxis=dict(title="Tổng chi phí ($)", tickformat="$,.0f"),
                               yaxis2=dict(title="Tỷ lệ hoàn thành (%)", overlaying="y", side="right", range=[0, 105], ticksuffix="%"),
                               legend=dict(orientation="h", y=1.12, x=0, font=dict(size=9.5)))
            st.plotly_chart(fdb4, key="chart_db4_new", use_container_width=True, config={"displayModeBar":False})
            st.markdown(f'<div class="cap">Cắt tại Turn {TURN_CUT} tối ưu hóa chi phí mà hầu như không hy sinh tỷ lệ hoàn thành.</div>', unsafe_allow_html=True)

    c35, c36 = st.columns(2)
    with c35:
        with get_card():
            st.markdown('<div class="panel-title">Biểu đồ 3.5: Xác định số bước tối đa để đạt lợi nhuận cao nhất (Sonnet · SWE-bench)</div>', unsafe_allow_html=True)
            son_s5 = df_swe_s[df_swe_s["model"]=="claude-sonnet-4-6"]
            son_t5 = df_swe_t[df_swe_t["model"]=="claude-sonnet-4-6"]
            if son_s5.empty or son_t5.empty:
                st.markdown(empty_html(), unsafe_allow_html=True)
            else:
                k_range = list(range(1, 51))
                cost_per_issue = []
                valid_k = []
                for k in k_range:
                    c = son_t5[son_t5["turn_number"] <= k]["turn_cost"].sum()
                    r_count = len(son_s5[(son_s5["resolved"] == 1) & (son_s5["n_turns"] <= k)])
                    if r_count > 0:
                        valid_k.append(k)
                        cost_per_issue.append(c / r_count)
                
                fdb5 = go.Figure()
                fdb5.add_trace(go.Scatter(x=valid_k, y=cost_per_issue, mode="lines+markers",
                    line=dict(color=SKY_D, width=3), marker=dict(size=6, color=CLR_BLUE),
                    fill="tozeroy", fillcolor="rgba(14,165,233,0.08)",
                    hovertemplate="<b>Max Turns: %{x}</b><br>Chi phí/Issue: $%{y:,.2f}<extra></extra>"))
                
                if valid_k:
                    min_cost = min(cost_per_issue)
                    min_idx = cost_per_issue.index(min_cost)
                    opt_k = valid_k[min_idx]
                    
                    fdb5.add_vline(x=opt_k, line_dash="dash", line_color=CLR_POS, line_width=2)
                    fdb5.add_annotation(x=opt_k, y=min_cost,
                        text=f"<b>Tối ưu nhất: Turn {opt_k}</b><br>${min_cost:,.2f}/issue",
                        showarrow=True, arrowhead=2, ax=50, ay=-40,
                        font=dict(size=11, color=CLR_POS, family="Inter"),
                        bgcolor="rgba(16,185,129,0.12)", bordercolor=CLR_POS, borderwidth=1.2)
                    
                    fdb5.add_vline(x=TURN_CUT, line_dash="dot", line_color=CUT_CLR, line_width=1.5)

                pls(fdb5, h=280)
                fdb5.update_layout(xaxis=dict(title="Ngưỡng cắt (Max Turns)", dtick=5),
                                   yaxis=dict(title="Chi phí / Issue thành công ($)", tickformat="$,.2f"),
                                   showlegend=False)
                st.plotly_chart(fdb5, key="chart_db5_new", use_container_width=True, config={"displayModeBar":False})
                
                opt_text = f"Điểm đáy tại Turn {opt_k}. Nếu chạy quá điểm này, chi phí biên tăng nhanh hơn giá trị mang lại!" if valid_k else ""
                st.markdown(f'<div class="cap">{opt_text}</div>', unsafe_allow_html=True)

    with c36:
        with get_card():
            st.markdown('<div class="panel-title">Biểu đồ 3.6: Dự Báo Sớm Nguy Cơ Kẹt Vòng Lặp (5 turn đầu · SWE-bench)</div>', unsafe_allow_html=True)
            try:
                from sklearn.ensemble import RandomForestClassifier
                t5 = df_swe_t[df_swe_t["turn_number"] <= 5]
                X_df = t5.groupby("session_id").agg(
                    t5_tokens=("input_tokens", "sum"),
                    t5_cost=("turn_cost", "sum"),
                    t5_errors=("has_error", "sum"),
                    t5_out_len=("output_length", "mean"),
                ).reset_index()
                ml_data = pd.merge(df_swe_s, X_df, on="session_id", how="inner")

                # Không dùng feature 'model': Minimax & DeepSeek gần 100% lỗi trên SWE-bench nên
                # model gần như đoán đúng 'failed' tuyệt đối, lấn át tín hiệu hành vi 5-turn đầu
                # mà biểu đồ này muốn đo (t5_tokens, t5_cost, t5_errors, t5_out_len).
                features = ["t5_tokens", "t5_cost", "t5_errors", "t5_out_len"]
                feat_label = {"t5_tokens": "Tokens (5 turns đầu)", "t5_cost": "Chi phí (5 turns đầu)",
                               "t5_errors": "Số lỗi (5 turns đầu)", "t5_out_len": "Output Length TB (5 turns đầu)"}

                if len(ml_data) > 20:
                    X = ml_data[features]
                    y = ml_data["failed"]

                    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                    clf.fit(X, y)

                    if len(clf.classes_) > 1:
                        importances = clf.feature_importances_
                        feat_names = [feat_label[f] for f in features]
                        fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values("Importance", ascending=True)
                        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation='h', title="")
                        fig_fi.update_traces(marker_color=CLR_BLUE)
                        pls(fig_fi, h=280, ml=140)
                        fig_fi.update_layout(xaxis_title="Mức độ quan trọng (%)", yaxis_title="", xaxis_tickformat=".0%")
                        st.plotly_chart(fig_fi, key="ml_fi_new", use_container_width=True, config={"displayModeBar":False})
                        st.markdown(f'<div class="cap">Thuật toán học từ {len(ml_data)} phiên SWE-bench, chỉ dùng tín hiệu hành vi 5-turn đầu (không dùng model làm feature để tránh dự báo "ăn gian").</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(empty_html(), unsafe_allow_html=True)
                else:
                    st.markdown('<div class="empty-state">Không đủ dữ liệu huấn luyện ML.</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Lỗi khi chạy ML: {str(e)}")



    _son_swe = df_swe_s[df_swe_s["model"]=="claude-sonnet-4-6"]
    _son_resolved_tot = int((_son_swe["resolved"]==1).sum())
    _son_resolved_at_cut = int(((_son_swe["resolved"]==1) & (_son_swe["n_turns"]<=TURN_CUT)).sum())
    _pres_pct = (_son_resolved_at_cut/_son_resolved_tot*100) if _son_resolved_tot>0 else 0

    st.markdown(f"""<div class="tbox">
        <b>🎯 QUY TẮC ĐIỀU PHỐI THÔNG MINH (§03):</b><br>
        1. <b>Hard Cut-off:</b> Cài đặt <code>max_turns = {TURN_CUT}</code> cho Agent Runner Claude Sonnet.<br>
        2. <b>Hiệu quả kinh tế:</b> Chặn đứng chi phí thiêu đốt lãng phí, bảo toàn {_pres_pct:.1f}% số session đã từng thành công ({_son_resolved_at_cut}/{_son_resolved_tot}) — vì từ Turn {TURN_CUT} trở đi, số session thành công tăng thêm rất chậm.<br>
        3. <b>Tối ưu chi phí biên:</b> Chi phí trung bình để giải quyết xong 1 issue sẽ tăng vọt nếu vượt quá ngưỡng này — tiếp tục chạy chỉ làm giảm hiệu quả đầu tư.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 · KHUYẾN NGHỊ
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-hdr">§04. KHUYẾN NGHỊ — Các giải pháp tối ưu dòng tiền</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="cap" style="margin:-6px 0 10px;">📌 Toàn bộ tính toán ROI dưới đây cố định trên ngân sách gốc SWE-bench ({fc(swe_budget)}) — không phụ thuộc bộ lọc Model/Benchmark phía trên.</div>', unsafe_allow_html=True)

    # Tính toán ĐỘNG cho KN1 từ dữ liệu SWE-bench cố định
    _cost_base = df_swe_s["total_cost"].sum()
    _res_rate_base = df_swe_s["resolved"].mean() * 100 if not df_swe_s.empty else 0.0
    
    # Chi phí sau khi áp dụng Circuit Breaker tại TURN_CUT
    _t_cut = df_swe_t[df_swe_t["turn_number"] <= TURN_CUT]
    _cost_after_cb = _t_cut["turn_cost"].sum()
    _saving_cb = max(0.0, _cost_base - _cost_after_cb)
    _pct_saving_cb = (_saving_cb / _cost_base * 100) if _cost_base > 0 else 0.0
    
    # Resolve rate sau khi cắt (chỉ tính những session giải quyết xong trong vòng TURN_CUT turns)
    _res_after_cb = df_swe_s[(df_swe_s["resolved"] == 1) & (df_swe_s["n_turns"] <= TURN_CUT)]["session_id"].count()
    _res_rate_after_cb = (_res_after_cb / len(df_swe_s) * 100) if not df_swe_s.empty else 0.0
    _res_diff = _res_rate_base - _res_rate_after_cb
    
    kn1, kn2 = st.columns(2)
    with kn1:
        with get_card():
            st.markdown(f'<div class="panel-title">Khuyến nghị 1: Thiết lập tự động ngắt ở bước số {TURN_CUT}</div>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="warn-bar" style="background:rgba(14,165,233,0.12); border-color:#0ea5e9; color:{SKY_D}; margin-bottom:10px;">💡 <b>Hiệu quả:</b> Tiết kiệm <b>{_pct_saving_cb:.1f}% ngân sách ({fc(_saving_cb)})</b> — Chỉ giảm <b>{_res_diff:.1f}% Resolve Rate</b></div>', unsafe_allow_html=True)
            
            fkn1 = go.Figure()
            # Bar for cost
            for lbl, yv, clr in [("Trước khi cắt", _cost_base, MUT), (f"Sau cắt (≤{TURN_CUT})", _cost_after_cb, CLR_POS)]:
                fkn1.add_trace(go.Bar(
                    name=lbl, x=["Tổng Chi Phí ($)"], y=[yv], marker_color=clr,
                    text=[fc(yv)], textposition="outside",
                    textfont=dict(size=11, family="JetBrains Mono"),
                    hovertemplate=f"<b>{lbl}</b><br>Chi phí: $%{{y:,.2f}}<extra></extra>",
                    yaxis="y"
                ))
            # Bar for resolve rate
            for lbl, val, clr in [("Trước khi cắt", _res_rate_base, MUT), (f"Sau cắt (≤{TURN_CUT})", _res_rate_after_cb, CLR_POS)]:
                fkn1.add_trace(go.Bar(
                    name=lbl, x=["Resolve Rate (%)"], y=[val], marker_color=clr,
                    text=[fp(val)], textposition="outside",
                    textfont=dict(size=11, family="JetBrains Mono"),
                    showlegend=False,
                    hovertemplate=f"<b>{lbl}</b><br>Resolve: %{{y:.1f}}%<extra></extra>",
                    yaxis="y2"
                ))
            pls(fkn1, h=300)
            max_c_disp = max(_cost_base, _cost_after_cb, 1.0) * 1.25
            max_r_disp = max(_res_rate_base, _res_rate_after_cb, 10.0) * 1.35
            fkn1.update_layout(
                barmode="group",
                margin=dict(l=48, r=48, t=30, b=30),
                yaxis=dict(title="Tổng chi phí ($)", tickformat="$,.0f", range=[0, max_c_disp]),
                yaxis2=dict(title="Resolve Rate (%)", overlaying="y", side="right", ticksuffix="%", range=[0, min(100, max_r_disp)]),
                legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10.5))
            )
            st.plotly_chart(fkn1, key="chart_kn1", use_container_width=True, config={"displayModeBar":False})
            st.markdown(f'<div class="cap">{fc(_cost_base)} → {fc(_cost_after_cb)} (tiết kiệm {fc(_saving_cb)}), Resolve: {_res_rate_base:.1f}% → {_res_rate_after_cb:.1f}%.</div>', unsafe_allow_html=True)
    
    with kn2:
        with get_card():
            st.markdown('<div class="panel-title">Khuyến nghị 2: Ước tính số tiền có thể tiết kiệm được</div>', unsafe_allow_html=True)
            
            # Tính toán động các trụ cột thu hồi lãng phí (cố định SWE-bench):
            # 1. Thu hồi từ Circuit Breaker
            # 2. Thu hồi từ Dynamic Routing (loại bỏ minimax & deepseek 100% fail)
            w_rout = wasted_narrow_swe
            net_optimal = max(0.0, _cost_base - _saving_cb - w_rout)
            
            fkn2 = go.Figure(go.Waterfall(
                orientation="v",
                measure=["absolute", "relative", "relative", "total"],
                x=["Ngân Sách Gốc", f"Thu Hồi CB (≤{TURN_CUT})", "Thu Hồi Routing", "Ngân Sách Tối Ưu"],
                y=[_cost_base, -_saving_cb, -w_rout, None],
                connector=dict(line=dict(color="rgba(14,165,233,0.3)", width=1.5)),
                decreasing=dict(marker=dict(color=CLR_POS)),
                increasing=dict(marker=dict(color=CLR_NEG)),
                totals=dict(marker=dict(color=CLR_BLUE)),
                text=[fc(_cost_base), f"-{fc(_saving_cb)}", f"-{fc(w_rout)}", fc(net_optimal)],
                textposition="outside", textfont=dict(size=11, family="JetBrains Mono"),
                hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>"
            ))
            pls(fkn2, h=300)
            fkn2.update_layout(yaxis_tickformat="$,.2f", showlegend=False)
            st.plotly_chart(fkn2, key="chart_kn2", use_container_width=True, config={"displayModeBar":False})
            st.markdown(f'<div class="cap w">Tổng thu hồi tiềm năng: <b>{fc(_saving_cb + w_rout)}</b> ({((_saving_cb + w_rout)/_cost_base*100) if _cost_base>0 else 0:.1f}% ngân sách).</div>', unsafe_allow_html=True)

    with get_card():
        st.markdown('<div class="panel-title">Khuyến nghị 3: Không dùng AI kém cho các công việc khó (SWE-bench)</div>', unsafe_allow_html=True)
        if ms_swe.empty:
            st.markdown(empty_html(), unsafe_allow_html=True)
        else:
            k3a, k3b = st.columns(2)
            sz = ms_swe["pct_turns"].fillna(0); r_ = sz.max()-sz.min()
            ms3 = ms_swe.copy(); ms3["bs"] = 20+(sz-sz.min())/r_*60 if r_>0 else 40
            with k3a:
                st.markdown(f"**Hiện trạng {len(SWE_MODELS)} models (SWE-bench):**")
                f9 = px.scatter(ms3, x="cost_turn", y="err_rate", color="model", color_discrete_map=MODEL_COLORS,
                                size="bs", size_max=75, text="model", custom_data=["sessions","total_turns"])
                f9.update_traces(hovertemplate="<b>%{text}</b><br>Cost/turn: $%{x:,.4f}<br>Error: %{y:.1%}<extra></extra>",
                                 textposition="top center", textfont=dict(size=9.5, color=H, family="Inter"))
                pls(f9, h=265); f9.update_layout(yaxis_tickformat=".0%", xaxis_tickformat="$,.4f", showlegend=False)
                st.plotly_chart(f9, key="chart_kn3_before", use_container_width=True, config={"displayModeBar":False})
            with k3b:
                st.markdown("**Sau Dynamic Routing (chỉ Sonnet):**")
                ma = ms3[ms3["model"]=="claude-sonnet-4-6"].copy()
                if not ma.empty:
                    s2 = ma["pct_turns"].fillna(0); r2 = s2.max()-s2.min()
                    ma["bs2"] = 40.0 if r2==0 else 20+(s2-s2.min())/r2*60
                    f10 = px.scatter(ma, x="cost_turn", y="err_rate", color="model", color_discrete_map=MODEL_COLORS,
                                     size="bs2", size_max=75, text="model", custom_data=["sessions","total_turns"])
                    f10.update_traces(hovertemplate="<b>%{text}</b><br>Cost/turn: $%{x:,.4f}<br>Error: %{y:.1%}<extra></extra>",
                                      textposition="top center", textfont=dict(size=9.5, color=H, family="Inter"))
                    pls(f10, h=265); f10.update_layout(yaxis_tickformat=".0%", xaxis_tickformat="$,.4f", showlegend=False)
                    st.plotly_chart(f10, key="chart_kn3_after", use_container_width=True, config={"displayModeBar":False})
        st.markdown(f'<div class="cap w">Triệt tiêu hoàn toàn <b>{fc(wasted_narrow_swe)}</b> chi phí lãng phí bằng cách chuyển tải sang Sonnet.</div>', unsafe_allow_html=True)

    # ACTION PLAN & EXECUTIVE SUMMARY TABLE
    st.markdown(f"""
    <div style="margin-top: 16px;">
        <table class="dt">
            <thead>
                <tr>
                    <th width="15%">Đề xuất</th>
                    <th width="45%">Nội dung hành động cụ thể</th>
                    <th width="22%" class="r">Tiết kiệm / Tác động</th>
                    <th width="18%">Ưu tiên & Lead-time</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><b>P0: Circuit Breaker</b></td>
                    <td>Cài đặt ngưỡng ngắt cứng <code>max_turns = {TURN_CUT}</code> cho runner Sonnet. Cắt đứng chu kỳ lặp vô vọng.</td>
                    <td class="r"><b>+{fc(_saving_cb)}</b> ({_pct_saving_cb:.1f}%)</td>
                    <td><span style="color:#0ea5e9;font-weight:800;">P0 · 1 Tuần</span></td>
                </tr>
                <tr>
                    <td><b>P1: Dynamic Routing</b></td>
                    <td>Ngừng cấp quota cho Minimax & Deepseek ở task phức tạp; định tuyến trực tiếp sang Sonnet.</td>
                    <td class="r"><b>+{fc(wasted_narrow_swe)}</b> thu hồi 100%</td>
                    <td><span style="color:#10b981;font-weight:800;">P1 · 3 Ngày</span></td>
                </tr>
                <tr>
                    <td><b>P2: Context Pruning</b></td>
                    <td>Tóm tắt ngữ cảnh khi context vượt >20k tokens nhằm chặn đà tăng cấp số nhân của chi phí per turn.</td>
                    <td class="r"><b>Giảm 30-40%</b> token bloat</td>
                    <td><span style="color:#f59e0b;font-weight:800;">P2 · 2 Tuần</span></td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""<div class="tbox">
        <b>📋 LỘ TRÌNH TRIỂN KHAI (§04 · dựa trên SWE-bench {fc(_cost_base)}):</b><br>
        1. <b>P0 — Circuit Breaker (Turn {TURN_CUT}):</b> Triển khai ngay lập tức. Tiết kiệm ước tính <b>{fc(_saving_cb)}</b>, rủi ro ảnh hưởng năng lực &lt;{_res_diff:.1f} điểm %.<br>
        2. <b>P1 — Dynamic Routing:</b> Ngừng quota Minimax &amp; Deepseek. Thu hồi trực tiếp <b>{fc(wasted_narrow_swe)}</b> chi phí thiêu đốt.<br>
        3. <b>P2 — Context Pruning:</b> Áp dụng tóm tắt ngữ cảnh tự động khi Token vượt 20K.<br><br>
        👉 <b>Kỳ vọng:</b> Giảm <b>{((_saving_cb + wasted_narrow_swe)/_cost_base*100) if _cost_base>0 else 0:.1f}% chi phí vận hành</b> trên SWE-bench mà không suy giảm đáng kể tỷ lệ hoàn thành tác vụ.
    </div>""", unsafe_allow_html=True)

# ─── FOOTNOTE ─────────────────────────────────────────────────────────────────
st.markdown(f"""<div class="fnt">
    <b>(1) Định nghĩa (Tab 01 · theo bộ lọc):</b> wasted_narrow={fc(wasted_narrow)} (Minimax+Deepseek 100% fail); wasted_full={fc(wasted_full)} (mọi session fail 100%).<br>
    <b>(1b) Định nghĩa (Tab 02-04 · cố định SWE-bench):</b> wasted_narrow_swe={fc(wasted_narrow_swe)}; wasted_full_swe={fc(wasted_full_swe)} trên tổng {fc(swe_budget)} ngân sách SWE-bench.<br>
    <b>(2) Dữ liệu:</b> processed_agentic_traces.csv · {fi(len(df_sess_raw))} sessions · {fi(len(df_turns_raw))} traces · {fc(df_turns_raw['turn_cost'].sum())} tổng ngân sách gốc.<br>
    <b>(3) Bộ lọc (chỉ Tab 01):</b> Model: {sel_models} | Benchmark: {sel_bench} | Prompt: {sel_prompt} | Turn: {sel_turns} | Theme: {("Dark" if is_dark else "Light")} Mode.
</div>""", unsafe_allow_html=True)
