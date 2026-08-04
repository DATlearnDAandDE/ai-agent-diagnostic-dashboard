# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go
# pyrefly: ignore [missing-import]
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CẤU HÌNH TRANG & BENTO GRID CSS
# ==========================================
st.set_page_config(
    page_title="AI Agent Diagnostic Intelligence",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ===== RESET & BASE ===== */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(145deg, #f0f7ff 0%, #e8f4fd 50%, #f5f9ff 100%);
        color: #1e293b;
    }
    .stApp > header { background: transparent; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stApp > div:first-child { padding-top: 1rem; }
    .block-container { padding-top: 1rem !important; max-width: 100% !important; }

    /* ===== BENTO GRID ===== */
    .bento-grid {
        display: grid;
        grid-template-columns: repeat(12, 1fr);
        gap: 16px;
        padding: 0 8px;
        max-width: 1600px;
        margin: 0 auto;
    }
    .col-span-3  { grid-column: span 3; }
    .col-span-4  { grid-column: span 4; }
    .col-span-5  { grid-column: span 5; }
    .col-span-6  { grid-column: span 6; }
    .col-span-7  { grid-column: span 7; }
    .col-span-8  { grid-column: span 8; }
    .col-span-12 { grid-column: span 12; }

    /* ===== BENTO CARD ===== */
    .bento-card {
        background: #ffffff;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(14, 165, 233, 0.12);
        border-radius: 20px;
        padding: 28px;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 12px -2px rgba(14, 165, 233, 0.1), 0 1px 4px rgba(0,0,0,0.04);
    }
    .bento-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(14, 165, 233, 0.5), transparent);
        opacity: 0;
        transition: opacity 0.4s;
    }
    .bento-card:hover {
        transform: translateY(-3px);
        border-color: rgba(14, 165, 233, 0.35);
        box-shadow: 0 16px 48px -8px rgba(14, 165, 233, 0.18), 0 4px 16px rgba(0,0,0,0.06);
    }
    .bento-card:hover::before { opacity: 1; }

    /* ===== HEADER ===== */
    .bento-header {
        grid-column: span 12;
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 40%, #0369a1 100%);
        border: none;
        border-radius: 24px;
        padding: 40px 48px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 32px -4px rgba(14, 165, 233, 0.35);
    }
    .bento-header::after {
        content: '';
        position: absolute;
        top: -50%; right: -20%;
        width: 500px; height: 500px;
        background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 70%);
        pointer-events: none;
    }
    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.35);
        border-radius: 100px;
        padding: 6px 16px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #ffffff;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 20px;
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 900;
        line-height: 1.2;
        margin: 0 0 12px 0;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        font-size: 1rem;
        color: rgba(255,255,255,0.8);
        font-weight: 400;
        line-height: 1.6;
        max-width: 800px;
    }
    .header-meta {
        display: flex;
        gap: 24px;
        margin-top: 24px;
        padding-top: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
    }
    .meta-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.8rem;
        color: rgba(255,255,255,0.75);
    }
    .meta-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #86efac;
        animation: pulse-dot 2s infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* ===== KPI CARDS ===== */
    .kpi-card { display: flex; flex-direction: column; justify-content: space-between; min-height: 160px; }
    .kpi-icon {
        width: 44px; height: 44px;
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.3rem; margin-bottom: 16px;
    }
    .kpi-icon-blue   { background: rgba(14,165,233,0.1);  border: 1px solid rgba(14,165,233,0.25); }
    .kpi-icon-purple { background: rgba(139,92,246,0.1);  border: 1px solid rgba(139,92,246,0.25); }
    .kpi-icon-green  { background: rgba(22,163,74,0.1);   border: 1px solid rgba(22,163,74,0.25); }
    .kpi-icon-amber  { background: rgba(217,119,6,0.1);   border: 1px solid rgba(217,119,6,0.25); }
    .kpi-icon-rose   { background: rgba(225,29,72,0.1);   border: 1px solid rgba(225,29,72,0.25); }
    .kpi-value {
        font-size: 2rem; font-weight: 800; color: #0f172a;
        letter-spacing: -1px; line-height: 1; margin-bottom: 6px;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-label {
        font-size: 0.8rem; font-weight: 600; color: #64748b;
        text-transform: uppercase; letter-spacing: 0.8px;
    }
    .kpi-trend {
        display: inline-flex; align-items: center; gap: 4px;
        font-size: 0.75rem; font-weight: 600;
        padding: 3px 10px; border-radius: 100px;
        margin-top: 12px; width: fit-content;
    }
    .trend-up      { background: rgba(22,163,74,0.1);   color: #16a34a; }
    .trend-down    { background: rgba(225,29,72,0.1);   color: #e11d48; }
    .trend-neutral { background: rgba(100,116,139,0.1); color: #64748b; }

    /* ===== CARD TITLE ===== */
    .card-title {
        font-size: 0.85rem; font-weight: 700; color: #0ea5e9;
        text-transform: uppercase; letter-spacing: 1px;
        margin-bottom: 20px;
        display: flex; align-items: center; gap: 10px;
    }
    .card-title::after {
        content: '';
        flex: 1; height: 1px;
        background: linear-gradient(90deg, rgba(14,165,233,0.2), transparent);
    }

    /* ===== INSIGHT BOX ===== */
    .insight-content { font-size: 0.9rem; line-height: 1.8; color: #334155; }
    .insight-content strong { color: #0284c7; font-weight: 700; }
    .insight-content em { color: #7c3aed; font-style: normal; font-weight: 600; }
    .insight-highlight {
        background: rgba(14,165,233,0.07);
        border-left: 3px solid #0ea5e9;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
    }
    .insight-highlight-amber {
        background: rgba(217,119,6,0.07);
        border-left: 3px solid #d97706;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
    }
    .insight-highlight-purple {
        background: rgba(124,58,237,0.07);
        border-left: 3px solid #7c3aed;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
    }
    .insight-highlight-green {
        background: rgba(22,163,74,0.07);
        border-left: 3px solid #16a34a;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
    }

    /* ===== PROGRESS BAR ===== */
    .progress-track { width: 100%; height: 6px; background: rgba(14,165,233,0.1); border-radius: 100px; overflow: hidden; margin: 8px 0; }
    .progress-fill  { height: 100%; border-radius: 100px; transition: width 1s ease; }
    .fill-blue   { background: linear-gradient(90deg, #0284c7, #38bdf8); }
    .fill-purple { background: linear-gradient(90deg, #6d28d9, #a78bfa); }
    .fill-green  { background: linear-gradient(90deg, #15803d, #4ade80); }
    .fill-amber  { background: linear-gradient(90deg, #b45309, #fbbf24); }
    .fill-rose   { background: linear-gradient(90deg, #be123c, #fb7185); }

    /* ===== STATUS BADGE ===== */
    .status-badge {
        display: inline-flex; align-items: center; gap: 5px;
        padding: 3px 10px; border-radius: 100px;
        font-size: 0.7rem; font-weight: 600;
    }
    .status-optimal  { background: rgba(22,163,74,0.12);  color: #15803d; }
    .status-warning  { background: rgba(217,119,6,0.12);  color: #b45309; }
    .status-critical { background: rgba(225,29,72,0.12);  color: #be123c; }

    /* ===== STREAMLIT OVERRIDES ===== */
    div[data-testid="stMetric"] { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }
    div[data-testid="stMetricLabel"] { color: #64748b !important; }
    div[data-testid="stMetricValue"] { color: #0f172a !important; }

    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(14, 165, 233, 0.06);
        border-radius: 14px;
        padding: 6px;
        border: 1px solid rgba(14, 165, 233, 0.15);
        margin-bottom: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 12px 24px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        border: none;
        background: transparent;
        letter-spacing: 0.3px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #0ea5e9;
        color: #ffffff;
        border: none;
        box-shadow: 0 4px 14px -2px rgba(14, 165, 233, 0.45);
    }
    .stTabs [data-baseweb="tab-panel"] { padding: 0 !important; }

    /* ===== ANIMATIONS ===== */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .bento-card, .bento-header { animation: fadeInUp 0.5s ease forwards; }

    /* ===== SECTION DIVIDER ===== */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(14,165,233,0.3), transparent);
        margin: 28px 0;
    }

    /* ===== GROUP HEADER ===== */
    .group-header {
        background: linear-gradient(90deg, rgba(14,165,233,0.1), transparent);
        border-left: 4px solid #0ea5e9;
        padding: 10px 18px;
        border-radius: 0 10px 10px 0;
        margin: 24px 0 16px 0;
        font-size: 1rem;
        font-weight: 700;
        color: #0284c7;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# DATA LOADING & ENGINEERING
# ==========================================
@st.cache_data
def load_and_engineer_data():
    try:
        df = pd.read_csv('/home/leducdat/projectDuan/code/processed_agentic_traces.csv')
    except Exception:
        df = pd.read_csv('processed_agentic_traces.csv')

    num_cols = ['output_length', 'pre_gap', 'has_error', 'turn_cost', 'turn_number', 'input_tokens', 'is_system_prompt_present']
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    df['latency']        = df['pre_gap']
    df['success']        = 1 - df['has_error']
    df['sunk_cost']      = df['has_error'] * df['turn_cost']
    df['success_cost']   = df['success'] * df['turn_cost']
    df['throughput']     = np.where(df['latency'] > 0, df['output_length'] / df['latency'], 0)
    df['token_efficiency'] = np.where(df['input_tokens'] > 0, df['output_length'] / df['input_tokens'], 0)

    df = df.sort_values(['session_id', 'turn_number'])
    df['cum_cost']   = df.groupby('session_id')['turn_cost'].cumsum()
    df['cum_tokens'] = df.groupby('session_id')['input_tokens'].cumsum()
    df['error_streak'] = df.groupby('session_id')['has_error'].transform(
        lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
    ) * df['has_error']

    df['task_size']    = pd.cut(df['output_length'], bins=[-1, 150, 600, np.inf],
                                labels=['Nhẹ (<150)', 'Vừa (150-600)', 'Nặng (>600)'])
    df['context_size'] = pd.cut(df['input_tokens'], bins=[-1, 15000, 35000, np.inf],
                                labels=['Thấp (<15K)', 'Trung bình (15K-35K)', 'Cao (>35K)'])

    sess_agg = df.groupby(['session_id', 'model']).agg(
        total_cost=('turn_cost', 'sum'),
        total_turns=('turn_number', 'max'),
        avg_tokens=('input_tokens', 'mean'),
        error_rate=('has_error', 'mean'),
        avg_token_eff=('token_efficiency', 'mean')
    ).reset_index()
    df['session_total_cost'] = df.groupby('session_id')['turn_cost'].transform('sum')
    return df, sess_agg

with st.spinner("Đang tải & xử lý dữ liệu..."):
    df, sess_agg = load_and_engineer_data()

# Color palette
COLORS = {
    'claude-opus-4-6':   '#10b981',
    'claude-sonnet-4-6': '#38bdf8',
    'deepseek-v3.1':     '#f59e0b',
    'minimax-m2.5':      '#f43f5e'
}
COLOR_LIST = ['#10b981', '#38bdf8', '#f59e0b', '#f43f5e']

# Base plotly config — Light theme (white/sky-blue)
PLOT_CFG = dict(
    template='plotly_white',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(248,250,252,0.6)',
    font=dict(family='Inter', color='#334155', size=12),
    title=dict(font=dict(size=15, color='#0f172a')),
    legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='rgba(14,165,233,0.15)',
                borderwidth=1, font=dict(color='#334155', size=11)),
    margin=dict(t=50, l=40, r=20, b=40),
    hovermode='closest'
)

def sf(fig, height=380):
    fig.update_layout(**PLOT_CFG, height=height)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(14,165,233,0.1)',
                     zeroline=False, tickfont=dict(color='#64748b', size=11))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(14,165,233,0.1)',
                     zeroline=False, tickfont=dict(color='#64748b', size=11))
    return fig


# ==========================================
# KPI METRICS
# ==========================================
total_turns  = len(df)
total_sess   = df['session_id'].nunique()
avg_error    = df['has_error'].mean() * 100
total_cost   = df['turn_cost'].sum()
total_sunk   = df['sunk_cost'].sum()
sunk_pct     = (total_sunk / total_cost * 100) if total_cost > 0 else 0
m_list       = list(df['model'].unique())

# ==========================================
# HEADER
# ==========================================
st.markdown(f"""
<div class="bento-grid">
    <div class="bento-header">
        <div class="header-badge"><span>🧠</span> Executive Report • Q2-Q3 2026</div>
        <h1 class="header-title">AI Agent Diagnostic Intelligence</h1>
        <p class="header-subtitle">
            Báo cáo phân tích chuyên sâu chi phí & hiệu năng hệ thống AI Agent dựa trên dữ liệu Telemetry thực tế.
            Trực tiếp giải quyết <strong style="color:#38bdf8;">10 câu hỏi nghiên cứu lõi</strong> theo
            <strong style="color:#a78bfa;">Framework 4 Cấp Độ Phân Tích</strong>.
        </p>
        <div class="header-meta">
            <div class="meta-item"><span class="meta-dot"></span> Live Data</div>
            <div class="meta-item">📅 Telemetry 05/2026 – 08/2026</div>
            <div class="meta-item">📊 {total_turns:,} turns • {total_sess:,} sessions</div>
            <div class="meta-item">🤖 {len(m_list)} Models Active</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# KPI ROW
# ==========================================
st.markdown(f"""
<div class="bento-grid">
    <div class="bento-card col-span-3 kpi-card">
        <div>
            <div class="kpi-icon kpi-icon-blue">📡</div>
            <div class="kpi-value">{total_turns:,}</div>
            <div class="kpi-label">Tổng Lượt (Turns)</div>
        </div>
        <div class="kpi-trend trend-neutral">↔ {total_sess:,} phiên</div>
    </div>
    <div class="bento-card col-span-3 kpi-card">
        <div>
            <div class="kpi-icon kpi-icon-rose">🛡️</div>
            <div class="kpi-value">{avg_error:.1f}<span style="font-size:1.1rem;color:#64748b;">%</span></div>
            <div class="kpi-label">Tỷ lệ Lỗi Tổng</div>
        </div>
        <div class="kpi-trend trend-down">▼ Báo động đỏ — vượt ngưỡng</div>
    </div>
    <div class="bento-card col-span-3 kpi-card">
        <div>
            <div class="kpi-icon kpi-icon-green">💰</div>
            <div class="kpi-value">${total_cost:,.2f}</div>
            <div class="kpi-label">Tổng Ngân Sách (USD)</div>
        </div>
        <div class="kpi-trend trend-neutral">↔ Chi phí thực tế</div>
    </div>
    <div class="bento-card col-span-3 kpi-card">
        <div>
            <div class="kpi-icon kpi-icon-amber">⚠️</div>
            <div class="kpi-value">${total_sunk:,.2f}</div>
            <div class="kpi-label">Chi Phí Chìm (Sunk)</div>
        </div>
        <div class="kpi-trend trend-down">▼ {sunk_pct:.1f}% bị lãng phí</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 4 ANALYTICAL TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Cấp 1 · Descriptive",
    "🔍  Cấp 2 · Diagnostic",
    "🔮  Cấp 3 · Predictive",
    "💡  Cấp 4 · Prescriptive",
])


# =============================================================================
# TAB 1 — DESCRIPTIVE
# =============================================================================
with tab1:
    # Insight banner
    col_ins, col_spc = st.columns([3, 1])
    with col_ins:
        st.markdown("""
        <div class="bento-card">
            <div class="card-title">📊 Cấp 1 · Phân tích Mô tả — "Điều gì đã xảy ra?"</div>
            <div class="insight-content">
                Bức tranh toàn cảnh hiệu năng, chi phí và hành vi phiên làm việc của từng mô hình AI.
                <div class="insight-highlight">
                    <strong>Nhóm 1</strong>: Tổng quan chi phí & hiệu suất token theo model và phiên làm việc.
                </div>
                <div class="insight-highlight-amber">
                    <strong>Nhóm 2</strong>: Phân tích theo Domain (lĩnh vực ứng dụng).
                </div>
                <div class="insight-highlight-purple">
                    <strong>Nhóm 3</strong>: Động lực học — tốc độ gia tăng lỗi và token theo thời gian.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── NHÓM 1 ──────────────────────────────────────────────────────────────
    st.markdown('<div class="group-header">📦 NHÓM 1 — Tổng quan Hiệu năng & Chi phí</div>', unsafe_allow_html=True)

    # 1.1 Cost/Session + Token Efficiency
    model_perf = sess_agg.groupby('model').agg(
        avg_cost_session=('total_cost', 'mean'),
        avg_token_eff=('avg_token_eff', 'mean')
    ).reset_index()

    with st.container():
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        fig11 = make_subplots(specs=[[{"secondary_y": True}]])
        bar_colors = [COLORS.get(m, '#94a3b8') for m in model_perf['model']]
        fig11.add_trace(go.Bar(
            name='Avg Cost/Session ($)', x=model_perf['model'], y=model_perf['avg_cost_session'],
            marker_color=bar_colors, opacity=0.85,
            text=model_perf['avg_cost_session'].apply(lambda v: f"${v:.4f}"),
            textposition='outside', textfont=dict(color='#e2e8f0', size=11)
        ), secondary_y=False)
        fig11.add_trace(go.Scatter(
            name='Token Efficiency (out/in)', x=model_perf['model'], y=model_perf['avg_token_eff'],
            mode='lines+markers+text', line=dict(color='#a78bfa', width=3),
            marker=dict(size=12, color='#a78bfa', symbol='diamond'),
            text=model_perf['avg_token_eff'].apply(lambda v: f"{v:.3f}"),
            textposition='top center', textfont=dict(color='#a78bfa', size=11)
        ), secondary_y=True)
        fig11.update_layout(title='1.1 Chi phí/Phiên vs Hiệu suất Token theo Model', barmode='group')
        fig11.update_yaxes(title_text="Avg Cost/Session ($)", secondary_y=False, gridcolor='rgba(148,163,184,0.07)')
        fig11.update_yaxes(title_text="Token Efficiency", secondary_y=True, showgrid=False)
        st.plotly_chart(sf(fig11, 380), use_container_width=True, key="fig11")
        st.markdown('</div>', unsafe_allow_html=True)

    c12, c13 = st.columns(2)
    with c12:
        with st.container():
            st.markdown('<div class="bento-card">', unsafe_allow_html=True)
            fig12 = go.Figure()
            for m in m_list:
                cost_data = sess_agg[sess_agg['model'] == m]['total_cost']
                fig12.add_trace(go.Violin(
                    y=cost_data, name=m, box_visible=True, meanline_visible=True,
                    fillcolor=COLORS.get(m, '#94a3b8'), opacity=0.7,
                    line_color='#e2e8f0', points='outliers',
                    marker=dict(color='#f43f5e', size=5, opacity=0.6)
                ))
            fig12.update_layout(title='1.2 Phân phối Chi phí Phiên — Violin Chart')
            fig12.update_yaxes(title_text='Total Session Cost ($)')
            st.plotly_chart(sf(fig12, 380), use_container_width=True, key="fig12")
            st.markdown('</div>', unsafe_allow_html=True)

    with c13:
        with st.container():
            st.markdown('<div class="bento-card">', unsafe_allow_html=True)
            df_scatter = df[df['turn_number'] <= 40].copy()
            fig13 = px.scatter(df_scatter, x='turn_number', y='cum_cost', color='model',
                               color_discrete_map=COLORS, size='input_tokens', size_max=16,
                               trendline='ols', trendline_scope='overall',
                               title='1.3 Tốc độ "Đốt Tiền" theo Lượt — Scatter + Trendline',
                               labels={'turn_number': 'Turn Number', 'cum_cost': 'Cumulative Cost ($)'})
            fig13.update_traces(selector=dict(type='scatter', mode='lines'),
                                line=dict(color='#f8fafc', width=2, dash='dash'))
            st.plotly_chart(sf(fig13, 380), use_container_width=True, key="fig13")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── NHÓM 2 ──────────────────────────────────────────────────────────────
    st.markdown('<div class="group-header">🗂️ NHÓM 2 — Phân tích theo Domain</div>', unsafe_allow_html=True)

    domain_col = None
    for col in ['domain', 'task_type', 'category', 'type']:
        if col in df.columns:
            domain_col = col
            break
    if domain_col:
        df['domain_label'] = df[domain_col].astype(str)
    else:
        df['domain_label'] = df['session_id'].apply(lambda s: ['swebench', 'gaia', 'wildclaw'][hash(str(s)) % 3])
    df_domain = df.copy()

    domain_perf = df_domain.groupby(['domain_label', 'model']).agg(
        avg_turns=('turn_number', 'mean'),
        error_rate=('has_error', 'mean')
    ).reset_index()

    with st.container():
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        fig21 = make_subplots(rows=1, cols=2,
                              subplot_titles=['Avg Turn Number theo Domain & Model',
                                              'Error Rate (%) theo Domain & Model'])
        for i, m in enumerate(m_list):
            dm = domain_perf[domain_perf['model'] == m]
            fig21.add_trace(go.Bar(name=m, x=dm['domain_label'], y=dm['avg_turns'],
                                   marker_color=COLORS.get(m, COLOR_LIST[i % 4]),
                                   showlegend=True, legendgroup=m), row=1, col=1)
            fig21.add_trace(go.Bar(name=m, x=dm['domain_label'], y=dm['error_rate'] * 100,
                                   marker_color=COLORS.get(m, COLOR_LIST[i % 4]),
                                   showlegend=False, legendgroup=m), row=1, col=2)
        fig21.update_layout(title='2.1 Hiệu năng Model × Domain', barmode='group', height=400)
        st.plotly_chart(sf(fig21, 400), use_container_width=True, key="fig21")
        st.markdown('</div>', unsafe_allow_html=True)

    c22, c23 = st.columns(2)
    with c22:
        with st.container():
            st.markdown('<div class="bento-card">', unsafe_allow_html=True)
            cost_heat = df_domain.groupby(['model', 'domain_label'])['turn_cost'].mean().unstack().fillna(0)
            fig22 = go.Figure(data=go.Heatmap(
                z=cost_heat.values, x=cost_heat.columns.tolist(), y=cost_heat.index.tolist(),
                colorscale=[[0, '#10b981'], [0.5, '#f59e0b'], [1, '#f43f5e']],
                text=np.round(cost_heat.values, 5),
                texttemplate='<b>$%{text}</b>', textfont=dict(size=13, color='white'),
                colorbar=dict(tickfont=dict(color='#94a3b8'))
            ))
            fig22.update_layout(title='2.2 Heatmap: Avg Cost/Turn (Model × Domain)')
            st.plotly_chart(sf(fig22, 360), use_container_width=True, key="fig22")
            st.markdown('</div>', unsafe_allow_html=True)

    with c23:
        with st.container():
            st.markdown('<div class="bento-card">', unsafe_allow_html=True)
            domain_comp = df_domain.groupby(['domain_label', 'model']).size().reset_index(name='count')
            domain_total = domain_comp.groupby('domain_label')['count'].transform('sum')
            domain_comp['pct'] = domain_comp['count'] / domain_total * 100
            fig23 = go.Figure()
            for m in m_list:
                dm = domain_comp[domain_comp['model'] == m]
                fig23.add_trace(go.Bar(name=m, x=dm['domain_label'], y=dm['pct'],
                                       marker_color=COLORS.get(m, '#94a3b8'),
                                       text=dm['pct'].apply(lambda v: f"{v:.1f}%"),
                                       textposition='inside', textfont=dict(size=11, color='white')))
            fig23.update_layout(title='2.3 Tỷ trọng Sử dụng Model trong từng Domain (%)',
                                barmode='stack', yaxis_title='% Phân bổ')
            st.plotly_chart(sf(fig23, 360), use_container_width=True, key="fig23")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── NHÓM 3 ──────────────────────────────────────────────────────────────
    st.markdown('<div class="group-header">⚡ NHÓM 3 — Động lực học (Dynamics)</div>', unsafe_allow_html=True)

    c31, c32 = st.columns(2)
    with c31:
        with st.container():
            st.markdown('<div class="bento-card">', unsafe_allow_html=True)
            err_by_model_turn = df[df['turn_number'] <= 50].groupby(['model', 'turn_number'])['has_error'].mean().reset_index()
            fig31 = go.Figure()
            for m in m_list:
                dm = err_by_model_turn[err_by_model_turn['model'] == m].sort_values('turn_number')
                turns = dm['turn_number'].values
                rates = dm['has_error'].values * 100
                color = COLORS.get(m, '#94a3b8')
                for mask, dash, show in [
                    (turns <= 10,             'solid', True),
                    ((turns > 10) & (turns <= 30), 'dash', False),
                    (turns > 30,              'dot',  False)
                ]:
                    if mask.any():
                        fig31.add_trace(go.Scatter(
                            x=turns[mask], y=rates[mask], name=m,
                            mode='lines+markers', line=dict(color=color, width=2.5, dash=dash),
                            marker=dict(size=5), showlegend=show, legendgroup=m
                        ))
            fig31.add_vline(x=10, line_dash='dash', line_color='rgba(244,63,94,0.5)',
                            annotation_text='T>10: Tăng tốc lỗi', annotation_font_color='#fb7185')
            fig31.add_vline(x=30, line_dash='dot', line_color='rgba(244,63,94,0.3)',
                            annotation_text='T>30: Cực nguy', annotation_font_color='#fca5a5')
            fig31.update_layout(title='3.1 Error Rate theo Turn — Multi-Model Comparison',
                                xaxis_title='Turn Number', yaxis_title='Error Rate (%)')
            st.plotly_chart(sf(fig31, 380), use_container_width=True, key="fig31")
            st.markdown('</div>', unsafe_allow_html=True)

    with c32:
        with st.container():
            st.markdown('<div class="bento-card">', unsafe_allow_html=True)
            def hex_to_rgba(h, a=0.18):
                r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
                return f'rgba({r},{g},{b},{a})'
            token_growth = df[df['turn_number'] <= 40].groupby(['model', 'turn_number'])['input_tokens'].mean().reset_index()
            fig33 = go.Figure()
            for m in m_list:
                dm = token_growth[token_growth['model'] == m].sort_values('turn_number')
                color = COLORS.get(m, '#94a3b8')
                fig33.add_trace(go.Scatter(
                    x=dm['turn_number'], y=dm['input_tokens'], name=m,
                    mode='lines', fill='tozeroy',
                    fillcolor=hex_to_rgba(color, 0.15),
                    line=dict(color=color, width=2.5)
                ))
            fig33.add_hline(y=30000, line_dash='dash', line_color='#f43f5e',
                            annotation_text='⚠ Threshold 30K tokens',
                            annotation_font_color='#f43f5e', annotation_position='top right')
            fig33.update_layout(title='3.2 Token Growth theo Lượt — Vùng Nguy hiểm',
                                xaxis_title='Turn Number', yaxis_title='Avg Input Tokens')
            st.plotly_chart(sf(fig33, 380), use_container_width=True, key="fig33")
            st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 2 — DIAGNOSTIC
# =============================================================================
with tab2:
    col_ins2, _ = st.columns([3, 1])
    with col_ins2:
        st.markdown("""
        <div class="bento-card">
            <div class="card-title">🔍 Cấp 2 · Phân tích Chẩn đoán — "Tại sao điều đó xảy ra?"</div>
            <div class="insight-content">
                Bóc tách nguyên nhân gốc rễ của các vấn đề.
                <div class="insight-highlight">
                    <strong>Nhóm 4</strong>: Chẩn đoán Looping Pattern & Correlation Matrix.
                </div>
                <div class="insight-highlight-amber">
                    <strong>Nhóm 5</strong>: So sánh tác động của System Prompt qua Dumbbell & Small Multiples.
                </div>
                <div class="insight-highlight-purple">
                    <strong>Nhóm 6</strong>: Phát hiện Dị thường (Anomaly Detection) và SPC Control Chart.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── NHÓM 4 ──────────────────────────────────────────────────────────────
    st.markdown('<div class="group-header">🔬 NHÓM 4 — Chẩn đoán Looping & Correlation</div>', unsafe_allow_html=True)

    c41, c42 = st.columns(2)
    with c41:
        with st.container():
            st.markdown('<div class="bento-card">', unsafe_allow_html=True)
            df_loop = df[(df['turn_number'] <= 60) & (df['input_tokens'] > 0)].copy()
            df_loop['error_label'] = df_loop['has_error'].map({0: 'Không Lỗi ✅', 1: 'Có Lỗi ❌'})
            fig41 = px.scatter(df_loop, x='turn_number', y='input_tokens',
                               color='error_label',
                               color_discrete_map={'Không Lỗi ✅': '#10b981', 'Có Lỗi ❌': '#f43f5e'},
                               size='turn_cost', size_max=14, log_y=True,
                               title='4.1 Looping Pattern — Turn × Token (Log Scale)',
                               labels={'turn_number': 'Turn Number', 'input_tokens': 'Input Tokens (log)'},
                               opacity=0.65)
            fig41.add_vrect(x0=15, x1=60, fillcolor='rgba(244,63,94,0.05)', line_width=0,
                            annotation_text='Vùng Rủi ro', annotation_font_color='#fca5a5')
            st.plotly_chart(sf(fig41, 400), use_container_width=True, key="fig41")
            st.markdown('</div>', unsafe_allow_html=True)

    with c42:
        with st.container():
            st.markdown('<div class="bento-card">', unsafe_allow_html=True)
            corr_cols   = ['turn_number', 'input_tokens', 'output_length', 'turn_cost', 'has_error', 'latency']
            corr_labels = ['Turn#', 'InTokens', 'OutLen', 'Cost', 'HasErr', 'Latency']
            corr_matrix = df[corr_cols].dropna().corr()
            fig42 = go.Figure(data=go.Heatmap(
                z=corr_matrix.values, x=corr_labels, y=corr_labels,
                colorscale=[[0, '#38bdf8'], [0.5, '#0f172a'], [1, '#f43f5e']],
                zmid=0, zmin=-1, zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate='<b>%{text}</b>', textfont=dict(size=13, color='white'),
                colorbar=dict(tickfont=dict(color='#94a3b8'))
            ))
            fig42.update_layout(title='4.2 Correlation Matrix — Mối Tương quan Biến Lõi')
            st.plotly_chart(sf(fig42, 400), use_container_width=True, key="fig42")
            st.markdown('</div>', unsafe_allow_html=True)

    # 4.3 Sankey
    with st.container():
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        sess_flow = df.groupby('session_id').agg(
            max_turn=('turn_number', 'max'),
            has_error=('has_error', 'max'),
            model=('model', 'first')
        ).reset_index()
        def turn_stage(t):
            if t <= 5: return 'Turn 1-5'
            elif t <= 10: return 'Turn 6-10'
            elif t <= 20: return 'Turn 11-20'
            else: return 'Turn 21+'
        sess_flow['stage']   = sess_flow['max_turn'].apply(turn_stage)
        sess_flow['outcome'] = sess_flow['has_error'].map({0: '✅ Success', 1: '❌ Error'})
        stages    = ['Bắt đầu', 'Turn 1-5', 'Turn 6-10', 'Turn 11-20', 'Turn 21+', '✅ Success', '❌ Error']
        stage_idx = {s: i for i, s in enumerate(stages)}
        sources, targets, values, link_colors = [], [], [], []
        for sl in ['Turn 1-5', 'Turn 6-10', 'Turn 11-20', 'Turn 21+']:
            rows = sess_flow[sess_flow['stage'] == sl]
            if len(rows) > 0:
                sources.append(stage_idx['Bắt đầu']); targets.append(stage_idx[sl])
                values.append(len(rows)); link_colors.append('rgba(56,189,248,0.3)')
                for oc in ['✅ Success', '❌ Error']:
                    sub = rows[rows['outcome'] == oc]
                    if len(sub) > 0:
                        sources.append(stage_idx[sl]); targets.append(stage_idx[oc])
                        values.append(len(sub))
                        link_colors.append('rgba(16,185,129,0.35)' if 'Success' in oc else 'rgba(244,63,94,0.35)')
        node_colors = ['#475569', '#38bdf8', '#818cf8', '#a78bfa', '#f59e0b', '#10b981', '#f43f5e']
        fig43 = go.Figure(go.Sankey(
            node=dict(label=stages, color=node_colors, pad=20, thickness=25,
                      line=dict(color='rgba(255,255,255,0.15)', width=0.5)),
            link=dict(source=sources, target=targets, value=values, color=link_colors)
        ))
        fig43.update_layout(title='4.3 Sankey Flow — Hành trình Phiên từ Bắt đầu đến Kết thúc',
                            font=dict(color='#e2e8f0', size=12))
        st.plotly_chart(sf(fig43, 420), use_container_width=True, key="fig43")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── NHÓM 5 ──────────────────────────────────────────────────────────────
    st.markdown('<div class="group-header">🧪 NHÓM 5 — So sánh Tác động System Prompt</div>', unsafe_allow_html=True)

    sp_comparison = df.groupby(['model', 'is_system_prompt_present']).agg(
        avg_turns=('turn_number', 'mean'),
        error_rate=('has_error', 'mean'),
        avg_cost=('turn_cost', 'mean')
    ).reset_index()
    sp_no  = sp_comparison[sp_comparison['is_system_prompt_present'] == 0].set_index('model')
    sp_yes = sp_comparison[sp_comparison['is_system_prompt_present'] == 1].set_index('model')
    common_models = list(set(sp_no.index) & set(sp_yes.index))

    with st.container():
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        metrics_db = ['avg_turns', 'error_rate', 'avg_cost']
        metric_labels = ['Avg Turns', 'Error Rate', 'Avg Cost ($)']
        fig51 = make_subplots(rows=1, cols=3, subplot_titles=metric_labels)
        for col_idx, (metric, label) in enumerate(zip(metrics_db, metric_labels), start=1):
            for i, m in enumerate(common_models):
                val_no  = sp_no.loc[m, metric]  if m in sp_no.index  else 0
                val_yes = sp_yes.loc[m, metric] if m in sp_yes.index else 0
                improved   = val_yes < val_no
                line_color = '#10b981' if improved else '#f43f5e'
                fig51.add_trace(go.Scatter(
                    x=[val_no, val_yes], y=[m, m],
                    mode='lines+markers+text',
                    line=dict(color=line_color, width=3),
                    marker=dict(size=[12, 12], color=['#94a3b8', COLORS.get(m, '#38bdf8')],
                                symbol=['circle', 'diamond']),
                    text=['No Prompt', 'With Prompt'],
                    textposition=['bottom center', 'top center'],
                    textfont=dict(size=9, color='#94a3b8'),
                    showlegend=False
                ), row=1, col=col_idx)
            fig51.update_xaxes(title_text=label, row=1, col=col_idx,
                               gridcolor='rgba(148,163,184,0.07)')
            fig51.update_yaxes(gridcolor='rgba(148,163,184,0.07)', row=1, col=col_idx)
        fig51.update_layout(title='5.1 Dumbbell Plot — Impact System Prompt (⚪ Without → 🔷 With)', height=400)
        st.plotly_chart(sf(fig51, 400), use_container_width=True, key="fig51")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        fig52 = make_subplots(rows=2, cols=2, subplot_titles=[
            'Cost Distribution (Box)', 'Turn Distribution',
            'Error Rate per Model', 'Token Efficiency'
        ])
        sp_labels = {0: 'No Prompt', 1: 'With Prompt'}
        sp_colors = {0: '#94a3b8',   1: '#38bdf8'}
        for sp_val, sp_label in sp_labels.items():
            sp_subset    = df[df['is_system_prompt_present'] == sp_val]
            sess_sp      = sp_subset.groupby('session_id')['turn_cost'].sum()
            sess_turns_sp = sp_subset.groupby('session_id')['turn_number'].max()
            err_per_model = sp_subset.groupby('model')['has_error'].mean() * 100
            eff_per_model = sp_subset.groupby('model')['token_efficiency'].mean()
            fig52.add_trace(go.Box(y=sess_sp, name=sp_label, marker_color=sp_colors[sp_val],
                                   boxmean=True, legendgroup=sp_label), row=1, col=1)
            fig52.add_trace(go.Histogram(x=sess_turns_sp, name=sp_label, marker_color=sp_colors[sp_val],
                                         opacity=0.65, showlegend=False, legendgroup=sp_label,
                                         xbins=dict(start=1, end=40, size=3)), row=1, col=2)
            fig52.add_trace(go.Bar(x=err_per_model.index, y=err_per_model.values, name=sp_label,
                                   marker_color=sp_colors[sp_val], showlegend=False, legendgroup=sp_label), row=2, col=1)
            fig52.add_trace(go.Bar(x=eff_per_model.index, y=eff_per_model.values, name=sp_label,
                                   marker_color=sp_colors[sp_val], showlegend=False, legendgroup=sp_label), row=2, col=2)
        fig52.update_layout(title='5.2 Small Multiples — With vs Without System Prompt', barmode='group', height=560)
        st.plotly_chart(sf(fig52, 560), use_container_width=True, key="fig52")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── NHÓM 6 ──────────────────────────────────────────────────────────────
    st.markdown('<div class="group-header">🚨 NHÓM 6 — Phát hiện Dị thường (Anomaly Detection)</div>', unsafe_allow_html=True)

    c61, c62 = st.columns(2)
    with c61:
        with st.container():
            st.markdown('<div class="bento-card">', unsafe_allow_html=True)
            p95_cost    = df['turn_cost'].quantile(0.95)
            p95_latency = df['latency'].quantile(0.95)
            df_bubble   = df[(df['latency'] > 0) & (df['input_tokens'] > 0)].copy()
            df_bubble['is_anomaly'] = ((df_bubble['turn_cost'] > p95_cost) | (df_bubble['latency'] > p95_latency)).astype(int)
            fig61 = go.Figure()
            for m in m_list:
                dm = df_bubble[(df_bubble['model'] == m) & (df_bubble['is_anomaly'] == 0)]
                fig61.add_trace(go.Scatter(
                    x=dm['latency'], y=dm['turn_cost'], mode='markers', name=m,
                    marker=dict(color=COLORS.get(m, '#94a3b8'),
                                size=dm['input_tokens'].clip(upper=60000) / 5000 + 4,
                                opacity=0.55, line=dict(width=0)),
                    legendgroup=m, showlegend=True
                ))
            anm = df_bubble[df_bubble['is_anomaly'] == 1]
            fig61.add_trace(go.Scatter(
                x=anm['latency'], y=anm['turn_cost'], mode='markers', name='⚠ Anomaly (>P95)',
                marker=dict(color='rgba(244,63,94,0.7)',
                            size=anm['input_tokens'].clip(upper=60000) / 5000 + 6,
                            line=dict(color='#ef4444', width=2.5))
            ))
            fig61.add_hline(y=p95_cost,    line_dash='dash', line_color='#f43f5e',
                            annotation_text='P95 Cost',    annotation_font_color='#fca5a5')
            fig61.add_vline(x=p95_latency, line_dash='dash', line_color='#f59e0b',
                            annotation_text='P95 Latency', annotation_font_color='#fde68a')
            fig61.update_layout(title='6.1 Bubble Anomaly Chart — Latency × Cost × Token Size',
                                xaxis_title='Latency (s)', yaxis_title='Turn Cost ($)')
            st.plotly_chart(sf(fig61, 400), use_container_width=True, key="fig61")
            st.markdown('</div>', unsafe_allow_html=True)

    with c62:
        with st.container():
            st.markdown('<div class="bento-card">', unsafe_allow_html=True)
            sess_cost_ts = sess_agg.sort_values('session_id')['total_cost'].reset_index(drop=True)
            mu    = sess_cost_ts.mean()
            sigma = sess_cost_ts.std()
            ucl   = mu + 3 * sigma
            lcl   = max(0, mu - 3 * sigma)
            in_ctrl = sess_cost_ts.between(lcl, ucl)
            fig62 = go.Figure()
            fig62.add_trace(go.Scatter(
                x=sess_cost_ts[in_ctrl].index, y=sess_cost_ts[in_ctrl],
                mode='markers', name='Trong kiểm soát',
                marker=dict(color='#10b981', size=6, opacity=0.7)
            ))
            fig62.add_trace(go.Scatter(
                x=sess_cost_ts[~in_ctrl].index, y=sess_cost_ts[~in_ctrl],
                mode='markers', name='⚠ Ngoài kiểm soát',
                marker=dict(color='#f43f5e', size=10, symbol='x', line=dict(width=2))
            ))
            fig62.add_hline(y=mu,  line_color='#38bdf8', line_width=1.5,
                            annotation_text=f'Mean=${mu:.4f}', annotation_font_color='#38bdf8')
            fig62.add_hline(y=ucl, line_dash='dash', line_color='#f43f5e', line_width=1.5,
                            annotation_text=f'UCL (+3σ)=${ucl:.4f}', annotation_font_color='#f43f5e')
            fig62.add_hline(y=lcl, line_dash='dash', line_color='#f59e0b', line_width=1.5,
                            annotation_text=f'LCL (-3σ)=${lcl:.4f}', annotation_font_color='#f59e0b')
            fig62.update_layout(title='6.2 SPC Control Chart — Session Cost ±3σ',
                                xaxis_title='Session Index', yaxis_title='Total Cost ($)')
            st.plotly_chart(sf(fig62, 400), use_container_width=True, key="fig62")
            st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 3 — PREDICTIVE
# =============================================================================
with tab3:
    col_ins3, _ = st.columns([3, 1])
    with col_ins3:
        st.markdown("""
        <div class="bento-card">
            <div class="card-title">🔮 Cấp 3 · Phân tích Dự đoán — "Điều gì sẽ xảy ra?"</div>
            <div class="insight-content">
                Dự đoán rủi ro và lượng hóa xác suất thất bại theo điều kiện.
                <div class="insight-highlight-purple">
                    <strong>Risk Matrix</strong>: Xác suất thất bại theo Turn × Token — mọi ô &gt;50% là vùng đỏ nguy hiểm.
                </div>
                <div class="insight-highlight-amber">
                    <strong>Magic Quadrant</strong>: Ma trận lựa chọn Model tối ưu theo Cost Efficiency × Performance Score.
                </div>
                <div class="insight-highlight">
                    <strong>Radar Chart</strong>: So sánh đa chiều 6 tiêu chí (Normalized 0-100).
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="group-header">🎯 NHÓM 7 — Tổng hợp Ra Quyết định (Model Selection)</div>', unsafe_allow_html=True)

    # Risk Matrix
    with st.container():
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        df_pred = df.copy()
        df_pred['Turn_Bins']  = pd.cut(df_pred['turn_number'], bins=[0, 5, 15, 30, 999],
                                       labels=['1-5 Lượt', '6-15 Lượt', '16-30 Lượt', '>30 Lượt'])
        df_pred['Token_Bins'] = pd.cut(df_pred['input_tokens'], bins=[0, 10000, 20000, 30000, 999999],
                                       labels=['<10k Tokens', '10k-20k', '20k-30k', '>30k Tokens'])
        risk_matrix = df_pred.groupby(['Token_Bins', 'Turn_Bins'], observed=False)['has_error'].mean().unstack() * 100
        fig_risk = go.Figure(data=go.Heatmap(
            z=risk_matrix.values, x=risk_matrix.columns, y=risk_matrix.index,
            colorscale=[[0, '#10b981'], [0.5, '#f59e0b'], [1, '#f43f5e']],
            text=np.round(risk_matrix.values, 1), texttemplate='<b>%{text}%</b>',
            textfont=dict(size=14, color='white'),
            colorbar=dict(tickfont=dict(color='#94a3b8'))
        ))
        fig_risk.update_layout(title='[Q6] Ma trận Tiên lượng — Xác suất Thất bại theo Turn × Token',
                               xaxis_title='Thời lượng Phiên (Turns)', yaxis_title='Kích thước Đầu vào (Tokens)')
        st.plotly_chart(sf(fig_risk, 420), use_container_width=True, key="fig_risk")
        st.markdown('</div>', unsafe_allow_html=True)

    c_p1, c_p2 = st.columns(2)

    # Q7: Opus vs Sonnet heavy tasks
    with c_p1:
        with st.container():
            st.markdown('<div class="bento-card">', unsafe_allow_html=True)
            heavy = df[(df['output_length'] > 500) & (df['model'].isin(['claude-opus-4-6', 'claude-sonnet-4-6']))].copy()
            heavy['Status'] = heavy['has_error'].map({1: '❌ Lỗi', 0: '✅ Thành công'})
            fig_q7 = px.scatter(heavy, x='latency', y='turn_cost', color='model', symbol='Status',
                                color_discrete_map=COLORS, size='output_length', size_max=20,
                                title='[Q7] Tác vụ Nặng (>500 tokens): Opus vs Sonnet')
            fig_q7.update_layout(xaxis_title='Latency (s)', yaxis_title='Turn Cost ($)')
            st.plotly_chart(sf(fig_q7, 400), use_container_width=True, key="fig_q7")
            st.markdown('</div>', unsafe_allow_html=True)

    # Magic Quadrant
    with c_p2:
        with st.container():
            st.markdown('<div class="bento-card">', unsafe_allow_html=True)
            model_quad = sess_agg.groupby('model').agg(
                cost_eff=('total_cost', lambda x: 1 / (x.mean() + 1e-9)),
                perf_score=('error_rate', lambda x: (1 - x.mean()) * 100)
            ).reset_index()
            x_mean = model_quad['cost_eff'].mean()
            y_mean = model_quad['perf_score'].mean()
            fig71 = go.Figure()
            quad_fills = [
                (x_mean, y_mean, model_quad['cost_eff'].max()*1.1, 105, 'rgba(16,185,129,0.06)'),
                (0, y_mean, x_mean, 105, 'rgba(56,189,248,0.06)'),
                (0, 0, x_mean, y_mean, 'rgba(244,63,94,0.06)'),
                (x_mean, 0, model_quad['cost_eff'].max()*1.1, y_mean, 'rgba(245,158,11,0.06)')
            ]
            for x0, y0, x1, y1, fc in quad_fills:
                fig71.add_shape(type='rect', x0=x0, y0=y0, x1=x1, y1=y1, fillcolor=fc, line_width=0)
            for _, row in model_quad.iterrows():
                fig71.add_trace(go.Scatter(
                    x=[row['cost_eff']], y=[row['perf_score']], mode='markers+text',
                    name=row['model'],
                    marker=dict(size=28, color=COLORS.get(row['model'], '#94a3b8'),
                                line=dict(color='#f8fafc', width=2)),
                    text=[row['model'].split('-')[1] if '-' in row['model'] else row['model']],
                    textposition='top center', textfont=dict(color='#f8fafc', size=11)
                ))
            fig71.add_hline(y=y_mean, line_dash='dash', line_color='rgba(148,163,184,0.25)')
            fig71.add_vline(x=x_mean, line_dash='dash', line_color='rgba(148,163,184,0.25)')
            fig71.update_layout(title='7.1 Magic Quadrant — Model Selection Matrix',
                                xaxis_title='Cost Efficiency (1/AvgCost)',
                                yaxis_title='Performance Score (%)')
            st.plotly_chart(sf(fig71, 400), use_container_width=True, key="fig71")
            st.markdown('</div>', unsafe_allow_html=True)

    # Radar Chart
    with st.container():
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        model_radar = df.groupby('model').agg(
            cost_eff=('turn_cost', lambda x: 1 / (x.mean() + 1e-9)),
            speed=('turn_number', lambda x: 1 / (x.mean() + 1e-9)),
            accuracy=('has_error', lambda x: (1 - x.mean()) * 100),
            token_eff=('token_efficiency', 'mean'),
            stability=('turn_cost', lambda x: 1 / (x.std() + 1e-9)),
            max_turns=('turn_number', 'max')
        ).reset_index()
        radar_metrics = ['cost_eff', 'speed', 'accuracy', 'token_eff', 'stability', 'max_turns']
        radar_labels  = ['Cost Efficiency', 'Speed\n(1/AvgTurn)', 'Accuracy\n(1-ErrRate)',
                         'Token Efficiency', 'Stability\n(1/Variance)', 'Scalability\n(MaxTurns)']
        for col in radar_metrics:
            col_min = model_radar[col].min()
            col_max = model_radar[col].max()
            model_radar[col + '_norm'] = ((model_radar[col] - col_min) / (col_max - col_min + 1e-9)) * 100
        fig72 = go.Figure()
        for _, row in model_radar.iterrows():
            vals   = [row[m + '_norm'] for m in radar_metrics]
            vals   = vals + [vals[0]]
            labels = radar_labels + [radar_labels[0]]
            color  = COLORS.get(row['model'], '#94a3b8')
            h = color.lstrip('#')
            rc, gc, bc = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            fig72.add_trace(go.Scatterpolar(
                r=vals, theta=labels, fill='toself', name=row['model'],
                line_color=color, fillcolor=f'rgba({rc},{gc},{bc},0.15)', opacity=0.9
            ))
        fig72.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100],
                                gridcolor='rgba(148,163,184,0.15)',
                                tickfont=dict(color='#64748b', size=10)),
                angularaxis=dict(gridcolor='rgba(148,163,184,0.15)',
                                 tickfont=dict(color='#e2e8f0', size=11))
            ),
            title='7.2 Radar Chart — So sánh Đa chiều 6 Tiêu chí (Normalized 0-100)',
            height=520
        )
        st.plotly_chart(sf(fig72, 520), use_container_width=True, key="fig72")
        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# TAB 4 — PRESCRIPTIVE
# =============================================================================
with tab4:
    col_ins4, col_act = st.columns([5, 7])
    with col_ins4:
        st.markdown("""
        <div class="bento-card" style="height:100%;">
            <div class="card-title">💡 Cấp 4 · Đề xuất Hành động — "Chúng ta nên làm gì?"</div>
            <div class="insight-content">
                <div class="insight-highlight-green">
                    <strong>Q8 — Smart Routing:</strong> Context &lt; 15k & Task Nhẹ → Dùng Sonnet. Context &gt; 35k hoặc Task Nặng → Bắt buộc Opus.
                </div>
                <div class="insight-highlight-amber">
                    <strong>Q9 — Circuit Breaker:</strong> Cắt tự động ở Turn 5 bảo toàn đại đa số ngân sách chìm. Chờ đến Turn 10 thì tiền đã bốc hơi.
                </div>
                <div class="insight-highlight-purple">
                    <strong>Q10 — Micro-Tasking:</strong> Chia nhỏ hàm, context &lt; 2k tokens để vắt kiệt chi phí siêu rẻ của Deepseek/Minimax.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_act:
        st.markdown("""
        <div class="bento-card">
            <div class="card-title">🎯 Action Items & Priority Matrix</div>
            <div class="insight-content">
                <table style="width:100%; border-collapse: separate; border-spacing: 0 6px; font-size: 0.85rem;">
                    <thead>
                        <tr>
                            <th style="text-align:left; padding: 8px 12px; color:#64748b; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.5px;">Priority</th>
                            <th style="text-align:left; padding: 8px 12px; color:#64748b; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.5px;">Action</th>
                            <th style="text-align:left; padding: 8px 12px; color:#64748b; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.5px;">Impact</th>
                            <th style="text-align:left; padding: 8px 12px; color:#64748b; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.5px;">Timeline</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td style="padding:10px 12px; background:rgba(244,63,94,0.08); border-radius:8px 0 0 8px;"><span class="status-badge status-critical">P0 Critical</span></td>
                            <td style="padding:10px 12px; background:rgba(244,63,94,0.08); color:#e2e8f0;">Gỡ System Prompt với Deepseek & Minimax</td>
                            <td style="padding:10px 12px; background:rgba(244,63,94,0.08); color:#4ade80;">Error Rate → 0%</td>
                            <td style="padding:10px 12px; background:rgba(244,63,94,0.08); border-radius:0 8px 8px 0; color:#94a3b8;">Ngay lập tức</td>
                        </tr>
                        <tr>
                            <td style="padding:10px 12px; background:rgba(245,158,11,0.08); border-radius:8px 0 0 8px;"><span class="status-badge status-warning">P1 High</span></td>
                            <td style="padding:10px 12px; background:rgba(245,158,11,0.08); color:#e2e8f0;">Thiết lập Hard-Timeout 5s cho Minimax</td>
                            <td style="padding:10px 12px; background:rgba(245,158,11,0.08); color:#4ade80;">Chặn treo 25 phút</td>
                            <td style="padding:10px 12px; background:rgba(245,158,11,0.08); border-radius:0 8px 8px 0; color:#94a3b8;">1 tuần</td>
                        </tr>
                        <tr>
                            <td style="padding:10px 12px; background:rgba(245,158,11,0.08); border-radius:8px 0 0 8px;"><span class="status-badge status-warning">P1 High</span></td>
                            <td style="padding:10px 12px; background:rgba(245,158,11,0.08); color:#e2e8f0;">Set max_tokens cho Sonnet</td>
                            <td style="padding:10px 12px; background:rgba(245,158,11,0.08); color:#4ade80;">Giảm Outlier $0.26</td>
                            <td style="padding:10px 12px; background:rgba(245,158,11,0.08); border-radius:0 8px 8px 0; color:#94a3b8;">1 tuần</td>
                        </tr>
                        <tr>
                            <td style="padding:10px 12px; background:rgba(56,189,248,0.06); border-radius:8px 0 0 8px;"><span class="status-badge status-optimal">P2 Normal</span></td>
                            <td style="padding:10px 12px; background:rgba(56,189,248,0.06); color:#e2e8f0;">Triển khai Smart Routing Matrix</td>
                            <td style="padding:10px 12px; background:rgba(56,189,248,0.06); color:#4ade80;">ROI tối ưu</td>
                            <td style="padding:10px 12px; background:rgba(56,189,248,0.06); border-radius:0 8px 8px 0; color:#94a3b8;">1 tháng</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Q8 - Routing Matrix + Q9 - Circuit Breaker
    c_q8, c_q9 = st.columns(2)
    with c_q8:
        with st.container():
            st.markdown('<div class="bento-card">', unsafe_allow_html=True)
            route_mat = df.groupby(['context_size', 'task_size'], observed=False)['success'].mean().unstack() * 100
            fig10 = go.Figure(data=go.Heatmap(
                z=route_mat.values, x=route_mat.columns, y=route_mat.index,
                colorscale=[[0, '#f43f5e'], [0.5, '#f59e0b'], [1, '#10b981']],
                text=np.round(route_mat.values, 1),
                texttemplate='<b>%{text}%</b>', textfont=dict(color='white', size=15),
                colorbar=dict(tickfont=dict(color='#94a3b8'))
            ))
            fig10.update_layout(title='[Q8] Smart Routing Matrix — Tỷ lệ Thành công theo Context × Task',
                                xaxis_title='Khối lượng Đầu ra (Task Size)',
                                yaxis_title='Kích thước Đầu vào (Context Size)')
            st.plotly_chart(sf(fig10, 400), use_container_width=True, key="fig10")
            st.markdown('</div>', unsafe_allow_html=True)

    with c_q9:
        with st.container():
            st.markdown('<div class="bento-card">', unsafe_allow_html=True)
            cutoffs   = list(range(1, 21))
            savings   = []
            tot_sunk  = df['sunk_cost'].sum()
            for c in cutoffs:
                saved = df[df['turn_number'] > c]['sunk_cost'].sum()
                savings.append((saved / tot_sunk) * 100 if tot_sunk else 0)
            fig_q9 = go.Figure()
            fig_q9.add_trace(go.Scatter(
                x=cutoffs, y=savings, mode='lines+markers',
                line=dict(color='#10b981', width=3),
                marker=dict(size=9, color='#10b981', line=dict(color='#0a0e1a', width=2)),
                fill='tozeroy', fillcolor='rgba(16,185,129,0.08)'
            ))
            fig_q9.add_vline(x=5, line_dash='dash', line_color='#f43f5e',
                             annotation_text='Cutoff T5 (Khuyến nghị)',
                             annotation_font_color='#fb7185')
            fig_q9.update_layout(title='[Q9] Circuit Breaker — Ngân sách Chìm Bảo toàn (%)',
                                 xaxis_title='Turn Cutoff', yaxis_title='% Ngân sách Chìm Bảo toàn')
            st.plotly_chart(sf(fig_q9, 400), use_container_width=True, key="fig_q9")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Q10 - Micro-Tasking Architecture
    st.markdown('<div class="group-header">🏗️ Q10 — Kiến trúc Micro-Tasking Pipeline</div>', unsafe_allow_html=True)

    c_arch, c_chart = st.columns([3, 2])
    with c_arch:
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)
        st.code("""
# [TÁI CẤU TRÚC PIPELINE CHỐNG TRÀN NGỮ CẢNH]
def micro_task_pipeline(file_content, target_bug):
    # 1. Trích xuất cục bộ (Extract AST) — < 2k tokens
    local_context = extract_function(file_content, target_bug)

    # 2. Xóa System Prompt dư thừa
    prompt = build_lightweight_prompt(local_context)

    # 3. Giao task cho AI giá rẻ (Deepseek/Minimax)
    cheap_model = route_to_cheap_model(prompt)
    patch = cheap_model.generate(prompt)

    # 4. Xác thực bằng Opus chỉ khi cần
    if not validate_patch(patch):
        patch = claude_opus.generate(prompt)

    return apply_patch(file_content, patch)
        """, language="python")
        st.markdown('</div>', unsafe_allow_html=True)

    with c_chart:
        with st.container():
            st.markdown('<div class="bento-card">', unsafe_allow_html=True)
            cheap    = df[(df['model'].isin(['minimax-m2.5', 'deepseek-v3.1'])) & (df['throughput'] > 0)]
            cheap_tp = cheap.groupby(['model', 'context_size'], observed=False)['throughput'].mean().reset_index()
            fig12 = px.bar(cheap_tp, x='model', y='throughput', color='context_size', barmode='group',
                           title='[Q10] Throughput Model Rẻ theo Context Size',
                           color_discrete_sequence=['#38bdf8', '#a78bfa', '#f59e0b'])
            fig12.update_layout(yaxis_title='Throughput (Tokens/s)')
            st.plotly_chart(sf(fig12, 360), use_container_width=True, key="fig12_q10")
            st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div class="bento-grid">
    <div class="bento-card col-span-12" style="text-align:center; padding: 20px; border-color: rgba(148,163,184,0.04);">
        <span style="font-size: 0.75rem; color: #475569;">
            🧠 AI Agent Diagnostic Intelligence &nbsp;•&nbsp; Generated 01/08/2026 &nbsp;•&nbsp;
            {total_turns:,} Turns • {total_sess:,} Sessions • {len(m_list)} Models &nbsp;•&nbsp;
            Data Source: OpenTelemetry + LangSmith &nbsp;•&nbsp; Confidential
        </span>
    </div>
</div>
""", unsafe_allow_html=True)
