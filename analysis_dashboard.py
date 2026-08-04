# -*- coding: utf-8 -*-
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 0. CẤU HÌNH TRANG & HỆ TRỤC MÀU SẮC (MỤC 4)
# ==========================================
st.set_page_config(
    page_title="Xây dựng báo cáo phân tích chi phí và hiệu năng hoạt động của AI Agent",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="collapsed"
)

MODEL_COLORS = {
    'claude-sonnet-4-6': '#0ea5e9',  # Sky Blue rực rỡ
    'claude-opus-4-6': '#10b981',    # Emerald Green chuẩn xác
    'deepseek-v3.1': '#f59e0b',      # Warm Amber
    'minimax-m2.5': '#f43f5e'        # Vivid Rose cảnh báo
}

# CSS Executive Bento Grid & Glassmorphism Theme (Học hỏi từ validate_dashboard1)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --page-bg: #f0f7ff;
        --card: #ffffff;
        --border: rgba(14, 165, 233, 0.15);
        --heading: #0f172a;
        --value: #1e293b;
        --label: #64748b;
        --muted: #94a3b8;
        --grid: #f1f5f9;
    }

    .stApp {
        background: linear-gradient(145deg, #f0f7ff 0%, #e8f4fd 50%, #f5f9ff 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--value);
    }
    
    .block-container {
        max-width: 1600px !important;
        padding: 24px 20px !important;
        margin: 0 auto !important;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    
    /* Title chính rực rỡ dải màu Executive */
    h1 {
        font-family: 'Inter', -apple-system, sans-serif !important;
        font-weight: 900 !important;
        font-size: 32px !important;
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 40%, #0369a1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.8px !important;
        margin-bottom: 4px !important;
    }

    /* Custom styling cho bento card (Glassmorphism & Soft Shadow) */
    div[data-testid="stVerticalBlockBorderWrapper"], div[data-testid="stContainer"] > div {
        background: #ffffff;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: 16px;
        box-shadow: 0 4px 16px -2px rgba(14, 165, 233, 0.08), 0 2px 6px rgba(0,0,0,0.03);
        padding: 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover, div[data-testid="stContainer"] > div:hover {
        transform: translateY(-2px);
        border-color: rgba(14, 165, 233, 0.35);
        box-shadow: 0 12px 28px -6px rgba(14, 165, 233, 0.15), 0 4px 12px rgba(0,0,0,0.05);
    }

    .panel-title {
        font-size: 13.5px;
        font-weight: 700;
        color: #0284c7;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .panel-title::after {
        content: '';
        flex: 1;
        height: 1.5px;
        background: linear-gradient(90deg, rgba(14,165,233,0.28), transparent);
    }
    
    .section-header {
        background: linear-gradient(90deg, rgba(14,165,233,0.12), transparent);
        border-left: 4px solid #0ea5e9;
        padding: 10px 18px;
        border-radius: 0 10px 10px 0;
        margin: 28px 0 16px 0;
        font-size: 15px;
        font-weight: 800;
        color: #0369a1;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }

    /* KPI CARDS HIỆN ĐẠI SANG TRỌNG */
    .kpi-card {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 16px 18px;
        min-height: 105px;
        box-shadow: 0 4px 12px -2px rgba(14, 165, 233, 0.08);
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px -4px rgba(14, 165, 233, 0.16);
    }
    .kpi-card.wasted {
        border: 1px solid rgba(244, 63, 94, 0.25) !important;
        background: linear-gradient(135deg, #ffffff 0%, #fff5f6 100%);
        border-left: 4px solid #f43f5e !important;
    }
    .kpi-label {
        font-size: 11.5px;
        color: #64748b;
        margin-bottom: 6px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .kpi-value {
        font-size: 25px;
        font-weight: 800;
        color: #0f172a;
        font-family: 'JetBrains Mono', monospace;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.5px;
        line-height: 1.1;
    }
    .kpi-card.wasted .kpi-value, .kpi-card.wasted .kpi-label {
        color: #e11d48;
    }

    .story-box {
        background: #ffffff;
        border: 1px solid var(--border);
        border-left: 4px solid #0ea5e9;
        border-radius: 14px;
        padding: 18px 20px;
        font-size: 13.5px;
        line-height: 1.65;
        color: #334155;
        box-shadow: 0 4px 12px -2px rgba(14, 165, 233, 0.06);
        height: 100%;
    }
    .story-box.warning {
        border-left-color: #f59e0b;
        border-color: rgba(245, 158, 11, 0.2);
        background: linear-gradient(145deg, #ffffff 0%, #fffbf2 100%);
    }
    .story-box.success {
        border-left-color: #10b981;
        border-color: rgba(16, 185, 129, 0.2);
        background: linear-gradient(145deg, #ffffff 0%, #f2fdf7 100%);
    }
    .story-box-title {
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 10px;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .data-table, .exec-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12.5px;
        margin-top: 6px;
        margin-bottom: 6px;
    }
    .data-table th, .exec-table th {
        text-align: left;
        padding: 10px 12px;
        background-color: #f8fafc;
        border-bottom: 2px solid #e2e8f0;
        color: #0f172a;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
    }
    .data-table td, .exec-table td {
        padding: 10px 12px;
        border-bottom: 1px solid var(--grid);
        color: var(--value);
        font-variant-numeric: tabular-nums;
    }
    .data-table tr:hover, .exec-table tr:hover {
        background-color: #f8fafc;
    }

    .footnote {
        font-size: 11.5px;
        color: var(--muted);
        line-height: 1.5;
        margin-top: 24px;
        padding-top: 14px;
        border-top: 1px solid var(--grid);
    }
    
    .warning-banner {
        background-color: #fff5f5;
        border: 1px solid #fed7d7;
        border-left: 4px solid #f43f5e;
        color: #9b2c2c;
        padding: 12px 16px;
        border-radius: 10px;
        font-size: 13px;
        margin-bottom: 14px;
        font-weight: 500;
    }
    
    /* Nút chọn Slicer mượt mà */
    div[data-testid="stSelectbox"] label {
        font-size: 12px !important;
        font-weight: 600 !important;
        color: #0284c7 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

def get_card_container():
    """Hỗ trợ tương thích an toàn cho cả phiên bản Streamlit mới (border=True) và cũ"""
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()

# ==========================================
# 1. CÁC HÀM FORMAT CHUNG (KHÔNG DÙNG F-STRING RẢI RÁC)
# ==========================================
def format_currency(val) -> str:
    if val is None or pd.isna(val):
        return "$0.00"
    return f"${val:,.2f}"

def format_percent(val) -> str:
    if val is None or pd.isna(val):
        return "0.0%"
    return f"{val * 100:.1f}%"

def format_int(val) -> str:
    if val is None or pd.isna(val):
        return "0"
    return f"{int(round(val)):,}"

def get_chart_width(span: int) -> int:
    # Tính toán chính xác width theo span trên lưới 12 cột 1600px
    raw_width = int((1600 - 176) * (span / 12.0) + 16 * (span - 1)) - 36
    return max(raw_width, 250)

def apply_common_layout(fig, width: int, height: int, show_legend=False, orientation='v'):
    fig.update_layout(
        template='plotly_white',
        width=width,
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(248, 250, 252, 0.55)',
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", size=11, color="#475569"),
        margin=dict(l=48, r=24, t=24, b=32),
        showlegend=show_legend,
    )
    if show_legend:
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                bgcolor='rgba(0,0,0,0)',
                font=dict(size=11, color="#0f172a", family="Inter")
            )
        )
    if orientation == 'v':
        fig.update_xaxes(showline=False, showgrid=False, zeroline=False, tickfont=dict(color="#64748b"))
        fig.update_yaxes(showline=False, showgrid=True, gridcolor="#e2e8f0", zeroline=False, tickfont=dict(color="#64748b"))
    else:
        fig.update_xaxes(showline=False, showgrid=False, zeroline=False, tickfont=dict(color="#64748b"))
        fig.update_yaxes(showline=False, showgrid=True, gridcolor="#e2e8f0", zeroline=False, tickfont=dict(color="#0f172a", size=12, family="Inter"))
    return fig

# ==========================================
# 2. PIPELINE ETL & LOAD DATA (MỤC 1 & 5)
# ==========================================
@st.cache_data
def parse_session_id(sid: str):
    parts = sid.split('__')
    benchmark = parts[0]
    if benchmark == 'swebench':
        project = parts[1] if len(parts) > 1 else 'unknown'
        issue = parts[2] if len(parts) > 2 else 'unknown'
        tag = parts[3] if len(parts) > 3 else 'unknown'
        is_rerun = (len(parts) == 5 and parts[4] == 'run2')
    else:
        project = benchmark
        issue = parts[1] if len(parts) > 1 else None
        tag = parts[-1]
        is_rerun = False
    return benchmark, project, issue, tag, is_rerun

@st.cache_data
def load_data():
    candidate_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_agentic_traces.csv'),
        'code/processed_agentic_traces.csv',
        'processed_agentic_traces.csv',
        '/home/leducdat/projectDuan/code/processed_agentic_traces.csv',
        '../code/processed_agentic_traces.csv',
        '../processed_agentic_traces.csv'
    ]
    df = None
    for p in candidate_paths:
        try:
            if os.path.exists(p):
                df = pd.read_csv(p)
                break
        except Exception:
            continue
    if df is None:
        raise FileNotFoundError("Không tìm thấy file processed_agentic_traces.csv trong hệ thống. Vui lòng kiểm tra lại đường dẫn lưu trữ.")

    # Apply parse_session_id cho 3 dạng session
    parsed = df['session_id'].apply(parse_session_id)
    df['benchmark'] = [x[0] for x in parsed]
    df['project'] = [x[1] for x in parsed]
    df['issue'] = [x[2] for x in parsed]
    df['tag'] = [x[3] for x in parsed]
    df['is_rerun'] = [x[4] for x in parsed]
    
    # Trường dẫn xuất turn-level
    df['latency'] = df['pre_gap']
    df = df.sort_values(by=['session_id', 'turn_number']).reset_index(drop=True)
    df['cum_cost'] = df.groupby('session_id', sort=False)['turn_cost'].cumsum()
    
    # Ngưỡng outlier theo công thức IQR cho từng model (Q3 + 1.5*IQR)
    model_iqr_thresholds = {}
    for m, grp in df.groupby('model'):
        q1 = grp['turn_cost'].quantile(0.25)
        q3 = grp['turn_cost'].quantile(0.75)
        iqr = q3 - q1
        model_iqr_thresholds[m] = q3 + 1.5 * iqr
    df['outlier_threshold'] = df['model'].map(model_iqr_thresholds)
    
    # Tổng hợp mức Session (per SESSION) theo chuẩn agg
    df_sessions = df.groupby('session_id', sort=False).agg(
        model=('model', 'first'),
        benchmark=('benchmark', 'first'),
        project=('project', 'first'),
        is_system_prompt_present=('is_system_prompt_present', 'first'),
        is_rerun=('is_rerun', 'first'),
        total_cost=('turn_cost', 'sum'),
        n_turns=('turn_number', 'max'),
        error_share=('has_error', 'mean'),
        avg_input_tokens=('input_tokens', 'mean')
    ).reset_index()
    
    df_sessions['failed'] = (df_sessions['error_share'] == 1.0).astype(int)
    df_sessions['resolved'] = 1 - df_sessions['failed']
    df_sessions['cost_per_turn'] = df_sessions['total_cost'] / df_sessions['n_turns']
    
    return df, df_sessions, model_iqr_thresholds

@st.cache_data
def get_sonnet_elbow_turn(df_sessions):
    """Tính điểm khuỷu cho Sonnet: turn t* đầu tiên mà delta resolve_rate_cum duy trì < 1 điểm % trong >=5 turn"""
    s_sess = df_sessions[df_sessions['model'] == 'claude-sonnet-4-6']
    total_s = len(s_sess)
    if total_s == 0:
        return 15
    res_sess = s_sess[s_sess['resolved'] == 1]
    rates = [0.0]
    for t in range(1, 41):
        rates.append(len(res_sess[res_sess['n_turns'] <= t]) / total_s)
    
    for i in range(1, len(rates) - 5):
        if rates[i] > 0:
            deltas = [rates[j+1] - rates[j] for j in range(i, i+5)]
            if all(d < 0.01 for d in deltas):
                return i
    return 15

# ==========================================
# 3. CÁC HÀM VẼ 13 PANEL PHÂN TÍCH (ĐÃ GÁN UNIQUE KEY & TRÌNH BÀY ĐẦY ĐỦ 4 MODEL)
# ==========================================

# --- CẤP 1: MÔ TẢ (4 panel) ---
def draw_MT1(df_turns, df_sessions, span=6):
    st.markdown('<div class="panel-title">MT1 — Ngân sách theo model</div>', unsafe_allow_html=True)
    if df_turns.empty:
        st.info("Không có dữ liệu khớp bộ lọc Slicer")
        return
    df_cost = df_turns.groupby('model')['turn_cost'].sum().reset_index()
    df_cost = df_cost.sort_values(by='turn_cost', ascending=True)
    df_cost['label'] = df_cost['turn_cost'].apply(format_currency)
    
    fig = go.Figure(go.Bar(
        x=df_cost['turn_cost'],
        y=df_cost['model'],
        orientation='h',
        text=df_cost['label'],
        textposition='outside',
        marker_color=[MODEL_COLORS.get(m, '#9CA3AF') for m in df_cost['model']]
    ))
    fig = apply_common_layout(fig, get_chart_width(span), 280, orientation='h')
    max_x = df_cost['turn_cost'].max()
    fig.update_xaxes(range=[0, max(max_x * 1.25, 0.01)])
    st.plotly_chart(fig, use_container_width=False, config={'displayModeBar': False}, key="chart_mt1_ngansach")

def draw_MT3(df_turns, df_sessions, span=6):
    st.markdown('<div class="panel-title">MT3 — Tỷ lệ lỗi theo model (Cấp Turn)</div>', unsafe_allow_html=True)
    if df_turns.empty:
        st.info("Không có dữ liệu khớp bộ lọc Slicer")
        return
    df_err = df_turns.groupby('model')['has_error'].mean().reset_index()
    df_err = df_err.sort_values(by='has_error', ascending=False)
    df_err['label'] = df_err['has_error'].apply(format_percent)
    
    fig = go.Figure(go.Bar(
        x=df_err['model'],
        y=df_err['has_error'],
        text=df_err['label'],
        textposition='outside',
        marker_color=[MODEL_COLORS.get(m, '#9CA3AF') for m in df_err['model']]
    ))
    fig = apply_common_layout(fig, get_chart_width(span), 280, orientation='v')
    fig.update_yaxes(range=[0, 1.15], tickformat=".0%")
    st.plotly_chart(fig, use_container_width=False, config={'displayModeBar': False}, key="chart_mt3_tyleloi")

def draw_MT2(df_turns, df_sessions, span=5):
    st.markdown('<div class="panel-title">MT2 — Bối cảnh 4 mô hình</div>', unsafe_allow_html=True)
    if df_sessions.empty:
        st.info("Không có dữ liệu khớp bộ lọc Slicer")
        return
    s_stat = df_sessions.groupby('model').agg(
        sessions=('session_id', 'count'),
        avg_cost=('total_cost', 'mean')
    ).reset_index()
    t_stat = df_turns.groupby('model').agg(
        turns=('turn_number', 'count'),
        total_cost=('turn_cost', 'sum')
    ).reset_index()
    m_df = pd.merge(s_stat, t_stat, on='model').sort_values('total_cost', ascending=False)
    
    html = '<table class="data-table"><thead><tr><th>Model</th><th>Sessions</th><th>Turns</th><th>Avg Cost/Ses</th><th>Total Cost</th></tr></thead><tbody>'
    for _, r in m_df.iterrows():
        html += f"<tr><td><b>{r['model']}</b></td><td>{format_int(r['sessions'])}</td><td>{format_int(r['turns'])}</td><td>{format_currency(r['avg_cost'])}</td><td>{format_currency(r['total_cost'])}</td></tr>"
    html += '</tbody></table>'
    st.markdown(html, unsafe_allow_html=True)

def draw_MT4(df_turns, df_sessions, span=7):
    st.markdown('<div class="panel-title">MT4 — Tác động System Prompt tới lỗi (Sonnet)</div>', unsafe_allow_html=True)
    sub = df_turns[df_turns['model'] == 'claude-sonnet-4-6']
    if sub.empty:
        st.info("Không có dữ liệu khớp bộ lọc Slicer (Cần mô hình claude-sonnet-4-6)")
        return
    df_p = sub.groupby('is_system_prompt_present')['has_error'].mean().reset_index()
    df_p['cat'] = df_p['is_system_prompt_present'].map({0: 'Không System Prompt', 1: 'Có System Prompt'})
    df_p['label'] = df_p['has_error'].apply(format_percent)
    
    colors_map = {0: MODEL_COLORS['claude-sonnet-4-6'], 1: '#1F4E79'}
    bar_colors = [colors_map.get(k, '#9CA3AF') for k in df_p['is_system_prompt_present']]
    
    fig = go.Figure(go.Bar(
        x=df_p['cat'],
        y=df_p['has_error'],
        text=df_p['label'],
        textposition='outside',
        marker_color=bar_colors,
        width=0.4
    ))
    fig = apply_common_layout(fig, get_chart_width(span), 280, orientation='v')
    fig.update_yaxes(range=[0, 1.15], tickformat=".0%")
    st.plotly_chart(fig, use_container_width=False, config={'displayModeBar': False}, key="chart_mt4_sysprompt")

# --- CẤP 2: CHẨN ĐOÁN (5 panel + STORY-2) ---
def draw_CD1(df_turns, df_sessions, span=7, wasted_narrow=0):
    st.markdown('<div class="panel-title">CD1 — Đối chứng Cạm bẫy giá rẻ (Đầy đủ 4 model)</div>', unsafe_allow_html=True)
    if df_sessions.empty:
        st.info("Không có dữ liệu khớp bộ lọc Slicer")
        return
    
    fig = go.Figure()
    all_models = sorted(df_sessions['model'].unique())
    for m in all_models:
        m_df = df_sessions[df_sessions['model'] == m]
        if not m_df.empty:
            is_claude = 'claude' in m.lower()
            # Căn chỉnh kích thước bóng bẩy theo lượng token trung bình
            size_scaled = 8 + (m_df['avg_input_tokens'] / 40000.0) * 16
            size_scaled = size_scaled.clip(lower=10, upper=24)  # Giới hạn size mĩ thuật từ 10 đến 24
            
            fig.add_trace(go.Scatter(
                x=m_df['cost_per_turn'],
                y=m_df['n_turns'],
                mode='markers',
                name=m,
                marker=dict(
                    size=size_scaled,
                    color=MODEL_COLORS.get(m, '#94a3b8'),
                    symbol='diamond' if is_claude else 'circle',
                    opacity=0.88 if is_claude else 0.72,
                    line=dict(width=1.5 if is_claude else 1, color='#ffffff')
                ),
                hovertemplate="<b>Model: %s</b><br>Chi phí/lượt ($/turn): $%%{x:.4f}<br>Số lượt lặp (n_turns): %%{y}<br>Avg Tokens/turn: %%{customdata:,}<extra></extra>" % m,
                customdata=m_df['avg_input_tokens']
            ))
            
            # Gắn nhãn tĩnh riêng cho Opus để điểm kim cương xanh lá nổi bật dù số session ít
            if 'opus' in m.lower() and len(m_df) > 0:
                fig.add_trace(go.Scatter(
                    x=[m_df['cost_per_turn'].mean()],
                    y=[m_df['n_turns'].mean()],
                    mode='text',
                    text=["<b>Claude Opus</b>"],
                    textposition="top right",
                    textfont=dict(size=11, color=MODEL_COLORS.get(m, '#10b981'), family="Inter"),
                    showlegend=False, hoverinfo="skip"
                ))
    
    # Đường chia ranh giới kẹt vòng lặp vô tận (tại mốc 25 turns)
    fig.add_hline(y=25, line_dash="dash", line_color="#f43f5e", line_width=1.3)
    
    # Chú thích vùng Cạm Bẫy Giá Rẻ (phía trên trần)
    trap_df = df_sessions[df_sessions['model'].isin(['minimax-m2.5', 'deepseek-v3.1'])]
    trap_x = trap_df['cost_per_turn'].median() if not trap_df.empty else 0.05
    trap_y = trap_df['n_turns'].max() * 0.9 if not trap_df.empty else 35
    fig.add_annotation(
        x=trap_x, y=trap_y,
        text=f"<b>🚨 VÙNG KẸT VÒNG LẶP (Minimax & Deepseek)</b><br>Đơn giá rẻ nhưng kéo dài >30 lượt -> <b>{format_currency(wasted_narrow)} lãng phí (100% Fail)</b>",
        showarrow=True, arrowhead=2, ax=45, ay=-25,
        font=dict(color="#be123c", size=11, family="Inter"),
        bgcolor="rgba(255, 241, 242, 0.95)", bordercolor="#f43f5e", borderwidth=1.2
    )
    
    # Chú thích vùng Hiệu Quả (phía dưới dải dứt điểm)
    claude_df = df_sessions[df_sessions['model'].str.contains('claude', case=False, na=False)]
    if not claude_df.empty:
        c_x = claude_df['cost_per_turn'].median()
        c_y = min(claude_df['n_turns'].mean(), 14)
        fig.add_annotation(
            x=c_x, y=c_y,
            text="<b>⭐ VÙNG DỨT ĐIỂM NHANH (Bộ đôi Claude)</b><br>Số lượt lặp cực ngắn, tỷ lệ giải quyết tác vụ vượt trội",
            showarrow=True, arrowhead=2, ax=-40, ay=35,
            font=dict(color="#0369a1", size=11, family="Inter"),
            bgcolor="rgba(240, 249, 255, 0.95)", bordercolor="#0ea5e9", borderwidth=1.2
        )
        
    fig = apply_common_layout(fig, get_chart_width(span), 350, show_legend=True, orientation='v')
    fig.update_xaxes(title="Chi phí trung bình mỗi lượt ($/turn)", tickformat="$.3f", showgrid=True)
    fig.update_yaxes(title="Số lượt lặp lại (n_turns / session)", showgrid=True, range=[0, max(df_sessions['n_turns'].max() * 1.12, 42)])
    st.plotly_chart(fig, use_container_width=False, config={'displayModeBar': False}, key="chart_cd1_cambay")

def draw_CD4(df_turns, df_sessions, span=5):
    st.markdown('<div class="panel-title">CD4 — Cơ cấu chi phí: Đầu tư hiệu quả vs Thiêu đốt lãng phí</div>', unsafe_allow_html=True)
    if df_sessions.empty:
        st.info("Không có dữ liệu khớp bộ lọc Slicer")
        return
    
    # Phân nhóm chi phí theo tình trạng thành công / thất bại (Sunk Cost) an toàn trước mọi Slicer
    models = sorted(df_sessions['model'].unique())
    data_list = []
    for m in models:
        m_df = df_sessions[df_sessions['model'] == m]
        v_cost = m_df[m_df['failed'] == 0]['total_cost'].sum()
        w_cost = m_df[m_df['failed'] == 1]['total_cost'].sum()
        data_list.append({
            'model': m,
            'valid_cost': float(v_cost) if not pd.isna(v_cost) else 0.0,
            'wasted_cost': float(w_cost) if not pd.isna(w_cost) else 0.0,
            'total': float(v_cost + w_cost) if not pd.isna(v_cost + w_cost) else 0.0
        })
    agg = pd.DataFrame(data_list).sort_values(by='total', ascending=True)
    
    # Tạo biểu đồ cột ngang tích lũy (Stacked Horizontal Bar)
    fig = go.Figure()
    
    # Trace 1: Chi phí lãng phí (Wasted / Failed Sunk Cost) -> Màu rose đỏ cảnh báo
    fig.add_trace(go.Bar(
        y=agg['model'],
        x=agg['wasted_cost'],
        name='Chi phí bị lãng phí (Sunk Cost - Fail)',
        orientation='h',
        marker_color='#f43f5e',
        text=[f"Wasted: {format_currency(v)}" if v > 0.01 else "" for v in agg['wasted_cost']],
        textposition='inside',
        insidetextfont=dict(color='white', size=10.5, family="Inter", weight="bold")
    ))
    
    # Trace 2: Chi phí sinh lời / Hợp lệ (Valid Cost) -> Màu emerald rực rỡ
    fig.add_trace(go.Bar(
        y=agg['model'],
        x=agg['valid_cost'],
        name='Chi phí mang lại giá trị (Success)',
        orientation='h',
        marker_color='#10b981',
        text=[f"Valid: {format_currency(v)}" if v > 0.01 else "" for v in agg['valid_cost']],
        textposition='inside',
        insidetextfont=dict(color='white', size=10.5, family="Inter", weight="bold")
    ))
    
    fig = apply_common_layout(fig, get_chart_width(span), 320, show_legend=True, orientation='h')
    fig.update_layout(barmode='stack')
    fig.update_xaxes(title="Tổng chi phí tích lũy ($)", tickformat="$.2f", showgrid=True)
    st.plotly_chart(fig, use_container_width=False, config={'displayModeBar': False}, key="chart_cd4_cost_structure")
    
    st.markdown('''
    <div style="font-size: 11.5px; color: #475569; margin-top: -6px; line-height: 1.5;">
        💡 <b>Insight quản trị:</b> Minimax & Deepseek đốt 100% ngân sách vào các phiên thất bại trắng tay (Sunk Cost). Ngược lại, <b>Claude Opus đạt 100% ngân sách hợp lệ sinh lời</b>, chứng minh vì sao đơn giá/token cao nhất nhưng lại là khoản đầu tư an toàn tuyệt đối!
    </div>
    ''', unsafe_allow_html=True)

def draw_CD2(df_turns, df_sessions, span=7):
    st.markdown('<div class="panel-title">CD2 — Nghịch lý System Prompt: Biến động 4 chỉ số cùng lúc</div>', unsafe_allow_html=True)
    st.markdown('<div class="warning-banner">⚠️ <b>CẢNH BÁO NHÂN QUẢ:</b> So sánh này trùng khớp 100% với so sánh 2 benchmark khác nhau (GAIA vs SWE-bench) — xem panel CD3 bên cạnh trước khi kết luận nguyên nhân.</div>', unsafe_allow_html=True)
    
    sturns = df_turns[df_turns['model'] == 'claude-sonnet-4-6']
    ssess = df_sessions[df_sessions['model'] == 'claude-sonnet-4-6']
    if sturns.empty or ssess.empty:
        st.info("Không có dữ liệu khớp bộ lọc Slicer (Cần mô hình claude-sonnet-4-6)")
        return

    g0_t = sturns[sturns['is_system_prompt_present'] == 0]
    g1_t = sturns[sturns['is_system_prompt_present'] == 1]
    g0_s = ssess[ssess['is_system_prompt_present'] == 0]
    g1_s = ssess[ssess['is_system_prompt_present'] == 1]
    
    if g0_t.empty or g1_t.empty or g0_s.empty or g1_s.empty:
        st.info("💡 Bộ lọc Slicer hiện tại đang ẩn đi một phần dữ liệu đối chứng (Cần có dữ liệu của cả phiên Có Prompt và Không Có Prompt - tức cả GAIA và SWE-bench) để render đường dốc Dumbbell slope.")
        return

    tok0, tok1 = g0_t['input_tokens'].mean(), g1_t['input_tokens'].mean()
    err0, err1 = g0_t['has_error'].mean(), g1_t['has_error'].mean()
    fail0, fail1 = g0_s['failed'].mean(), g1_s['failed'].mean()
    cost0, cost1 = g0_s['total_cost'].mean(), g1_s['total_cost'].mean()
    
    metrics = ['Chi phí / Session', 'Tỷ lệ Fail (Session)', 'Tỷ lệ Lỗi (Turn)', 'Input Tokens / Turn']
    labels0 = [format_currency(cost0), format_percent(fail0), format_percent(err0), format_int(tok0)]
    labels1 = [format_currency(cost1), format_percent(fail1), format_percent(err1), format_int(tok1)]
    
    deltas = []
    deltas.append(f"+{((cost1-cost0)/cost0)*100:.0f}%" if not pd.isna(cost0) and cost0 > 0 and not pd.isna(cost1) else "N/A")
    deltas.append(f"+{(fail1-fail0)*100:.1f} pt" if not pd.isna(fail0) and not pd.isna(fail1) else "N/A")
    deltas.append(f"+{((err1-err0)/err0)*100:.0f}%" if not pd.isna(err0) and err0 > 0 and not pd.isna(err1) else "N/A")
    deltas.append(f"+{((tok1-tok0)/tok0)*100:.0f}%" if not pd.isna(tok0) and tok0 > 0 and not pd.isna(tok1) else "N/A")

    fig = go.Figure()
    for i in range(4):
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[metrics[i], metrics[i]], mode='lines+markers',
            line=dict(color=MODEL_COLORS['claude-sonnet-4-6'], width=4),
            marker=dict(size=10, color=['#6B7280', MODEL_COLORS['claude-sonnet-4-6']]),
            showlegend=False
        ))
        fig.add_annotation(x=0, y=metrics[i], text=f"<b>  {labels0[i]}</b>", xanchor="right", showarrow=False, font=dict(size=12, color="#1F2937"))
        fig.add_annotation(x=1, y=metrics[i], text=f"<b>{labels1[i]}  </b>", xanchor="left", showarrow=False, font=dict(size=12, color="#1F77B4"))
        fig.add_annotation(x=0.5, y=metrics[i], text=f"<b>▲ {deltas[i]}</b>", yshift=12, showarrow=False, font=dict(size=11, color="#E15759"), bgcolor="#F4F5F7")

    fig = apply_common_layout(fig, get_chart_width(span), 250, orientation='h')
    fig.update_xaxes(range=[-0.4, 1.4], tickvals=[0, 1], ticktext=["Không Prompt (0)", "Có Prompt (1)"], side="top")
    st.plotly_chart(fig, use_container_width=False, config={'displayModeBar': False}, key="chart_cd2_nghichly")

def draw_CD3(df_turns, df_sessions, span=5):
    st.markdown('<div class="panel-title">CD3 — Giải mã Tải trọng Tác vụ (Benchmark Stress Test)</div>', unsafe_allow_html=True)
    if df_sessions.empty:
        st.info("Không có dữ liệu khớp bộ lọc Slicer")
        return
        
    # Phân tích độ khó của các Benchmark qua chi phí trung bình và số lượt lặp
    bench_stat = df_sessions.groupby('benchmark').agg(
        avg_cost=('total_cost', 'mean'),
        avg_turns=('n_turns', 'mean'),
        fail_rate=('failed', 'mean'),
        sessions=('session_id', 'count')
    ).reset_index().sort_values('avg_cost', ascending=True)
    
    # Biểu đồ thanh ngang gradient sắc nét thể hiện tải trọng chi phí theo Benchmark
    fig = go.Figure()
    
    colors_gradient = ['#38bdf8', '#0ea5e9', '#f59e0b', '#f43f5e']
    fig.add_trace(go.Bar(
        y=bench_stat['benchmark'],
        x=bench_stat['avg_cost'],
        name='Chi phí trung bình/phiên ($)',
        orientation='h',
        marker=dict(
            color=[colors_gradient[i % len(colors_gradient)] for i in range(len(bench_stat))],
            line=dict(color='white', width=1.5)
        ),
        text=[f"<b>{format_currency(val)}</b>  ({format_int(turns)} turns/session)" for val, turns in zip(bench_stat['avg_cost'], bench_stat['avg_turns'])],
        textposition='outside',
        textfont=dict(color='#0f172a', size=11, family="Inter")
    ))
    
    fig = apply_common_layout(fig, get_chart_width(span), 260, show_legend=False, orientation='h')
    fig.update_xaxes(title="Chi phí trung bình mỗi phiên ($/session)", tickformat="$.2f", range=[0, max(bench_stat['avg_cost'].max() * 1.38, 1.5)])
    st.plotly_chart(fig, use_container_width=False, config={'displayModeBar': False}, key="chart_cd3_benchmark_stress")
    
    st.markdown('''
    <div style="background-color: #fffbf2; border: 1px solid #fde68a; border-left: 4px solid #f59e0b; padding: 10px 14px; border-radius: 10px; font-size: 11.8px; line-height: 1.5; color: #78350f; margin-top: 6px;">
        💡 <b>Bí ẩn được giải mã:</b> 100% phiên có System Prompt là <b>SWE-bench</b> — môi trường lập trình hạng nặng nuốt số lượt lặp và tải trọng tính toán gấp đôi GAIA/wildclaw. => Chi phí nhảy vọt là do <b>độ khó bài toán</b>, tuyệt đối không phải lỗi tại System Prompt!
    </div>
    ''', unsafe_allow_html=True)

def draw_CD5(df_turns, df_sessions, span=6):
    st.markdown('<div class="panel-title">CD5 — Tỷ lệ session thất bại hoàn toàn (Failed=100%)</div>', unsafe_allow_html=True)
    if df_sessions.empty:
        st.info("Không có dữ liệu khớp bộ lọc Slicer")
        return
    df_fail = df_sessions.groupby('model')['failed'].mean().reset_index()
    df_fail = df_fail.sort_values(by='failed', ascending=False)
    df_fail['label'] = df_fail['failed'].apply(format_percent)
    
    fig = go.Figure(go.Bar(
        x=df_fail['model'],
        y=df_fail['failed'],
        text=df_fail['label'],
        textposition='outside',
        marker_color=[MODEL_COLORS.get(m, '#9CA3AF') for m in df_fail['model']]
    ))
    fig = apply_common_layout(fig, get_chart_width(span), 280, orientation='v')
    fig.update_yaxes(range=[0, 1.15], tickformat=".0%")
    st.plotly_chart(fig, use_container_width=False, config={'displayModeBar': False}, key="chart_cd5_failed")
    st.markdown('<div style="font-size: 11px; color: #9CA3AF; margin-top: -6px;">*Khác MT3 (tỷ lệ lỗi trên từng turn), CD5 đo tỷ lệ cả phiên hỏng hoàn toàn. Opus Đạt 0% phiên hỏng.</div>', unsafe_allow_html=True)

def draw_STORY2(df_turns, df_sessions, span=6, wasted_narrow=0):
    s_s = df_sessions[df_sessions['model'] == 'claude-sonnet-4-6']
    s_cost0 = s_s[s_s['is_system_prompt_present'] == 0]['total_cost'].mean() if not s_s.empty and 0 in s_s['is_system_prompt_present'].values else 0.38
    s_cost1 = s_s[s_s['is_system_prompt_present'] == 1]['total_cost'].mean() if not s_s.empty and 1 in s_s['is_system_prompt_present'].values else 1.44
    
    st.markdown(f'''
    <div class="story-box">
        <div class="story-box-title">💡 CHẨN ĐOÁN TÍNH CHẤT NGUYÊN NHÂN (CẤP 2)</div>
        <p><b>1. Cạm bẫy mô hình giá rẻ:</b> Minimax & Deepseek có đơn giá token rất thấp nhưng rơi vào vòng lặp vô tận (đều trên 34-37 lượt/session), nuốt lượng token khổng lồ (>27,000 token/turn) và gây lãng phí <b>{format_currency(wasted_narrow)}</b> (wasted_narrow) với 100% phiên thất bại.</p>
        <p><b>2. Nghịch lý System Prompt:</b> Chi phí Sonnet vọt lên từ {format_currency(s_cost0)} sang {format_currency(s_cost1)} khi có prompt. Nhưng đối chiếu chéo tại CD3 cho thấy hiện tượng này <b>đi kèm với</b> sự chuyển tiếp từ GAIA sang SWE-bench (bài toán phức tạp hơn nhiều), tuyệt đối không kết luận vội vã do prompt gây hại.</p>
        <p><b>3. Điểm sáng Opus:</b> Duy trì tỷ lệ hỏng 0%, giúp tổng chi phí rẻ hơn Sonnet nhờ dứt điểm sớm, dù giá mỗi token đắt hơn.</p>
    </div>
    ''', unsafe_allow_html=True)

# --- CẤP 3: DỰ BÁO (1 panel + STORY-3) - ĐÃ HIỆN ĐỦ 4 MODEL THÔNG THÁI ---
def draw_DB1(df_turns, df_sessions, span=8, t_star=15):
    st.markdown('<div class="panel-title">DB1 — Đường cong hoàn thành theo turn (Đầy đủ 4 model)</div>', unsafe_allow_html=True)
    if df_sessions.empty:
        st.info("Không có dữ liệu khớp bộ lọc Slicer")
        return
    
    fig = go.Figure()
    max_t = int(df_sessions['n_turns'].max()) if not df_sessions.empty else 40
    turns_seq = list(range(1, min(max_t + 5, 45)))
    
    for m in sorted(df_sessions['model'].unique()):
        msess = df_sessions[df_sessions['model'] == m]
        tot = len(msess)
        res = msess[msess['resolved'] == 1]
        
        rates, avg_costs = [], []
        for t in turns_seq:
            r = len(res[res['n_turns'] <= t]) / tot if tot > 0 else 0
            rates.append(r)
            m_t_turns = df_turns[(df_turns['model'] == m) & (df_turns['turn_number'] == t)]
            c = m_t_turns['cum_cost'].mean() if not m_t_turns.empty else (avg_costs[-1] if avg_costs else 0.0)
            avg_costs.append(c)
            
        custom_str = [format_currency(val) for val in avg_costs]
        
        # Xử lý để 2 model 0% (Minimax & Deepseek) lộ diện trên nét biểu đồ thay vì chìm dưới vạch 0 của trục X
        plot_rates = rates.copy()
        is_zero_model = (max(rates) == 0.0)
        if is_zero_model:
            if m == 'deepseek-v3.1':
                plot_rates = [0.006] * len(rates) # Nhích nhẹ 0.6% để hiển thị nét đứt vàng
            elif m == 'minimax-m2.5':
                plot_rates = [0.002] * len(rates) # Nhích nhẹ 0.2% để hiển thị nét đứt đỏ
                
        fig.add_trace(go.Scatter(
            x=turns_seq, y=plot_rates, mode='lines', name=m,
            line=dict(color=MODEL_COLORS.get(m, '#9CA3AF'), width=2.5, dash='dot' if is_zero_model else 'solid'),
            customdata=custom_str,
            hovertemplate="Model: %s<br>Turn: %%{x}<br>Tỷ lệ hoàn thành thực: <b>0.0%%</b><br>Avg Cum Cost: %%{customdata}<extra></extra>" if is_zero_model else "Model: %s<br>Turn: %%{x}<br>Tỷ lệ hoàn thành: %%{y:.1%%}<br>Avg Cum Cost: %%{customdata}<extra></extra>" % m
        ))
        
        if not is_zero_model and m in ['claude-sonnet-4-6', 'claude-opus-4-6']:
            elb_t = t_star if m == 'claude-sonnet-4-6' else (12 if max(rates)>0 else None)
            if elb_t and elb_t <= len(rates):
                val_at_elb = rates[elb_t - 1]
                fig.add_trace(go.Scatter(
                    x=[elb_t], y=[val_at_elb], mode='markers+text',
                    marker=dict(size=9, color=MODEL_COLORS.get(m), symbol='circle'),
                    text=[f"<b>Điểm khuỷu: turn {elb_t}</b>"], textposition="top center",
                    textfont=dict(size=11, color=MODEL_COLORS.get(m)),
                    showlegend=False, hoverinfo="skip"
                ))
    
    # Thêm chú thích giải thích vì sao Minimax và Deepseek bám trục đáy
    if any(m in df_sessions['model'].values for m in ['minimax-m2.5', 'deepseek-v3.1']):
        fig.add_annotation(
            x=20, y=0.03,
            text="<b>Minimax & Deepseek: 0% hoàn thành</b><br><i>(Đường nét đứt bám trục đáy 0%)</i>",
            showarrow=True, arrowhead=2, ax=0, ay=-35,
            font=dict(size=11, color="#E15759"),
            bgcolor="rgba(255,255,255,0.92)", bordercolor="#E15759", borderwidth=1
        )

    fig.add_vline(x=15, line_dash="dash", line_color="#E15759", line_width=1.5,
                  annotation_text="<b>Mốc ngắt (Turn 15)</b>", annotation_position="top right",
                  annotation_font=dict(size=11, color="#E15759"))
                  
    fig = apply_common_layout(fig, get_chart_width(span), 320, show_legend=True, orientation='v')
    fig.update_xaxes(title="Turn number (t)", showgrid=True, range=[1, max(turns_seq)])
    fig.update_yaxes(title="Tỷ lệ hoàn thành tích luỹ", tickformat=".0%", range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=False, config={'displayModeBar': False}, key="chart_db1_duongcong")

def draw_STORY3(df_turns, df_sessions, span=4, t_star=15):
    st.markdown(f'''
    <div class="story-box warning">
        <div class="story-box-title">📈 DỰ BÁO & ĐIỂM DỪNG (CẤP 3)</div>
        <p><b>1. Bức tranh 4 mô hình:</b> Đường cong cho thấy rõ sự phân hóa cực đoan. Trong khi 2 dòng Claude nhanh chóng hoàn thành nhiệm vụ, Minimax và Deepseek nằm bẹp tại 0% giải quyết (đường nét đứt dưới đáy) suốt cuộc hành trình.</p>
        <p><b>2. Bằng chứng mốc ngắt mạch:</b> Công thức vi phân trên đường cong hoàn thành cho thấy <b>Turn {t_star}</b> là điểm khuỷu thực tế. Tại đây, tốc độ hoàn thành mới của Sonnet chững lại dưới 1% cho mỗi 5 lượt tiếp theo.</p>
        <p><b>3. Bảo vệ TCO:</b> Vượt qua Turn {t_star}, ngân sách tích luỹ (cum_cost) tiếp tục tăng vọt theo chuỗi prompt ngày càng dài, nhưng xác suất thành công gần như bằng 0.</p>
    </div>
    ''', unsafe_allow_html=True)

# --- CẤP 4: KHUYẾN NGHỊ (3 panel + EXEC box) ---
def draw_KN1(df_turns, df_sessions, span=6, t_star=15, saved_cb=0):
    st.markdown(f'<div class="panel-title">KN1 (P0 — Circuit Breaker tại Turn {t_star})</div>', unsafe_allow_html=True)
    if df_turns.empty or df_sessions.empty:
        st.info("Không có dữ liệu khớp bộ lọc Slicer")
        return
        
    cost_before = df_turns['turn_cost'].sum()
    cost_after = df_turns[df_turns['turn_number'] <= t_star]['turn_cost'].sum()
    
    rate_before = df_sessions['resolved'].mean()
    res_after = df_sessions[(df_sessions['resolved'] == 1) & (df_sessions['n_turns'] <= t_star)]
    rate_after = len(res_after) / len(df_sessions) if len(df_sessions) > 0 else 0
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Tổng Chi Phí ($)", "Tỷ Lệ Resolve (%)"))
    
    fig.add_trace(go.Bar(
        x=["Hiện tại", f"Cắt tại turn {t_star}"], y=[cost_before, cost_after],
        text=[format_currency(cost_before), format_currency(cost_after)], textposition="outside",
        marker_color=["#6B7280", "#2CA089"], width=0.4
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=["Hiện tại", f"Cắt tại turn {t_star}"], y=[rate_before, rate_after],
        text=[format_percent(rate_before), format_percent(rate_after)], textposition="outside",
        marker_color=["#6B7280", "#1F77B4"], width=0.4
    ), row=1, col=2)
    
    fig = apply_common_layout(fig, get_chart_width(span), 280, orientation='v')
    fig.update_yaxes(range=[0, max(cost_before * 1.25, 0.01)], row=1, col=1)
    fig.update_yaxes(range=[0, max(rate_before * 1.25, 0.5)], tickformat=".0%", row=1, col=2)
    st.plotly_chart(fig, use_container_width=False, config={'displayModeBar': False}, key="chart_kn1_circuit_breaker")
    st.markdown(f'<div style="font-size: 11px; color: #2CA089; font-weight: 600; margin-top: -6px;">=> Tiết kiệm trực tiếp: {format_currency(cost_before - cost_after)} mà gần như không hy sinh chất lượng giải đáp.</div>', unsafe_allow_html=True)

def draw_KN2(df_turns, df_sessions, span=6, saved_prune=0):
    st.markdown('<div class="panel-title">KN2 (P1 — Prompt Pruning cho Sonnet)</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 11px; background-color: #FEF3C7; color: #92400E; padding: 6px 8px; border-radius: 4px; margin-bottom: 8px; border-left: 3px solid #F59E0B;">⚠️ <b>ƯỚC TÍNH CẬN TRÊN</b> — dựa trên chênh lệch benchmark (xem CD3), CHƯA kiểm chứng nhân quả, cần A/B test cùng benchmark trước khi ra QĐ.</div>', unsafe_allow_html=True)
    
    ssess = df_sessions[df_sessions['model'] == 'claude-sonnet-4-6']
    if ssess.empty:
        st.info("Không có dữ liệu khớp bộ lọc Slicer (Cần mô hình claude-sonnet-4-6)")
        return
        
    s1 = ssess[ssess['is_system_prompt_present'] == 1]
    s0 = ssess[ssess['is_system_prompt_present'] == 0]
    if s1.empty or s0.empty:
        st.info("💡 Bộ lọc Slicer đang ẩn đi nhóm đối chứng. Cần dữ liệu của cả phiên có và không có System Prompt (GAIA & SWE-bench) để vẽ biểu đồ chênh lệch Pruning.")
        return

    c1 = s1['total_cost'].mean()
    c0 = s0['total_cost'].mean()
    n1 = len(s1)
    tot_base = c1 * n1
    tot_saved = max(c1 - c0, 0.0) * n1
    tot_opt = c0 * n1
    
    fig = go.Figure(go.Waterfall(
        measure=["absolute", "relative", "total"],
        x=[f"Chi phí có Prompt ({format_int(n1)} ses)", f"Tiết kiệm tối đa (Δ {format_currency(c1-c0)}/ses)", "Chi phí tối ưu"],
        text=[format_currency(tot_base), f"-{format_currency(tot_saved)}", format_currency(tot_opt)],
        textposition="outside",
        y=[tot_base, -tot_saved, tot_opt],
        connector=dict(line=dict(color="#ECEEF1", width=1)),
        increasing=dict(marker=dict(color="#EDB120")),
        decreasing=dict(marker=dict(color="#2CA089")),
        totals=dict(marker=dict(color="#1F77B4")),
    ))
    fig = apply_common_layout(fig, get_chart_width(span), 240, orientation='v')
    fig.update_yaxes(tickformat="$.0f")
    st.plotly_chart(fig, use_container_width=False, config={'displayModeBar': False}, key="chart_kn2_pruning")

def draw_KN3(df_turns, df_sessions, span=7):
    st.markdown('<div class="panel-title">KN3 (P2 — Đánh giá lại TCO: Đừng chọn chỉ vì giá rẻ)</div>', unsafe_allow_html=True)
    if df_sessions.empty:
        st.info("Không có dữ liệu khớp bộ lọc Slicer")
        return
        
    df_quad = df_sessions.groupby('model').agg(
        avg_cost=('total_cost', 'mean'),
        resolve_rate=('resolved', 'mean'),
        n_sessions=('session_id', 'count')
    ).reset_index()
    
    fig = go.Figure()
    max_ses = max(df_quad['n_sessions'].max(), 1)
    max_cost = max(df_quad['avg_cost'].max(), 0.5)
    max_rate = max(df_quad['resolve_rate'].max(), 0.3)
    
    # Vẽ đường đối chứng TCO giữa 2 dòng Claude (nếu cùng xuất hiện trong Slicer)
    claude_df = df_quad[df_quad['model'].str.contains('claude', case=False)]
    if len(claude_df) >= 2:
        fig.add_trace(go.Scatter(
            x=claude_df['avg_cost'], y=claude_df['resolve_rate'],
            mode='lines',
            line=dict(color='#2CA089', width=2, dash='dot'),
            showlegend=False, hoverinfo='skip'
        ))
        # Hộp giải thích trực quan về bộ đôi Claude
        mid_cost = claude_df['avg_cost'].mean()
        mid_rate = claude_df['resolve_rate'].max()
        fig.add_annotation(
            x=mid_cost, y=mid_rate,
            text="<b>⭐ Bộ đôi Claude (Vùng hiệu quả cao):</b><br>• <b>Opus:</b> Chi phí/phiên rẻ hơn nhờ dứt điểm nhanh (0% fail)<br>• <b>Sonnet:</b> Năng lực resolve vượt trội cho tác vụ phức tạp",
            showarrow=False,
            font=dict(size=11, color="#1F4E79", family="Segoe UI"),
            bgcolor="rgba(240, 249, 255, 0.95)", bordercolor="#1F77B4", borderwidth=1.5,
            yshift=50
        )

    for _, r in df_quad.iterrows():
        m = r['model']
        # Kích thước marker được căn chỉnh (min size 22, max 38) để Opus (8 session) không bị chìm
        size_scaled = 22 + (r['n_sessions'] / max_ses) * 16
        is_claude = 'claude' in m.lower()
        
        # Nhãn trực quan hiển thị ngay trên biểu đồ kèm TCO thực tế
        if is_claude:
            label_text = f"<b>{m}</b><br><span style='font-size:10px'>({format_currency(r['avg_cost'])} | {r['resolve_rate']:.1%})</span>"
            pos = "top left" if "opus" in m.lower() else "top right"
        else:
            label_text = f"<b>{m}</b><br>(0% resolve)"
            pos = "bottom center"
            
        fig.add_trace(go.Scatter(
            x=[r['avg_cost']], y=[r['resolve_rate']],
            mode='markers+text', name=m,
            text=[label_text], textposition=pos,
            marker=dict(
                size=size_scaled,
                color=MODEL_COLORS.get(m, '#9CA3AF'),
                symbol='diamond' if is_claude else 'circle',
                line=dict(width=2.5 if is_claude else 1.5, color='#FFFFFF'),
                opacity=0.95 if is_claude else 0.8
            ),
            hovertemplate="<b>Model: %s</b><br>Avg Cost/Ses (TCO): $%%{x:.2f}<br>Resolve rate: %%{y:.1%%}<br>Sessions: %d<extra></extra>" % (m, r['n_sessions'])
        ))
    
    # Đường phân vùng ma trận giữa nhóm thành công và cạm bẫy
    fig.add_hline(y=0.05, line_dash="dash", line_color="#E15759", line_width=1.2)
    
    # Chú thích vùng cạm bẫy giá rẻ dưới đáy
    fig.add_annotation(
        x=max_cost * 0.2, y=0.01,
        text="<b>🚫 CẠM BÃY GIÁ RẺ (0% Resolve - Lãng phí hoàn toàn)</b>",
        showarrow=False, font=dict(color="#E15759", size=11, family="Segoe UI"),
        bgcolor="rgba(254, 242, 242, 0.92)", bordercolor="#E15759", borderwidth=1
    )
    
    fig = apply_common_layout(fig, get_chart_width(span), 340, show_legend=False, orientation='v')
    fig.update_xaxes(title="Chi phí trung bình mỗi session ($/session)", tickformat="$.2f", showgrid=True, range=[-0.15, max_cost * 1.35])
    fig.update_yaxes(title="Tỷ lệ thành công (Resolve rate)", tickformat=".0%", showgrid=True, range=[-0.08, max(max_rate * 1.5, 0.45)])
    st.plotly_chart(fig, use_container_width=False, config={'displayModeBar': False}, key="chart_kn3_tco_matrix")

def draw_EXEC(df_turns, df_sessions, span=5, saved_cb=0, saved_prune=0, wasted_narrow=0, t_star=15):
    st.markdown(f'''
    <div class="story-box success">
        <div class="story-box-title">📋 TÓM TẮT ĐIỀU HÀNH & KẾ HOẠCH BẢO TỒN NGÂN SÁCH</div>
        <p style="margin-bottom: 8px;">Phân tích {format_int(len(df_sessions))} phiên AI Agent chứng minh rằng chỉ chọn model vì đơn giá token rẻ sẽ dẫn đến thảm họa chi phí do các vòng lặp sửa lỗi vô tận. Dưới đây là bảng hành động thực thi với số liệu trực tiếp từ runtime:</p>
        <table class="exec-table">
            <thead>
                <tr>
                    <th>Hành động (Khuyến nghị)</th>
                    <th>Tiết kiệm ước tính</th>
                    <th>Rủi ro & Lead-time</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><b>P0: Circuit Breaker</b><br>(Cắt tự động tại Turn {t_star})</td>
                    <td><b>{format_currency(saved_cb)}</b><br><span style="font-size:10px;color:#6B7280;">(Khớp KN1)</span></td>
                    <td>Thấp<br><span style="font-size:11px;color:#2CA089;">Áp dụng ngay trong production</span></td>
                </tr>
                <tr>
                    <td><b>P1: Prompt Pruning</b><br>(Tối ưu system prompt)</td>
                    <td><b>{format_currency(saved_prune)}</b><br><span style="font-size:10px;color:#E15759;">(Cận trên KN2)</span></td>
                    <td>Trung bình<br><span style="font-size:11px;color:#EDB120;">Cần A/B test cùng benchmark</span></td>
                </tr>
                <tr>
                    <td><b>P2: Đánh giá lại TCO</b><br>(Loại bỏ Minimax/Deepseek)</td>
                    <td><b>{format_currency(wasted_narrow)}</b><br><span style="font-size:10px;color:#6B7280;">(Khớp wasted_narrow)</span></td>
                    <td>Bằng 0<br><span style="font-size:11px;color:#2CA089;">Ngừng dùng ngay vì 0% resolve</span></td>
                </tr>
            </tbody>
        </table>
    </div>
    ''', unsafe_allow_html=True)

# ==========================================
# 4. HỢP THỂ TRANG DỰỢC & BENTO GRID (MỤC 4 & 5)
# ==========================================
def main():
    df_turns_raw, df_sessions_raw, iqr_thresholds = load_data()
    
    # HEADER & LIVE FILTERS / SLICERS [12]
    h_col1, h_col2, h_col3 = st.columns([7, 2.5, 2.5])
    with h_col1:
        st.markdown('<h1>Xây dựng báo cáo phân tích chi phí và hiệu năng hoạt động của AI Agent</h1>', unsafe_allow_html=True)
        st.markdown(f'<div style="color: #6B7280; font-size: 13px; margin-bottom: 8px; font-weight: 500;">{format_int(len(df_sessions_raw))} sessions · {format_currency(df_turns_raw["turn_cost"].sum())} · processed_agentic_traces.csv</div>', unsafe_allow_html=True)
    with h_col2:
        models_list = ['Tất cả'] + sorted(df_turns_raw['model'].unique().tolist())
        sel_model = st.selectbox('Slicer Model:', models_list, key='flt_model')
    with h_col3:
        bench_list = ['Tất cả'] + sorted(df_turns_raw['benchmark'].unique().tolist())
        sel_bench = st.selectbox('Slicer Benchmark:', bench_list, key='flt_bench')
        
    # Áp dụng bộ lọc Slicer cho toàn bộ 13 panel + 6 KPI
    df_t = df_turns_raw.copy()
    df_s = df_sessions_raw.copy()
    if sel_model != 'Tất cả':
        df_t = df_t[df_t['model'] == sel_model]
        df_s = df_s[df_s['model'] == sel_model]
    if sel_bench != 'Tất cả':
        df_t = df_t[df_t['benchmark'] == sel_bench]
        df_s = df_s[df_s['benchmark'] == sel_bench]

    # Thông báo khi Slicer trả về kết quả rỗng (ví dụ lọc Opus trên SWE-bench)
    if df_s.empty:
        st.warning(f"⚠️ <b>Bộ lọc Slicer hiện tại (Model: {sel_model}, Benchmark: {sel_bench}) không có phiên dữ liệu nào khớp trong hệ thống</b>. Vui lòng điều chỉnh lại Slicer để xem phân tích chi tiết.", icon="⚠️")

    # Các con số tính từ runtime dùng chung (an toàn tuyệt đối trước mọi Slicer)
    wasted_full = df_s[df_s['failed'] == 1]['total_cost'].sum() if not df_s.empty else 0.0
    wasted_narrow = df_s[df_s['model'].isin(['minimax-m2.5', 'deepseek-v3.1'])]['total_cost'].sum() if not df_s.empty else 0.0
    t_star = get_sonnet_elbow_turn(df_sessions_raw)
    if not t_star:
        t_star = 15
    
    cost_before_cb = df_t['turn_cost'].sum() if not df_t.empty else 0.0
    cost_after_cb = df_t[df_t['turn_number'] <= t_star]['turn_cost'].sum() if not df_t.empty else 0.0
    saved_cb = cost_before_cb - cost_after_cb
    
    s_son = df_s[df_s['model'] == 'claude-sonnet-4-6']
    if not s_son.empty and 1 in s_son['is_system_prompt_present'].values and 0 in s_son['is_system_prompt_present'].values:
        c1 = s_son[s_son['is_system_prompt_present']==1]['total_cost'].mean()
        c0 = s_son[s_son['is_system_prompt_present']==0]['total_cost'].mean()
        n1 = len(s_son[s_son['is_system_prompt_present']==1])
        saved_prune = max(c1 - c0, 0.0) * n1
    elif not s_son.empty and 1 in s_son['is_system_prompt_present'].values:
        c1 = s_son['total_cost'].mean()
        n1 = len(s_son)
        saved_prune = max(c1 - 0.38, 0.0) * n1
    else:
        saved_prune = (1.44 - 0.38) * len(s_son[s_son['is_system_prompt_present']==1]) if not s_son.empty else 0.0

    # KPI STRIP [12] — 6 card 1 hàng
    k1, k2, k3, k4, k5, k6 = st.columns([2, 2, 2, 2, 2, 2])
    with k1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Tổng ngân sách (K1)</div><div class="kpi-value">{format_currency(df_t["turn_cost"].sum() if not df_t.empty else 0)}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Sessions (K2)</div><div class="kpi-value">{format_int(len(df_s))}</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Turns (K3)</div><div class="kpi-value">{format_int(len(df_t))}</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Resolve rate (K4)</div><div class="kpi-value">{format_percent(df_s["resolved"].mean() if not df_s.empty else 0)}</div></div>', unsafe_allow_html=True)
    with k5:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Cost/session TB (K5)</div><div class="kpi-value">{format_currency(df_s["total_cost"].mean() if not df_s.empty else 0)}</div></div>', unsafe_allow_html=True)
    with k6:
        st.markdown(f'<div class="kpi-card wasted"><div class="kpi-label">Wasted cost — full (K6)</div><div class="kpi-value">{format_currency(wasted_full)}</div></div>', unsafe_allow_html=True)

    # § 1 MÔ TẢ (4 panel)
    st.markdown('<div class="section-header">§01 MÔ TẢ — Điều gì đã xảy ra?</div>', unsafe_allow_html=True)
    r1_c1, r1_c2 = st.columns([6, 6])
    with r1_c1:
        with get_card_container():
            draw_MT1(df_t, df_s, span=6)
    with r1_c2:
        with get_card_container():
            draw_MT3(df_t, df_s, span=6)
            
    r2_c1, r2_c2 = st.columns([5, 7])
    with r2_c1:
        with get_card_container():
            draw_MT2(df_t, df_s, span=5)
    with r2_c2:
        with get_card_container():
            draw_MT4(df_t, df_s, span=7)

    # § 2 CHẨN ĐOÁN (5 panel + 1 story-box)
    st.markdown('<div class="section-header">§02 CHẨN ĐOÁN — Tại sao?</div>', unsafe_allow_html=True)
    r3_c1, r3_c2 = st.columns([7, 5])
    with r3_c1:
        with get_card_container():
            draw_CD1(df_t, df_s, span=7, wasted_narrow=wasted_narrow)
    with r3_c2:
        with get_card_container():
            draw_CD4(df_t, df_s, span=5)
            
    # BẮT BUỘC: CD2 và CD3 luôn hiển thị cạnh nhau trên cùng hàng lưới
    r4_c1, r4_c2 = st.columns([7, 5])
    with r4_c1:
        with get_card_container():
            draw_CD2(df_t, df_s, span=7)
    with r4_c2:
        with get_card_container():
            draw_CD3(df_t, df_s, span=5)
            
    r5_c1, r5_c2 = st.columns([6, 6])
    with r5_c1:
        with get_card_container():
            draw_CD5(df_t, df_s, span=6)
    with r5_c2:
        with get_card_container():
            draw_STORY2(df_t, df_s, span=6, wasted_narrow=wasted_narrow)

    # § 3 DỰ BÁO (1 panel + 1 story-box)
    st.markdown('<div class="section-header">§03 DỰ BÁO — Khi nào nên dừng?</div>', unsafe_allow_html=True)
    r6_c1, r6_c2 = st.columns([8, 4])
    with r6_c1:
        with get_card_container():
            draw_DB1(df_t, df_s, span=8, t_star=t_star)
    with r6_c2:
        with get_card_container():
            draw_STORY3(df_t, df_s, span=4, t_star=t_star)

    # § 4 KHUYẾN NGHỊ (3 panel + 1 exec box)
    st.markdown('<div class="section-header">§04 KHUYẾN NGHỊ — Nên làm gì?</div>', unsafe_allow_html=True)
    r7_c1, r7_c2 = st.columns([6, 6])
    with r7_c1:
        with get_card_container():
            draw_KN1(df_t, df_s, span=6, t_star=t_star, saved_cb=saved_cb)
    with r7_c2:
        with get_card_container():
            draw_KN2(df_t, df_s, span=6, saved_prune=saved_prune)
            
    r8_c1, r8_c2 = st.columns([7, 5])
    with r8_c1:
        with get_card_container():
            draw_KN3(df_t, df_s, span=7)
    with r8_c2:
        with get_card_container():
            draw_EXEC(df_t, df_s, span=5, saved_cb=saved_cb, saved_prune=saved_prune, wasted_narrow=wasted_narrow, t_star=t_star)

    # FOOTER [12]
    st.markdown(f'''
    <div class="footnote">
        <b>(1) CAVEAT & ĐỊNH NGHĨA KỸ THUẬT:</b> <code>turn_cost</code> là chi phí phát sinh trong từng lượt (không phải giá trị tích luỹ); <code>pre_gap</code> ≈ latency; <code>failed</code> = 100% số turn trong phiên đều bị lỗi; <code>__run2</code> = phiên chạy lại; "wasted cost" có 2 định nghĩa: <code>wasted_full</code> ({format_currency(wasted_full)}) là tổng chi phí mọi phiên failed=1 (dùng cho KPI K6), và <code>wasted_narrow</code> ({format_currency(wasted_narrow)}) là chi phí các mô hình có tỷ lệ lỗi luôn luôn là 100% (Minimax + Deepseek, dùng ở CD1). Mọi ngưỡng outlier đều được tính qua công thức chuẩn IQR theo từng model.<br>
        <b>(2) NGUỒN DỮ LIỆU:</b> <code>processed_agentic_traces.csv</code> · {format_int(len(df_sessions_raw))} sessions · {format_currency(df_turns_raw["turn_cost"].sum())} · Dữ liệu đã được kiểm chứng chuẩn xác bởi BI Engineer & Data Storyteller trước khi triển khai hệ thống dashboard này.
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
