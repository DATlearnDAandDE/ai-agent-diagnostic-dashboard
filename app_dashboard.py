from pathlib import Path

# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
# pyrefly: ignore [missing-import]
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ==========================================
# CẤU HÌNH TRANG TỔNG THỂ & CSS (BENTO GRID)
# ==========================================
st.set_page_config(page_title="AI Agent Diagnostic Dashboard", layout="wide", page_icon="📈", initial_sidebar_state="collapsed")

# Premium Custom CSS (Glassmorphism + Bento Style)
st.markdown("""
<style>
    /* Dark Gradient Background for the entire app */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    }
    
    /* Bento Box Container Styling */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease, border 0.3s ease !important;
        padding: 0.5rem !important;
    }
    
    /* Hover Effect for Bento Boxes */
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 12px 40px 0 rgba(139, 92, 246, 0.25) !important;
        border: 1px solid rgba(139, 92, 246, 0.4) !important;
    }
    
    /* Text Styles */
    h1, h2, h3, h4, p, span {
        font-family: 'Inter', sans-serif;
        color: #f8fafc !important;
    }
    
    .bento-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 5px;
        color: #e2e8f0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* KPI Metrics custom styling */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 4px solid #3b82f6; /* Blue Accent */
    }
    
    /* Recommendations Box */
    .rec-box {
        background: linear-gradient(145deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.15));
        border-left: 4px solid #8b5cf6;
        padding: 15px;
        border-radius: 10px;
        font-size: 0.95rem;
        color: #cbd5e1;
        line-height: 1.6;
        height: 100%;
    }
    
    /* Hide top padding of main block */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 95% !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 AI Agents Code Debug - 4-Level Diagnostics")
st.markdown("*Bảng điều khiển phân tích chuyên sâu hiệu năng & ngân sách tuân theo framework 4 cấp độ (Descriptive, Diagnostic, Predictive, Prescriptive).*")

# ==========================================
# TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU
# ==========================================
@st.cache_data
def load_data():
    app_dir = Path(__file__).resolve().parent
    data_candidates = [
        app_dir / 'processed_agentic_traces.csv',
        app_dir / 'processed_agentic_traces (1).csv',
    ]

    data_path = None
    for candidate in data_candidates:
        if candidate.exists():
            data_path = candidate
            break

    if data_path is None:
        raise FileNotFoundError("Không tìm thấy file dữ liệu processed_agentic_traces.csv")

    df = pd.read_csv(data_path)
    
    # Ép kiểu an toàn tránh lỗi numeric
    numeric_cols = ['output_length', 'pre_gap', 'has_error', 'turn_cost', 'turn_number', 'input_tokens']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Tính toán các Feature phái sinh cốt lõi
    df['latency'] = df['pre_gap']
    df['success'] = 1 - df['has_error']
    df['sunk_cost'] = df['has_error'] * df['turn_cost']
    df['success_cost'] = df['success'] * df['turn_cost']
    
    # Tính cờ Outlier (Dựa trên báo cáo phân tích trước đó)
    Q3_cost = df['turn_cost'].quantile(0.75)
    IQR_cost = Q3_cost - df['turn_cost'].quantile(0.25)
    df['is_cost_outlier'] = df['turn_cost'] > (Q3_cost + 1.5 * IQR_cost)

    Q3_lat = df['latency'].quantile(0.75)
    IQR_lat = Q3_lat - df['latency'].quantile(0.25)
    df['is_lat_outlier'] = df['latency'] > (Q3_lat + 1.5 * IQR_lat)

    # Phân nhóm (Binning)
    df['task_size'] = pd.cut(df['output_length'], bins=[-1, 100, 500, np.inf], labels=['1. Nhẹ (<100 tokens)', '2. Vừa (100-500)', '3. Nặng (>500)'])
    df['context_size'] = pd.cut(df['input_tokens'], bins=[-1, 10000, 30000, np.inf], labels=['1. Thấp (<10K)', '2. TB (10K-30K)', '3. Cao (>30K)'])
    
    return df

with st.spinner("Đang tải & xử lý dữ liệu..."):
    df = load_data()

# Bộ màu chủ đạo
color_map = {
    'claude-opus-4-6': '#3b82f6',   # Blue
    'claude-sonnet-4-6': '#60a5fa', # Light Blue
    'deepseek-v3.1': '#8b5cf6',     # Purple
    'minimax-m2.5': '#ec4899'       # Pink
}

# Hàm cấu hình theme cho biểu đồ
def apply_bento_theme(fig, title):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e2e8f0")),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8', family="Inter"),
        margin=dict(l=10, r=10, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color="#64748b")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, color="#64748b")
    return fig

# ==========================================
# BENTO GRID LAYOUT
# ==========================================

total_turns = len(df)
avg_error_rate = df['has_error'].mean() * 100
total_cost = df['turn_cost'].sum()
total_sunk = df['sunk_cost'].sum()
sunk_pct = (total_sunk / total_cost) * 100 if total_cost > 0 else 0

# --- ROW 1: KPIs & Insights (Span 2/3 and 1/3) ---
r1_col1, r1_col2 = st.columns([2.5, 1.5])

with r1_col1:
    with st.container(border=True):
        st.markdown("<div class='bento-title'>📊 Tổng Quan Hệ Thống (KPIs)</div>", unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Tổng lượt (Turns)", f"{total_turns:,.0f}")
        k2.metric("Tỷ lệ Lỗi", f"{avg_error_rate:.2f}%", delta="- Báo động", delta_color="inverse")
        k3.metric("Tổng Ngân sách", f"${total_cost:,.2f}")
        k4.metric("Sunk Cost", f"${total_sunk:,.2f}", delta=f"- {sunk_pct:.1f}%", delta_color="inverse")

with r1_col2:
    with st.container(border=True):
        st.markdown("<div class='bento-title'>💡 Hành Động Đề Xuất (Prescriptive)</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='rec-box'>
            <b>1.</b> <b>Gỡ bỏ System Prompt</b> khi dùng Minimax/Deepseek.<br/>
            <b>2.</b> Thiết lập <b>Hard-Timeout (5s)</b> cho Minimax.<br/>
            <b>3.</b> Smart Routing: <b>Opus</b> cho tác vụ khó, <b>Sonnet</b> cho tác vụ dễ (kèm max_tokens).
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# --- ROW 2: Descriptive & Diagnostic Analytics ---
r2_col1, r2_col2, r2_col3 = st.columns(3)

with r2_col1:
    with st.container(border=True):
        # 1. Cost Outliers (Descriptive)
        fig1 = px.strip(df, x='model', y='turn_cost', color='model', 
                        color_discrete_map=color_map, stripmode='overlay')
        fig1 = apply_bento_theme(fig1, "Cost Outliers (Phơi bày Sonnet)")
        fig1.update_yaxes(title_text="Chi phí ($)")
        fig1.update_xaxes(title_text="")
        st.plotly_chart(fig1, use_container_width=True, key="fig1")

with r2_col2:
    with st.container(border=True):
        # 2. System Prompt Allergy (Diagnostic)
        df_sys = df.groupby(['model', 'is_system_prompt_present'])['has_error'].mean().reset_index()
        df_sys['Prompt Status'] = df_sys['is_system_prompt_present'].map({0: 'No System Prompt', 1: 'Has System Prompt'})
        fig3 = px.bar(df_sys, x='model', y='has_error', color='Prompt Status', barmode='group',
                      color_discrete_sequence=['#3b82f6', '#ec4899'], text_auto='.1%')
        fig3 = apply_bento_theme(fig3, "Dị ứng System Prompt (Lỗi tăng vọt)")
        fig3.update_yaxes(title_text="Tỷ lệ Lỗi")
        fig3.update_xaxes(title_text="")
        st.plotly_chart(fig3, use_container_width=True, key="fig3")

with r2_col3:
    with st.container(border=True):
        # 3. Context Bubble (Predictive)
        df_ctx = df.groupby(['context_size', 'model'], observed=False)['has_error'].mean().reset_index()
        fig5 = px.line(df_ctx, x='context_size', y='has_error', color='model', markers=True,
                       color_discrete_map=color_map)
        fig5 = apply_bento_theme(fig5, "Bong bóng Context (Lỗi tỷ lệ thuận)")
        fig5.update_yaxes(title_text="Tỷ lệ Lỗi")
        fig5.update_xaxes(title_text="Context Size")
        st.plotly_chart(fig5, use_container_width=True, key="fig5")

# --- ROW 3: Predictive & Prescriptive ---
r3_col1, r3_col2, r3_col3 = st.columns(3)

with r3_col1:
    with st.container(border=True):
        # 4. Latency Outliers (Descriptive)
        fig2 = px.box(df, x='model', y='latency', color='model', color_discrete_map=color_map)
        fig2 = apply_bento_theme(fig2, "Độ trễ (Cái đuôi 25 phút của Minimax)")
        fig2.update_layout(yaxis_type="log")
        fig2.update_yaxes(title_text="Độ trễ (s) - Log")
        fig2.update_xaxes(title_text="")
        st.plotly_chart(fig2, use_container_width=True, key="fig2")

with r3_col2:
    with st.container(border=True):
        # 5. Sunk Cost (Diagnostic)
        df_cost = df.groupby('model')[['success_cost', 'sunk_cost']].sum().reset_index()
        fig4 = go.Figure(data=[
            go.Bar(name='ROI (Hữu ích)', x=df_cost['model'], y=df_cost['success_cost'], marker_color='#3b82f6'),
            go.Bar(name='Sunk Cost', x=df_cost['model'], y=df_cost['sunk_cost'], marker_color='#ec4899')
        ])
        fig4.update_layout(barmode='stack')
        fig4 = apply_bento_theme(fig4, "Giải phẫu Ngân sách (ROI vs Sunk)")
        fig4.update_yaxes(title_text="Tổng USD")
        fig4.update_xaxes(title_text="")
        st.plotly_chart(fig4, use_container_width=True, key="fig4")

with r3_col3:
    with st.container(border=True):
        # 6. Routing Matrix (Prescriptive)
        df_route = df.groupby(['model', 'task_size'], observed=False)['success'].mean().unstack().fillna(0)
        fig7 = go.Figure(data=go.Heatmap(
                    z=df_route.values, x=df_route.columns, y=df_route.index,
                    colorscale=[[0, '#ec4899'], [0.5, '#8b5cf6'], [1, '#3b82f6']],
                    text=np.round(df_route.values * 100, 1), texttemplate="%{text}%", showscale=False))
        fig7 = apply_bento_theme(fig7, "Smart Routing Matrix (Tỷ lệ Thành Công)")
        # Customize for heatmap
        fig7.update_layout(margin=dict(l=10, r=10, t=40, b=40))
        st.plotly_chart(fig7, use_container_width=True, key="fig7")