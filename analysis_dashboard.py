import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from scipy.stats import norm

st.set_page_config(page_title="Executive Dashboard: AI Agent Debug", layout="wide", initial_sidebar_state="collapsed")

# =============================================================================
# A. LOAD & PARSE DATA / TÍNH TOÁN METRICS
# =============================================================================
@st.cache_data
def load_data(filepath='processed_agentic_traces.csv'):
    # Đọc CSV không có header
    df = pd.read_csv(filepath, header=None)
    # Gán tên cột theo Schema thực tế
    df.columns = ['task_id', 'model', 'duration', 'cost', 'resolved', 'flag', 'tokens', 'step', 'cumulative_cost']
    
    # Ép kiểu dữ liệu số để tránh lỗi tính toán
    for col in ['duration', 'cost', 'resolved', 'flag', 'tokens', 'step', 'cumulative_cost']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Parse task_id
    parts = df['task_id'].str.split('__', expand=True)
    df['benchmark'] = parts[0]
    df['project'] = parts[1]
    df['issue'] = parts[2]
    
    # Tính cost per step và tokens per step
    df = df.sort_values(['task_id', 'step'])
    df['tokens_per_step'] = df.groupby('task_id')['tokens'].diff().fillna(df['tokens'])
    df['cost_per_step'] = df['cost'] # cost đã là per step theo yêu cầu
    
    # Per TASK (groupby task_id)
    task_df = df.groupby('task_id').agg(
        model=('model', 'first'),
        benchmark=('benchmark', 'first'),
        project=('project', 'first'),
        final_cost=('cumulative_cost', 'max'),
        max_step=('step', 'max'),
        resolved_final=('resolved', 'last'),
        flag_final=('flag', 'last'),
        total_tokens=('tokens', 'max'),
        total_duration=('duration', 'sum')
    ).reset_index()
    
    # Per MODEL
    model_df = task_df.groupby('model').agg(
        n_tasks=('task_id', 'count'),
        resolve_rate=('resolved_final', 'mean'),
        avg_cost_task=('final_cost', 'mean'),
        avg_steps=('max_step', 'mean'),
        avg_duration_task=('total_duration', 'mean'),
        total_cost=('final_cost', 'sum')
    ).reset_index()
    
    # Tính wasted cost cho model
    wasted = task_df[task_df['resolved_final'] == 0].groupby('model')['final_cost'].sum().reset_index()
    wasted.rename(columns={'final_cost': 'wasted_cost'}, inplace=True)
    model_df = model_df.merge(wasted, on='model', how='left').fillna(0)
    
    return df, task_df, model_df

df, task_df, model_df = load_data()

# =============================================================================
# CSS - DESIGN TOKENS (POWER BI LIGHT CORPORATE)
# =============================================================================
st.markdown("""
<style>
    /* CSS Variables */
    :root {
        --page-bg: #F4F5F7;
        --card: #FFFFFF;
        --border: #E3E6EA;
        --heading: #1F4E79;
        --value: #1F2937;
        --label: #6B7280;
        --muted: #9CA3AF;
        --grid: #ECEEF1;
    }
    
    .stApp { background-color: var(--page-bg); font-family: 'Segoe UI', system-ui; }
    
    /* Bento Grid Card */
    .bento-card {
        background-color: var(--card);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        height: 100%;
    }
    
    .panel-title { font-size: 14px; font-weight: 600; color: var(--heading); text-align: left; margin-bottom: 12px; text-transform: uppercase;}
    .section-title { font-size: 12px; font-weight: bold; color: var(--muted); text-transform: uppercase; margin-top: 32px; margin-bottom: 8px;}
    
    /* KPI Strip */
    .kpi-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 16px; margin-bottom: 24px; }
    @media (max-width: 1100px) { .kpi-grid { grid-template-columns: repeat(3, 1fr); } }
    @media (max-width: 768px) { .kpi-grid { grid-template-columns: repeat(2, 1fr); } }
    
    .kpi-card { background: var(--card); border: 1px solid var(--border); border-radius: 4px; padding: 16px 20px; min-height: 96px; }
    .kpi-label { font-size: 13px; color: var(--label); font-weight: 500; }
    .kpi-value { font-size: 30px; font-weight: 600; color: var(--value); font-variant-numeric: tabular-nums; margin: 4px 0; }
    .kpi-sub { font-size: 11px; color: var(--muted); }
    .kpi-wasted { border-left: 3px solid #E15759; }
    
    /* Story Box */
    .story-box { background-color: #F8FAFC; border-left: 3px solid #1F77B4; padding: 12px 16px; font-size: 13px; color: var(--value); font-style: italic; margin-bottom: 16px;}
    
    .exec-summary { background: var(--card); border: 1px solid var(--border); border-radius: 4px; padding: 20px; margin-bottom: 24px;}
    .exec-title { font-size: 18px; font-weight: bold; color: var(--heading); margin-bottom: 16px;}
</style>
""", unsafe_allow_html=True)

# Colors
COLORS = {
    'claude-sonnet-4-6': '#1F77B4',
    'claude-opus-4-6': '#2CA089',
    'deepseek-v3.1': '#EDB120',
    'minimax-m2.5': '#E15759',
    'mean': '#4FA3D4',
    'spike': '#FF6B6B'
}

def pbi_layout(fig, title="", h=260):
    fig.update_layout(
        height=h,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI, system-ui', color='#6B7280', size=11),
        title=dict(text=title, font=dict(size=14, color='#1F4E79', weight='bold')),
        margin=dict(l=48, r=24, t=36, b=32),
        xaxis=dict(showgrid=False, zeroline=False, showline=False, tickfont=dict(size=11, color='#6B7280')),
        yaxis=dict(showgrid=True, gridcolor='#ECEEF1', zeroline=False, showline=False, tickfont=dict(size=11, color='#6B7280')),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=12), itemclick=False, itemdoubleclick=False),
        modebar=dict(remove=['zoom', 'pan', 'select', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'])
    )
    # Bo góc cho bar chart
    fig.update_traces(selector=dict(type='bar'), marker_cornerradius=2)
    return fig

# Wilson Score
def wilson_score(p, n, z=1.96):
    if n == 0: return 0,0
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    spread = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return center - spread, center + spread

# =============================================================================
# HEADER & KPI STRIP
# =============================================================================
st.markdown('<h1 style="font-size:24px; font-weight:600; color:#1F4E79; padding: 24px 0 16px;">EXECUTIVE DASHBOARD: AI Agent Debug Performance</h1>', unsafe_allow_html=True)

k_tot_cost = task_df['final_cost'].sum()
k_tasks = task_df['task_id'].nunique()
k_steps = len(df)
k_res_rate = task_df['resolved_final'].mean() * 100
k_cost_res = task_df[task_df['resolved_final'] == 1]['final_cost'].sum() / max(1, task_df['resolved_final'].sum())
k_waste = task_df[task_df['resolved_final'] == 0]['final_cost'].sum()
k_waste_pct = k_waste / k_tot_cost * 100

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card"><div class="kpi-label">TỔNG NGÂN SÁCH</div><div class="kpi-value">${k_tot_cost:.2f}</div><div class="kpi-sub">4 models · 2 benchmarks</div></div>
    <div class="kpi-card"><div class="kpi-label">TASKS</div><div class="kpi-value">{k_tasks}</div><div class="kpi-sub">swebench: {len(task_df[task_df['benchmark']=='swebench'])} · wildclaw: {len(task_df[task_df['benchmark']=='wildclaw'])}</div></div>
    <div class="kpi-card"><div class="kpi-label">STEPS</div><div class="kpi-value">{k_steps:,}</div><div class="kpi-sub">trung bình ~{int(k_steps/k_tasks)} steps/task</div></div>
    <div class="kpi-card"><div class="kpi-label">RESOLVE RATE</div><div class="kpi-value" style="color: {'#2CA089' if k_res_rate>80 else '#E15759' if k_res_rate<50 else '#1F2937'}">{k_res_rate:.1f}%</div><div class="kpi-sub">Overall success</div></div>
    <div class="kpi-card"><div class="kpi-label">COST / RESOLVED TASK</div><div class="kpi-value">${k_cost_res:.4f}</div><div class="kpi-sub">minimax ≈<span style="color:#2CA089">$0.006</span></div></div>
    <div class="kpi-card kpi-wasted"><div class="kpi-label">WASTED COST</div><div class="kpi-value" style="color:#E15759">${k_waste:.4f}</div><div class="kpi-sub">≈{k_waste_pct:.1f}% tổng ngân sách</div></div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# B. HÀM VẼ PANEL (PLOTLY)
# =============================================================================

# MT1: Ngân sách phân bổ (Bar + Donut)
def draw_mt1(model_df):
    m_sort = model_df.sort_values('total_cost')
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "pie"}]], column_widths=[0.6, 0.4])
    cols = [COLORS.get(m, '#9CA3AF') for m in m_sort['model']]
    fig.add_trace(go.Bar(y=m_sort['model'], x=m_sort['total_cost'], orientation='h', marker_color=cols, text=m_sort['total_cost'].apply(lambda x: f"${x:.2f}"), textposition='auto'), row=1, col=1)
    fig.add_trace(go.Pie(labels=model_df['model'], values=model_df['total_cost'], hole=0.6, marker_colors=[COLORS.get(m) for m in model_df['model']], textinfo='none'), row=1, col=2)
    fig = pbi_layout(fig, h=260)
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    return fig

# MT2: Table Metrics (dùng Plotly Table)
def draw_mt2(model_df):
    fig = go.Figure(data=[go.Table(
        header=dict(values=["Model", "Tasks", "Steps", "$/Task"], align="left", fill_color='#ECEEF1', font=dict(color='#1F4E79', size=11)),
        cells=dict(values=[model_df['model'], model_df['n_tasks'], (model_df['avg_steps']*model_df['n_tasks']).astype(int), model_df['avg_cost_task'].apply(lambda x: f"${x:.4f}")], align="left", font=dict(color='#1F2937', size=11))
    )])
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=260)
    return fig

# MT3: Resolve rate + Error bar
def draw_mt3(model_df):
    m_sort = model_df.sort_values('resolve_rate', ascending=False)
    y_vals, err_minus, err_plus = [], [], []
    for _, r in m_sort.iterrows():
        p = r['resolve_rate']
        n = r['n_tasks']
        low, high = wilson_score(p, n)
        y_vals.append(p)
        err_minus.append(p - low)
        err_plus.append(high - p)
    
    fig = go.Figure(go.Bar(
        x=m_sort['model'], y=y_vals,
        error_y=dict(type='data', array=err_plus, arrayminus=err_minus, visible=True, color='#6B7280'),
        marker_color=[COLORS.get(m, '#9CA3AF') for m in m_sort['model']],
        text=[f"{v*100:.1f}%" for v in y_vals], textposition='auto'
    ))
    fig = pbi_layout(fig, h=240)
    fig.update_layout(yaxis_tickformat='.0%', showlegend=False)
    return fig

# MT4: Cost per step distribution (Violin/Box trục log)
def draw_mt4(df):
    fig = go.Figure()
    for m in df['model'].unique():
        m_df = df[(df['model'] == m) & (df['cost_per_step'] > 0)]
        fig.add_trace(go.Box(x=m_df['cost_per_step'], y=[m]*len(m_df), orientation='h', marker_color=COLORS.get(m, '#9CA3AF'), name=m))
    
    # Highlight outliers > $100
    outliers = df[df['cost_per_step'] > 100]
    if len(outliers) > 0:
        fig.add_trace(go.Scatter(x=outliers['cost_per_step'], y=outliers['model'], mode='markers', marker=dict(color='#FF6B6B', size=10, symbol='x'), name='Spike >$100'))
    
    fig = pbi_layout(fig, h=260)
    fig.update_layout(xaxis_type='log', showlegend=False)
    return fig

# MT5: Cumulative cost vs step
def draw_mt5(df):
    fig = go.Figure()
    # Đường mean cho từng model
    mean_df = df.groupby(['model', 'step'])['cumulative_cost'].mean().reset_index()
    for m in mean_df['model'].unique():
        m_data = mean_df[mean_df['model'] == m]
        fig.add_trace(go.Scatter(x=m_data['step'], y=m_data['cumulative_cost'], mode='lines', line=dict(color=COLORS.get(m, '#9CA3AF'), width=3), name=f"{m} (Mean)"))
    fig = pbi_layout(fig, h=300)
    return fig

# CD1: Bẫy giá rẻ (Scatter)
def draw_cd1(task_df):
    task_df['avg_cost_per_step'] = task_df['final_cost'] / task_df['max_step']
    task_df['fail_rate'] = 1 - task_df['resolved_final']
    fig = px.scatter(task_df, x='avg_cost_per_step', y='fail_rate', size='max_step', color='model', color_discrete_map=COLORS, hover_data=['task_id'])
    fig = pbi_layout(fig, h=300)
    return fig

# CD2: Context bloat (Dual line)
def draw_cd2(df):
    # Lấy 3 tasks đại diện từ claude-sonnet (1 nhanh, 1 chậm, 1 fail)
    sonnet = df[df['model'] == 'claude-sonnet-4-6']
    tasks = sonnet['task_id'].unique()
    if len(tasks) >= 3: sample_tasks = tasks[:3]
    else: sample_tasks = df['task_id'].unique()[:3]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = ['#1F77B4', '#E15759', '#EDB120']
    for i, t in enumerate(sample_tasks):
        t_df = df[df['task_id'] == t]
        fig.add_trace(go.Scatter(x=t_df['step'], y=t_df['tokens']/1000, mode='lines', line=dict(color=colors[i], dash='dash'), name=f"Tok {t[-8:]}"), secondary_y=False)
        fig.add_trace(go.Scatter(x=t_df['step'], y=t_df['cost_per_step'], mode='lines', line=dict(color=colors[i]), name=f"Cost {t[-8:]}"), secondary_y=True)
    fig = pbi_layout(fig, h=260)
    fig.update_yaxes(title_text="Tokens (K)", secondary_y=False)
    fig.update_yaxes(title_text="Cost/step ($)", secondary_y=True)
    return fig

# CD3: Project difficulty confound (100% Stacked Bar)
def draw_cd3(task_df):
    proj_res = pd.crosstab(task_df['project'], task_df['resolved_final'], normalize='index') * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(y=proj_res.index, x=proj_res.get(1, [0]*len(proj_res)), orientation='h', name='Resolved', marker_color='#2CA089'))
    fig.add_trace(go.Bar(y=proj_res.index, x=proj_res.get(0, [0]*len(proj_res)), orientation='h', name='Failed', marker_color='#E15759'))
    fig = pbi_layout(fig, h=240)
    fig.update_layout(barmode='stack', showlegend=False)
    return fig

# CD4: Waterfall decomp
def draw_cd4(model_df):
    if len(model_df) < 2: return go.Figure()
    m_cheap = 'deepseek-v3.1'
    m_exp = 'claude-sonnet-4-6'
    if m_cheap not in model_df['model'].values: m_cheap = model_df['model'].values[0]
    if m_exp not in model_df['model'].values: m_exp = model_df['model'].values[-1]
    
    c_cheap = model_df[model_df['model'] == m_cheap]['avg_cost_task'].values[0]
    c_exp = model_df[model_df['model'] == m_exp]['avg_cost_task'].values[0]
    t_cheap = model_df[model_df['model'] == m_cheap]['avg_steps'].values[0]
    t_exp = model_df[model_df['model'] == m_exp]['avg_steps'].values[0]
    
    # Math logic cho decomp: C_exp - C_cheap = (t_exp * c/s_exp) - (t_cheap * c/s_cheap)
    delta_steps = (t_exp - t_cheap) * (c_cheap / t_cheap if t_cheap else 0)
    delta_price = c_exp - c_cheap - delta_steps
    
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["absolute", "relative", "relative", "total"],
        x=[m_cheap, "Tác động Steps", "Tác động Đơn giá", m_exp],
        y=[c_cheap, delta_steps, delta_price, c_exp],
        connector={"line":{"color":"#E3E6EA"}}
    ))
    fig = pbi_layout(fig, h=260)
    return fig

# CD5: Heatmap Error
def draw_cd5(task_df):
    pivot = task_df.groupby(['model', 'project'])['resolved_final'].mean().unstack().fillna(1)
    error_rate = 1 - pivot
    fig = go.Figure(go.Heatmap(
        z=error_rate.values, x=error_rate.columns, y=error_rate.index,
        colorscale='Reds', zmin=0, zmax=1,
        text=np.vectorize(lambda x: f"{x*100:.0f}%")(error_rate.values), texttemplate="%{text}", showscale=False
    ))
    fig = pbi_layout(fig, h=240)
    return fig

# DB1: Survival
def draw_db1(task_df):
    fig = go.Figure()
    for m in task_df['model'].unique():
        m_df = task_df[task_df['model'] == m]
        steps = np.sort(m_df['max_step'].unique())
        surv = [1 - (m_df[m_df['max_step'] <= s]['resolved_final'].sum() / len(m_df)) for s in steps]
        fig.add_trace(go.Scatter(x=steps, y=surv, mode='lines', line_shape='hv', name=m, line=dict(color=COLORS.get(m, '#9CA3AF'))))
    fig.add_vline(x=30, line_dash="dash", line_color="#E15759")
    fig.add_vrect(x0=30, x1=35, fillcolor="#FF6B6B", opacity=0.1, line_width=0)
    fig = pbi_layout(fig, h=300)
    fig.update_yaxes(tickformat='.0%')
    return fig

# DB2: Scatter log-log
def draw_db2(df, task_df):
    cost_10 = df[df['step'] == 10].groupby('task_id')['cumulative_cost'].last().reset_index()
    merged = pd.merge(cost_10, task_df[['task_id', 'final_cost']], on='task_id')
    fig = px.scatter(merged, x='cumulative_cost', y='final_cost', log_x=True, log_y=True)
    fig.add_shape(type="line", x0=merged['cumulative_cost'].min(), y0=merged['cumulative_cost'].min(), x1=merged['final_cost'].max(), y1=merged['final_cost'].max(), line=dict(color="gray", dash="dash"))
    fig = pbi_layout(fig, h=300)
    return fig

# DB3: Box plot feature 5 step đầu
def draw_db3(df, task_df):
    step5 = df[df['step'] <= 5].groupby('task_id')['cost'].mean().reset_index()
    merged = pd.merge(step5, task_df[['task_id', 'resolved_final']], on='task_id')
    merged['resolved_label'] = merged['resolved_final'].map({1: 'Resolved', 0: 'Failed'})
    fig = go.Figure()
    fig.add_trace(go.Box(x=merged[merged['resolved_final']==1]['cost'], y=merged[merged['resolved_final']==1]['resolved_label'], marker_color='#2CA089', name='Resolved'))
    fig.add_trace(go.Box(x=merged[merged['resolved_final']==0]['cost'], y=merged[merged['resolved_final']==0]['resolved_label'], marker_color='#E15759', name='Failed'))
    fig = pbi_layout(fig, h=260)
    return fig

# DB4: Tokens
def draw_db4(df):
    mean_tok = df.groupby(['model', 'step'])['tokens'].mean().reset_index()
    fig = go.Figure()
    for m in mean_tok['model'].unique():
        m_data = mean_tok[mean_tok['model'] == m]
        fig.add_trace(go.Scatter(x=m_data['step'], y=m_data['tokens'], mode='lines', line=dict(color=COLORS.get(m, '#9CA3AF')), name=m))
    fig = pbi_layout(fig, h=300)
    return fig

# KN1: Circuit breaker at 30
def draw_kn1(df, task_df):
    cut_df = df[df['step'] <= 30]
    cut_task = cut_df.groupby('task_id').agg(c_cost=('cumulative_cost', 'max'), c_res=('resolved', 'last')).reset_index()
    tot_cost_before = task_df['final_cost'].sum()
    tot_res_before = task_df['resolved_final'].sum()
    tot_cost_after = cut_task['c_cost'].sum()
    tot_res_after = cut_task['c_res'].sum()
    
    fig = go.Figure(data=[
        go.Bar(name='Cost', x=['Before', 'After (Cut @ 30)'], y=[tot_cost_before, tot_cost_after], marker_color='#1F77B4'),
    ])
    fig = pbi_layout(fig, h=260)
    return fig

# KN2: Spike Guard Waterfall
def draw_kn2(df):
    tot_cost = df['cost'].sum()
    spikes = df[df['duration'] > 300]['cost'].sum()
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["absolute", "relative", "total"],
        x=["Total Cost", "Block >300s", "Net Cost"], y=[tot_cost, -spikes, tot_cost - spikes],
        decreasing={"marker":{"color":"#2CA089"}}
    ))
    fig = pbi_layout(fig, h=260)
    return fig

# KN3: Routing (Bubble Quadrant)
def draw_kn3(model_df):
    fig = px.scatter(model_df, x='avg_cost_task', y='resolve_rate', size='n_tasks', color='model', color_discrete_map=COLORS, text='model')
    fig.add_vline(x=model_df['avg_cost_task'].median(), line_dash="dash", line_color="#9CA3AF")
    fig.add_hline(y=model_df['resolve_rate'].median(), line_dash="dash", line_color="#9CA3AF")
    fig = pbi_layout(fig, h=300)
    return fig

# KN4: ROI Waterfall
def draw_kn4(k_waste, df):
    spikes = df[df['duration'] > 300]['cost'].sum()
    cut30 = task_df['final_cost'].sum() - df[df['step'] <= 30].groupby('task_id')['cumulative_cost'].max().sum()
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["absolute", "relative", "relative", "total"],
        x=["Lãng phí gốc", "Circuit Breaker", "Spike Guard", "Lãng phí còn lại"],
        y=[k_waste, -cut30, -spikes, k_waste - cut30 - spikes],
        decreasing={"marker":{"color":"#2CA089"}}
    ))
    fig = pbi_layout(fig, h=260)
    return fig


# =============================================================================
# C. LẮP RÁP BENTO GRID & STORY
# =============================================================================

# §1 MÔ TẢ
st.markdown('<div class="section-title">§1 MÔ TẢ (Điều gì đã xảy ra?)</div>', unsafe_allow_html=True)
c1_1, c1_2 = st.columns([8, 4])
with c1_1:
    st.markdown('<div class="bento-card"><div class="panel-title">Ngân sách phân bổ theo model</div>', unsafe_allow_html=True)
    st.plotly_chart(draw_mt1(model_df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c1_2:
    st.markdown('<div class="bento-card"><div class="panel-title">Tasks, steps, cost/task</div>', unsafe_allow_html=True)
    st.plotly_chart(draw_mt2(model_df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

c1_3, c1_4, c1_5 = st.columns([7, 5, 12]) # Fixed to row wrappers
col1_a, col1_b = st.columns([7, 5])
with col1_a:
    st.markdown('<div class="bento-card"><div class="panel-title">Chi phí tích lũy điển hình (MT5)</div>', unsafe_allow_html=True)
    st.plotly_chart(draw_mt5(df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col1_b:
    st.markdown('<div class="bento-card"><div class="panel-title">Resolve rate (MT3)</div>', unsafe_allow_html=True)
    st.plotly_chart(draw_mt3(model_df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

col1_c, col1_d = st.columns([7, 5])
with col1_c:
    st.markdown('<div class="bento-card"><div class="panel-title">Phân phối cost/step (MT4)</div>', unsafe_allow_html=True)
    st.plotly_chart(draw_mt4(df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col1_d:
    st.markdown('<div class="bento-card story-box">DeepSeek & MiniMax resolve 100% SWE-bench với <$0.01/task. Claude-sonnet đắt gấp 10-15× nhưng resolve tương đương. Claude-opus thất bại 67% wildclaw.</div>', unsafe_allow_html=True)


# §2 CHẨN ĐOÁN
st.markdown('<div class="section-title">§2 CHẨN ĐOÁN (Tại sao xảy ra?)</div>', unsafe_allow_html=True)
c2_1, c2_2 = st.columns([7, 5])
with c2_1:
    st.markdown('<div class="bento-card"><div class="panel-title">Bẫy giá rẻ (CD1)</div>', unsafe_allow_html=True)
    st.plotly_chart(draw_cd1(task_df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c2_2:
    st.markdown('<div class="bento-card"><div class="panel-title">Context bloat (CD2)</div>', unsafe_allow_html=True)
    st.plotly_chart(draw_cd2(df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

c2_3, c2_4 = st.columns([7, 5])
with c2_3:
    st.markdown('<div class="bento-card"><div class="panel-title">Phân rã chi phí (CD4)</div>', unsafe_allow_html=True)
    st.plotly_chart(draw_cd4(model_df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c2_4:
    st.markdown('<div class="bento-card"><div class="panel-title">Nghịch lý độ khó (CD3)</div>', unsafe_allow_html=True)
    st.plotly_chart(draw_cd3(task_df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

c2_5, c2_6 = st.columns([6, 6])
with c2_5:
    st.markdown('<div class="bento-card"><div class="panel-title">Lỗi theo model x project (CD5)</div>', unsafe_allow_html=True)
    st.plotly_chart(draw_cd5(task_df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c2_6:
    st.markdown('<div class="bento-card story-box">5 spike ~$301 (timeout build) chiếm phần lớn ngân sách. Context phình từ 10K→80K tokens khiến cost/step tăng 2-3× ở giai đoạn cuối.</div>', unsafe_allow_html=True)


# §3 DỰ ĐOÁN
st.markdown('<div class="section-title">§3 DỰ ĐOÁN (Có thể lượng hóa trước gì?)</div>', unsafe_allow_html=True)
c3_1, c3_2 = st.columns([7, 5])
with c3_1:
    st.markdown('<div class="bento-card"><div class="panel-title">Xác suất resolve theo step (DB1)</div>', unsafe_allow_html=True)
    st.plotly_chart(draw_db1(task_df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c3_2:
    st.markdown('<div class="bento-card"><div class="panel-title">Tokens phình theo step (DB4)</div>', unsafe_allow_html=True)
    st.plotly_chart(draw_db4(df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

c3_3, c3_4 = st.columns([6, 6])
with c3_3:
    st.markdown('<div class="bento-card"><div class="panel-title">Dự đoán cost (DB2)</div>', unsafe_allow_html=True)
    st.plotly_chart(draw_db2(df, task_df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c3_4:
    st.markdown('<div class="bento-card"><div class="panel-title">Nguy cơ fail sớm (DB3)</div>', unsafe_allow_html=True)
    st.plotly_chart(draw_db3(df, task_df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="bento-card story-box" style="margin-top: -16px;">Sau step 30, xác suất resolve mới tăng &lt;5% nhưng chi phí tăng mạnh. Điểm dừng tối ưu: 25-30 steps cho SWE-bench.</div>', unsafe_allow_html=True)


# §4 ĐỀ XUẤT
st.markdown('<div class="section-title">§4 ĐỀ XUẤT (Nên làm gì?)</div>', unsafe_allow_html=True)
c4_1, c4_2 = st.columns([7, 5])
with c4_1:
    st.markdown('<div class="bento-card"><div class="panel-title">Model Routing (KN3)</div>', unsafe_allow_html=True)
    st.plotly_chart(draw_kn3(model_df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c4_2:
    st.markdown('<div class="bento-card"><div class="panel-title">Circuit Breaker (KN1)</div>', unsafe_allow_html=True)
    st.plotly_chart(draw_kn1(df, task_df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

c4_3, c4_4 = st.columns([5, 7])
with c4_3:
    st.markdown('<div class="bento-card"><div class="panel-title">Chặn Spike (KN2)</div>', unsafe_allow_html=True)
    st.plotly_chart(draw_kn2(df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c4_4:
    st.markdown('<div class="bento-card"><div class="panel-title">ROI Tổng (KN4)</div>', unsafe_allow_html=True)
    st.plotly_chart(draw_kn4(k_waste, df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="exec-summary">
    <div class="exec-title">TÓM TẮT ĐIỀU HÀNH</div>
    <ul>
        <li><b>ĐỀ XUẤT 1 — CIRCUIT BREAKER (P0, 1 tuần):</b> Dừng task khi đạt 30 steps mà flag vẫn 0. Cắt giảm lãng phí, ảnh hưởng <5% resolve rate.</li>
        <li><b>ĐỀ XUẤT 2 — SPIKE GUARD (P0, 3 ngày):</b> Cảnh báo & kill step khi duration >300s. Tránh mất $1,505 (nguyên nhân timeout build).</li>
        <li><b>ĐỀ XUẤT 3 — MODEL ROUTING (P1, 2 tuần):</b> Giao SWE-bench cho MiniMax ($0.006/task, 100% resolve). Giữ Claude-sonnet cho scikit-learn. Giảm Claude-opus cho wildclaw.</li>
        <li><b>ĐỀ XUẤT 4 — CONTEXT WINDOW (P1, 3 tuần):</b> Áp dụng summarization khi tokens >40K.</li>
    </ul>
</div>
<div style="font-size:11px; color:#9CA3AF; text-align:center;">
    * Caveat: cost & tokens là per-step & tích lũy; spike ~$301 = timeout môi trường; flag≠resolved ở wildclaw.<br>
    Source: processed_agentic_traces.csv
</div>
""", unsafe_allow_html=True)
