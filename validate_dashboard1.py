import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

st.set_page_config(page_title="AI Agent Diagnostic Intelligence", layout="wide", page_icon="🧠", initial_sidebar_state="collapsed")

# =============================================================================
# CSS - GLASSMORPHISM BENTO GRID (Old UX/UI)
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

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

    .bento-grid {
        display: grid;
        grid-template-columns: repeat(12, 1fr);
        gap: 16px;
        padding: 0 8px;
        max-width: 1600px;
        margin: 0 auto;
    }
    .col-span-2  { grid-column: span 2; }
    .col-span-3  { grid-column: span 3; }
    .col-span-12 { grid-column: span 12; }
    
    .bento-card {
        background: #ffffff;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(14, 165, 233, 0.12);
        border-radius: 20px;
        padding: 24px;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 12px -2px rgba(14, 165, 233, 0.1), 0 1px 4px rgba(0,0,0,0.04);
        height: 100%;
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

    .kpi-card { display: flex; flex-direction: column; justify-content: space-between; min-height: 160px; }
    @media (max-width: 1100px) { .kpi-card { grid-column: span 4 !important; } }
    @media (max-width: 768px) { .kpi-card { grid-column: span 6 !important; } }
    
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

    .insight-highlight {
        background: rgba(14,165,233,0.07);
        border-left: 3px solid #0ea5e9;
        padding: 16px 20px;
        border-radius: 0 8px 8px 0;
        margin: 0;
        font-size: 0.95rem; line-height: 1.6; color: #334155;
        height: 100%;
    }
    .insight-highlight strong { color: #0284c7; }

    .group-header {
        background: linear-gradient(90deg, rgba(14,165,233,0.1), transparent);
        border-left: 4px solid #0ea5e9;
        padding: 10px 18px;
        border-radius: 0 10px 10px 0;
        margin: 32px 0 16px 0;
        font-size: 1rem;
        font-weight: 700;
        color: #0284c7;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .exec-list li { margin-bottom: 12px; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# A. DATA PIPELINE (Chuẩn 9 cột)
# =============================================================================
@st.cache_data
def load_data(filepath='processed_agentic_traces.csv'):
    df = pd.read_csv(filepath, header=None)
    df.columns = ['task_id', 'model', 'duration', 'cost', 'resolved', 'flag', 'tokens', 'step', 'cumulative_cost']
    
    for col in ['duration', 'cost', 'resolved', 'flag', 'tokens', 'step', 'cumulative_cost']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    parts = df['task_id'].str.split('__', expand=True)
    df['benchmark'] = parts[0]
    df['project'] = parts[1]
    df['issue'] = parts[2]
    
    df = df.sort_values(['task_id', 'step'])
    df['tokens_per_step'] = df.groupby('task_id')['tokens'].diff().fillna(df['tokens'])
    df['cost_per_step'] = df['cost']
    
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
    
    model_df = task_df.groupby('model').agg(
        n_tasks=('task_id', 'count'),
        resolve_rate=('resolved_final', 'mean'),
        avg_cost_task=('final_cost', 'mean'),
        avg_steps=('max_step', 'mean'),
        avg_duration_task=('total_duration', 'mean'),
        total_cost=('final_cost', 'sum')
    ).reset_index()
    
    wasted = task_df[task_df['resolved_final'] == 0].groupby('model')['final_cost'].sum().reset_index()
    wasted.rename(columns={'final_cost': 'wasted_cost'}, inplace=True)
    model_df = model_df.merge(wasted, on='model', how='left').fillna(0)
    
    return df, task_df, model_df

df, task_df, model_df = load_data()


# =============================================================================
# B. PLOTLY STYLING & FUNCTIONS
# =============================================================================
COLORS = {
    'claude-sonnet-4-6': '#1F77B4',
    'claude-opus-4-6': '#2CA089',
    'deepseek-v3.1': '#EDB120',
    'minimax-m2.5': '#E15759',
}

# UX/UI Light theme từ validate_dashboard1 cũ
PLOT_CFG = dict(
    template='plotly_white',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(248,250,252,0.6)',
    font=dict(family='Inter', color='#334155', size=11),
    title=dict(font=dict(size=14, color='#0f172a', weight='bold')),
    legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='rgba(14,165,233,0.15)',
                borderwidth=1, font=dict(color='#334155', size=11)),
    margin=dict(t=40, l=30, r=20, b=30),
    hovermode='closest'
)

def sf(fig, title="", height=280):
    fig.update_layout(**PLOT_CFG, height=height, title_text=title)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(14,165,233,0.1)',
                     zeroline=False, tickfont=dict(color='#64748b', size=11))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(14,165,233,0.1)',
                     zeroline=False, tickfont=dict(color='#64748b', size=11))
    fig.update_traces(selector=dict(type='bar'), marker_cornerradius=2)
    return fig

def wilson_score(p, n, z=1.96):
    if n == 0: return 0,0
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    spread = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return center - spread, center + spread

# --- MT Panels ---
def draw_mt1():
    m_sort = model_df.sort_values('total_cost')
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "pie"}]], column_widths=[0.6, 0.4])
    cols = [COLORS.get(m, '#9ca3af') for m in m_sort['model']]
    fig.add_trace(go.Bar(y=m_sort['model'], x=m_sort['total_cost'], orientation='h', marker_color=cols, text=m_sort['total_cost'].apply(lambda x: f"${x:.2f}"), textposition='auto'), row=1, col=1)
    fig.add_trace(go.Pie(labels=model_df['model'], values=model_df['total_cost'], hole=0.6, marker_colors=[COLORS.get(m, '#9ca3af') for m in model_df['model']], textinfo='none'), row=1, col=2)
    fig = sf(fig, "MT1. Ngân sách phân bổ", 280)
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
    return fig

def draw_mt2():
    fig = go.Figure(data=[go.Table(
        header=dict(values=["Model", "Tasks", "Steps", "$/Task"], align="left", fill_color='rgba(14,165,233,0.1)', font=dict(color='#0f172a', size=11)),
        cells=dict(values=[model_df['model'], model_df['n_tasks'], (model_df['avg_steps']*model_df['n_tasks']).astype(int), model_df['avg_cost_task'].apply(lambda x: f"${x:.4f}")], align="left", font=dict(color='#334155', size=11))
    )])
    fig.update_layout(margin=dict(l=0,r=0,t=30,b=0), height=280, title_text="MT2. Tóm tắt Model", title_font=dict(size=14, color='#0f172a', weight='bold'))
    return fig

def draw_mt3():
    m_sort = model_df.sort_values('resolve_rate', ascending=False)
    y_vals, err_minus, err_plus = [], [], []
    for _, r in m_sort.iterrows():
        p = r['resolve_rate']
        low, high = wilson_score(p, r['n_tasks'])
        y_vals.append(p)
        err_minus.append(p - low)
        err_plus.append(high - p)
    fig = go.Figure(go.Bar(
        x=m_sort['model'], y=y_vals,
        error_y=dict(type='data', array=err_plus, arrayminus=err_minus, visible=True, color='#64748b'),
        marker_color=[COLORS.get(m, '#9ca3af') for m in m_sort['model']],
        text=[f"{v*100:.1f}%" for v in y_vals], textposition='auto'
    ))
    fig = sf(fig, "MT3. Resolve rate (95% CI)", 280)
    fig.update_layout(yaxis_tickformat='.0%', showlegend=False)
    return fig

def draw_mt4():
    fig = go.Figure()
    for m in df['model'].unique():
        m_df = df[(df['model'] == m) & (df['cost_per_step'] > 0)]
        fig.add_trace(go.Box(x=m_df['cost_per_step'], y=[m]*len(m_df), orientation='h', marker_color=COLORS.get(m, '#9ca3af'), name=m))
    outliers = df[df['cost_per_step'] > 100]
    if len(outliers) > 0:
        fig.add_trace(go.Scatter(x=outliers['cost_per_step'], y=outliers['model'], mode='markers', marker=dict(color='#f43f5e', size=10, symbol='x'), name='Spike >$100'))
    fig = sf(fig, "MT4. Phân phối cost/step", 280)
    fig.update_layout(xaxis_type='log', showlegend=False)
    return fig

def draw_mt5():
    fig = go.Figure()
    mean_df = df.groupby(['model', 'step'])['cumulative_cost'].mean().reset_index()
    for m in mean_df['model'].unique():
        m_data = mean_df[mean_df['model'] == m]
        fig.add_trace(go.Scatter(x=m_data['step'], y=m_data['cumulative_cost'], mode='lines', line=dict(color=COLORS.get(m, '#9ca3af'), width=3), name=f"{m} (Mean)"))
    fig = sf(fig, "MT5. Chi phí tích lũy điển hình", 320)
    return fig

# --- CD Panels ---
def draw_cd1():
    task_df['avg_cost_per_step'] = task_df['final_cost'] / task_df['max_step']
    task_df['fail_rate'] = 1 - task_df['resolved_final']
    fig = px.scatter(task_df, x='avg_cost_per_step', y='fail_rate', size='max_step', color='model', color_discrete_map=COLORS, hover_data=['task_id'])
    return sf(fig, "CD1. Bẫy giá rẻ", 320)

def draw_cd2():
    sonnet = df[df['model'] == 'claude-sonnet-4-6']
    tasks = sonnet['task_id'].unique()
    sample_tasks = tasks[:3] if len(tasks) >= 3 else df['task_id'].unique()[:3]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = ['#10b981', '#38bdf8', '#f59e0b']
    for i, t in enumerate(sample_tasks):
        t_df = df[df['task_id'] == t]
        fig.add_trace(go.Scatter(x=t_df['step'], y=t_df['tokens']/1000, mode='lines', line=dict(color=colors[i], dash='dash'), name=f"Tok {t[-8:]}"), secondary_y=False)
        fig.add_trace(go.Scatter(x=t_df['step'], y=t_df['cost_per_step'], mode='lines', line=dict(color=colors[i]), name=f"Cost {t[-8:]}"), secondary_y=True)
    fig = sf(fig, "CD2. Context bloat", 280)
    fig.update_yaxes(title_text="Tokens (K)", secondary_y=False)
    fig.update_yaxes(title_text="Cost/step ($)", secondary_y=True)
    return fig

def draw_cd3():
    proj_res = pd.crosstab(task_df['project'], task_df['resolved_final'], normalize='index') * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(y=proj_res.index, x=proj_res.get(1, [0]*len(proj_res)), orientation='h', name='Resolved', marker_color='#10b981'))
    fig.add_trace(go.Bar(y=proj_res.index, x=proj_res.get(0, [0]*len(proj_res)), orientation='h', name='Failed', marker_color='#f43f5e'))
    fig = sf(fig, "CD3. Độ khó Project", 280)
    fig.update_layout(barmode='stack', showlegend=False)
    return fig

def draw_cd4():
    m_cheap, m_exp = 'deepseek-v3.1', 'claude-sonnet-4-6'
    if m_cheap not in model_df['model'].values: m_cheap = model_df['model'].values[0]
    if m_exp not in model_df['model'].values: m_exp = model_df['model'].values[-1]
    
    c_cheap = model_df[model_df['model'] == m_cheap]['avg_cost_task'].values[0]
    c_exp = model_df[model_df['model'] == m_exp]['avg_cost_task'].values[0]
    t_cheap = model_df[model_df['model'] == m_cheap]['avg_steps'].values[0]
    t_exp = model_df[model_df['model'] == m_exp]['avg_steps'].values[0]
    
    delta_steps = (t_exp - t_cheap) * (c_cheap / t_cheap if t_cheap else 0)
    delta_price = c_exp - c_cheap - delta_steps
    
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["absolute", "relative", "relative", "total"],
        x=[m_cheap, "Steps", "Đơn giá", m_exp],
        y=[c_cheap, delta_steps, delta_price, c_exp],
        connector={"line":{"color":"#cbd5e1"}}
    ))
    return sf(fig, "CD4. Phân rã chi phí", 280)

def draw_cd5():
    pivot = task_df.groupby(['model', 'project'])['resolved_final'].mean().unstack().fillna(1)
    error_rate = 1 - pivot
    fig = go.Figure(go.Heatmap(
        z=error_rate.values, x=error_rate.columns, y=error_rate.index,
        colorscale='Reds', zmin=0, zmax=1,
        text=np.vectorize(lambda x: f"{x*100:.0f}%")(error_rate.values), texttemplate="%{text}", showscale=False
    ))
    return sf(fig, "CD5. Heatmap Lỗi", 280)

# --- DB Panels ---
def draw_db1():
    fig = go.Figure()
    for m in task_df['model'].unique():
        m_df = task_df[task_df['model'] == m]
        steps = np.sort(m_df['max_step'].unique())
        surv = [1 - (m_df[m_df['max_step'] <= s]['resolved_final'].sum() / len(m_df)) for s in steps]
        fig.add_trace(go.Scatter(x=steps, y=surv, mode='lines', line_shape='hv', name=m, line=dict(color=COLORS.get(m, '#9ca3af'))))
    fig.add_vline(x=30, line_dash="dash", line_color="#f43f5e")
    fig.add_vrect(x0=30, x1=35, fillcolor="#f43f5e", opacity=0.05, line_width=0)
    fig = sf(fig, "DB1. Survival Curve (Resolve)", 320)
    fig.update_yaxes(tickformat='.0%')
    return fig

def draw_db2():
    cost_10 = df[df['step'] == 10].groupby('task_id')['cumulative_cost'].last().reset_index()
    merged = pd.merge(cost_10, task_df[['task_id', 'final_cost']], on='task_id')
    fig = px.scatter(merged, x='cumulative_cost', y='final_cost', log_x=True, log_y=True)
    fig.add_shape(type="line", x0=merged['cumulative_cost'].min(), y0=merged['cumulative_cost'].min(), x1=merged['final_cost'].max(), y1=merged['final_cost'].max(), line=dict(color="gray", dash="dash"))
    return sf(fig, "DB2. Cost @10 vs Final Cost", 320)

def draw_db3():
    step5 = df[df['step'] <= 5].groupby('task_id')['cost'].mean().reset_index()
    merged = pd.merge(step5, task_df[['task_id', 'resolved_final']], on='task_id')
    merged['resolved_label'] = merged['resolved_final'].map({1: 'Resolved', 0: 'Failed'})
    fig = go.Figure()
    fig.add_trace(go.Box(x=merged[merged['resolved_final']==1]['cost'], y=merged[merged['resolved_final']==1]['resolved_label'], marker_color='#10b981', name='Resolved'))
    fig.add_trace(go.Box(x=merged[merged['resolved_final']==0]['cost'], y=merged[merged['resolved_final']==0]['resolved_label'], marker_color='#f43f5e', name='Failed'))
    return sf(fig, "DB3. Feature 5 step đầu", 320)

def draw_db4():
    mean_tok = df.groupby(['model', 'step'])['tokens'].mean().reset_index()
    fig = go.Figure()
    for m in mean_tok['model'].unique():
        m_data = mean_tok[mean_tok['model'] == m]
        fig.add_trace(go.Scatter(x=m_data['step'], y=m_data['tokens'], mode='lines', line=dict(color=COLORS.get(m, '#9ca3af')), name=m))
    return sf(fig, "DB4. Tốc độ phình tokens", 320)

# --- KN Panels ---
def draw_kn1():
    cut_df = df[df['step'] <= 30]
    cut_task = cut_df.groupby('task_id').agg(c_cost=('cumulative_cost', 'max'), c_res=('resolved', 'last')).reset_index()
    tot_cost_before = task_df['final_cost'].sum()
    tot_cost_after = cut_task['c_cost'].sum()
    fig = go.Figure(data=[
        go.Bar(name='Cost', x=['Gốc', 'Cắt @ 30'], y=[tot_cost_before, tot_cost_after], marker_color='#38bdf8'),
    ])
    return sf(fig, "KN1. Circuit Breaker @30", 320)

def draw_kn2():
    tot_cost = df['cost'].sum()
    spikes = df[df['duration'] > 300]['cost'].sum()
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["absolute", "relative", "total"],
        x=["Total Cost", "Block >300s", "Net Cost"], y=[tot_cost, -spikes, tot_cost - spikes],
        decreasing={"marker":{"color":"#10b981"}}
    ))
    return sf(fig, "KN2. Spike Guard Waterfall", 280)

def draw_kn3():
    fig = px.scatter(model_df, x='avg_cost_task', y='resolve_rate', size='n_tasks', color='model', color_discrete_map=COLORS, text='model')
    fig.add_vline(x=model_df['avg_cost_task'].median(), line_dash="dash", line_color="#cbd5e1")
    fig.add_hline(y=model_df['resolve_rate'].median(), line_dash="dash", line_color="#cbd5e1")
    return sf(fig, "KN3. Model Routing", 320)

def draw_kn4(k_waste):
    spikes = df[df['duration'] > 300]['cost'].sum()
    cut30 = task_df['final_cost'].sum() - df[df['step'] <= 30].groupby('task_id')['cumulative_cost'].max().sum()
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["absolute", "relative", "relative", "total"],
        x=["Wasted gốc", "KN1 (Breaker)", "KN2 (Spike)", "Wasted ròng"],
        y=[k_waste, -cut30, -spikes, k_waste - cut30 - spikes],
        decreasing={"marker":{"color":"#10b981"}}
    ))
    return sf(fig, "KN4. ROI Net", 280)


# =============================================================================
# C. BENTO GRID LAYOUT & STORY BOXES
# =============================================================================
k_tot_cost = task_df['final_cost'].sum()
k_tasks = task_df['task_id'].nunique()
k_steps = len(df)
k_res_rate = task_df['resolved_final'].mean() * 100
k_cost_res = task_df[task_df['resolved_final'] == 1]['final_cost'].sum() / max(1, task_df['resolved_final'].sum())
k_waste = task_df[task_df['resolved_final'] == 0]['final_cost'].sum()
k_waste_pct = k_waste / k_tot_cost * 100

st.markdown(f"""
<div class="bento-grid">
    <div class="bento-header">
        <div class="header-badge"><span>🧠</span> Executive Report • Q2-Q3 2026</div>
        <h1 class="header-title">AI Agent Diagnostic Intelligence</h1>
        <p class="header-subtitle">Báo cáo phân tích chuyên sâu chi phí & hiệu năng hệ thống AI Agent dựa trên dữ liệu Telemetry thực tế. Áp dụng Framework 4 Cấp Độ Phân Tích.</p>
        <div class="header-meta">
            <div class="meta-item"><span class="meta-dot"></span> Live Data</div>
            <div class="meta-item">📊 {k_tasks:,} Tasks • {k_steps:,} Steps</div>
            <div class="meta-item">🤖 4 Models</div>
        </div>
    </div>
</div>
<br>
<div class="bento-grid">
    <div class="bento-card col-span-2 kpi-card">
        <div><div class="kpi-icon kpi-icon-blue">💰</div><div class="kpi-value">${k_tot_cost:.2f}</div><div class="kpi-label">Tổng Ngân Sách</div></div>
        <div class="kpi-trend trend-neutral">4 models · 2 benchmarks</div>
    </div>
    <div class="bento-card col-span-2 kpi-card">
        <div><div class="kpi-icon kpi-icon-purple">📋</div><div class="kpi-value">{k_tasks}</div><div class="kpi-label">Tasks</div></div>
        <div class="kpi-trend trend-neutral">swebench/wildclaw</div>
    </div>
    <div class="bento-card col-span-2 kpi-card">
        <div><div class="kpi-icon kpi-icon-amber">⚙️</div><div class="kpi-value">{k_steps:,}</div><div class="kpi-label">Steps</div></div>
        <div class="kpi-trend trend-neutral">~{int(k_steps/k_tasks)} steps/task</div>
    </div>
    <div class="bento-card col-span-2 kpi-card">
        <div><div class="kpi-icon kpi-icon-green">🎯</div><div class="kpi-value" style="color: {'#15803d' if k_res_rate>80 else '#be123c' if k_res_rate<50 else '#0f172a'}">{k_res_rate:.1f}%</div><div class="kpi-label">Resolve Rate</div></div>
        <div class="kpi-trend trend-up">Tỉ lệ fix thành công</div>
    </div>
    <div class="bento-card col-span-2 kpi-card">
        <div><div class="kpi-icon kpi-icon-blue">📈</div><div class="kpi-value">${k_cost_res:.4f}</div><div class="kpi-label">Cost / Resolved Task</div></div>
        <div class="kpi-trend trend-neutral">Minimax ≈ $0.006</div>
    </div>
    <div class="bento-card col-span-2 kpi-card" style="border-left: 4px solid #f43f5e;">
        <div><div class="kpi-icon kpi-icon-rose">🗑️</div><div class="kpi-value" style="color:#f43f5e">${k_waste:.4f}</div><div class="kpi-label">Wasted Cost</div></div>
        <div class="kpi-trend trend-down">≈{k_waste_pct:.1f}% tổng chi phí</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Helper function render bento block
def bento_box(fig, height_css="100%"):
    # Sử dụng tiêu đề biểu đồ làm key để đảm bảo tính duy nhất và ổn định giữa các lần rerun
    title_text = fig.layout.title.text if (fig.layout and fig.layout.title and fig.layout.title.text) else str(id(fig))
    
    st.markdown(f'<div class="bento-card" style="height: {height_css};">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, key=title_text)
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="group-header">§1 MÔ TẢ (Điều gì đã xảy ra?)</div>', unsafe_allow_html=True)
r1c1, r1c2 = st.columns([8, 4])
with r1c1: bento_box(draw_mt1())
with r1c2: bento_box(draw_mt2())

r1c3, r1c4 = st.columns([7, 5])
with r1c3: bento_box(draw_mt5())
with r1c4: bento_box(draw_mt3())

r1c5, r1c6 = st.columns([7, 5])
with r1c5: bento_box(draw_mt4())
with r1c6: 
    st.markdown("""
    <div class="bento-card" style="display:flex; align-items:center;">
        <div class="insight-highlight">
            <strong>Insight:</strong> DeepSeek & MiniMax resolve 100% SWE-bench với <$0.01/task. Claude-sonnet đắt gấp 10-15× nhưng resolve tương đương. Claude-opus thất bại 67% wildclaw.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="group-header">§2 CHẨN ĐOÁN (Tại sao xảy ra?)</div>', unsafe_allow_html=True)
r2c1, r2c2 = st.columns([7, 5])
with r2c1: bento_box(draw_cd1())
with r2c2: bento_box(draw_cd2())

r2c3, r2c4 = st.columns([7, 5])
with r2c3: bento_box(draw_cd4())
with r2c4: bento_box(draw_cd3())

r2c5, r2c6 = st.columns([6, 6])
with r2c5: bento_box(draw_cd5())
with r2c6:
    st.markdown("""
    <div class="bento-card" style="display:flex; align-items:center;">
        <div class="insight-highlight">
            <strong>Insight:</strong> 5 spike ~$301 (timeout build) chiếm phần lớn ngân sách. Context phình từ 10K→80K tokens khiến cost/step tăng 2-3× ở giai đoạn cuối.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="group-header">§3 DỰ ĐOÁN (Có thể lượng hóa trước gì?)</div>', unsafe_allow_html=True)
r3c1, r3c2 = st.columns([7, 5])
with r3c1: bento_box(draw_db1())
with r3c2: bento_box(draw_db4())

r3c3, r3c4 = st.columns([6, 6])
with r3c3: bento_box(draw_db2())
with r3c4: bento_box(draw_db3())

st.markdown("""
<div class="bento-grid">
    <div class="col-span-12">
        <div class="insight-highlight" style="border-radius: 8px; border-left: 4px solid #10b981; background: rgba(16, 185, 129, 0.05);">
            <strong>Kết luận:</strong> Sau step 30, xác suất resolve mới tăng &lt;5% nhưng chi phí tăng mạnh. Điểm dừng tối ưu: 25-30 steps cho SWE-bench.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="group-header">§4 ĐỀ XUẤT (Nên làm gì, tiết kiệm bao nhiêu?)</div>', unsafe_allow_html=True)
r4c1, r4c2 = st.columns([7, 5])
with r4c1: bento_box(draw_kn3())
with r4c2: bento_box(draw_kn1())

r4c3, r4c4 = st.columns([5, 7])
with r4c3: bento_box(draw_kn2())
with r4c4: bento_box(draw_kn4(k_waste))

st.markdown("""
<div class="bento-grid" style="margin-top: 24px;">
    <div class="bento-card col-span-12">
        <h3 style="color:#0f172a; margin-top:0;">TÓM TẮT ĐIỀU HÀNH</h3>
        <ul class="exec-list" style="color:#334155;">
            <li><strong>ĐỀ XUẤT 1 — CIRCUIT BREAKER (P0, 1 tuần):</strong> Dừng task khi đạt 30 steps mà flag vẫn 0. Giảm lãng phí, ảnh hưởng &lt;5% resolve rate.</li>
            <li><strong>ĐỀ XUẤT 2 — SPIKE GUARD (P0, 3 ngày):</strong> Cảnh báo & kill step khi duration >300s. Tránh mất $1,505 (nguyên nhân timeout build, không phải phí suy luận).</li>
            <li><strong>ĐỀ XUẤT 3 — MODEL ROUTING (P1, 2 tuần):</strong> Giao SWE-bench cho MiniMax ($0.006/task, 100% resolve). Giữ Claude-sonnet cho scikit-learn. Giảm Claude-opus cho wildclaw.</li>
            <li><strong>ĐỀ XUẤT 4 — CONTEXT WINDOW (P1, 3 tuần):</strong> Áp dụng summarization khi tokens >40K để kìm hãm cost/step tăng phi mã.</li>
        </ul>
    </div>
    <div class="col-span-12" style="font-size:11px; color:#94a3b8; text-align:center; padding: 12px 0;">
        * Caveat: cost & tokens là per-step & tích lũy; spike ~$301 = timeout môi trường; flag≠resolved ở wildclaw.<br>
        Source: processed_agentic_traces.csv
    </div>
</div>
""", unsafe_allow_html=True)
