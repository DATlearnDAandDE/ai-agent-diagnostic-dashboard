import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

st.set_page_config(page_title="Data Story: AI Agent", layout="wide", initial_sidebar_state="collapsed")

# =============================================================================
# UI/UX STYLING (CSS)
# =============================================================================
THEME = st.session_state.get('theme', 'dark')

def toggle_theme():
    st.session_state.theme = 'light' if st.session_state.get('theme', 'dark') == 'dark' else 'dark'

if THEME == 'dark':
    BG_COLOR = "#0D1117"
    TEXT_COLOR = "#C9D1D9"
    H_COLOR = "#FFFFFF"
    GRID_COLOR = "rgba(255,255,255,0.05)"
    BOX_BG = "rgba(255,255,255,0.03)"
else:
    BG_COLOR = "#EEF1F7"
    TEXT_COLOR = "#334155"
    H_COLOR = "#1F4E79"
    GRID_COLOR = "rgba(0,0,0,0.05)"
    BOX_BG = "#FFFFFF"

PALETTE = {
    'minimax-m2.5': '#34D399',
    'deepseek-v3.1': '#38E1D6',
    'claude-sonnet-4-6': '#F5B544',
    'claude-opus-4-6': '#9B8CFF'
}

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@500;700&display=swap');
    
    .stApp {{
        background-color: {BG_COLOR};
        color: {TEXT_COLOR};
        font-family: 'IBM Plex Sans', sans-serif;
        background-image: 
            linear-gradient({GRID_COLOR} 1px, transparent 1px),
            linear-gradient(90deg, {GRID_COLOR} 1px, transparent 1px);
        background-size: 40px 40px;
    }}
    h1, h2, h3, h4 {{ font-family: 'Space Grotesk', sans-serif; color: {H_COLOR}; }}
    
    /* Layout with sticky rail */
    .layout-wrapper {{ display: flex; max-width: 1600px; margin: 0 auto; gap: 40px; }}
    .sticky-rail {{
        width: 200px;
        position: sticky;
        top: 2rem;
        height: max-content;
        border-right: 1px solid {GRID_COLOR};
        padding-right: 20px;
    }}
    .main-content {{ flex: 1; padding-bottom: 100px; max-width: 1100px; }}
    
    @media (max-width: 1100px) {{
        .layout-wrapper {{ flex-direction: column; }}
        .sticky-rail {{ position: relative; width: 100%; border-right: none; border-bottom: 1px solid {GRID_COLOR}; padding-bottom: 20px; display: flex; flex-wrap: wrap; gap: 10px; top: 0; }}
        .main-content {{ max-width: 100%; }}
    }}
    
    .nav-item {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: {TEXT_COLOR};
        opacity: 0.5;
        margin-bottom: 24px;
        cursor: pointer;
        transition: 0.3s;
    }}
    .nav-item:hover, .nav-item.active {{ opacity: 1; color: {H_COLOR}; }}
    
    .kpi-strip {{ display: flex; gap: 16px; margin-bottom: 32px; flex-wrap: wrap; }}
    .kpi-box {{
        background: {BOX_BG};
        border: 1px solid {GRID_COLOR};
        border-radius: 8px;
        padding: 16px 20px;
        flex: 1;
        min-width: 140px;
    }}
    .kpi-val {{ font-family: 'JetBrains Mono', monospace; font-size: 34px; font-weight: 700; color: {H_COLOR}; line-height: 1.2; }}
    .kpi-lbl {{ font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: {TEXT_COLOR}; opacity: 0.8; }}
    .wasted-box {{ border-color: #FF6B6B; background: rgba(255, 107, 107, 0.05); }}
    .wasted-box .kpi-val {{ color: #FF6B6B; }}
    
    .story-box {{
        border-left: 4px solid #5B8DEF;
        background: {BOX_BG};
        padding: 24px 32px;
        margin: 40px 0;
        border-radius: 0 8px 8px 0;
        font-size: 1.15rem;
        line-height: 1.6;
        color: {TEXT_COLOR};
    }}
    
    .act-title {{
        font-size: 2.5rem;
        margin: 60px 0 24px 0;
        border-bottom: 2px solid {GRID_COLOR};
        padding-bottom: 12px;
    }}
    
    .footer-caveat {{
        font-size: 0.85rem; color: {TEXT_COLOR}; opacity: 0.6; margin-top: 60px;
        border-top: 1px solid {GRID_COLOR}; padding-top: 20px;
    }}
</style>
""", unsafe_allow_html=True)

PLOT_CFG = dict(
    template='plotly_dark' if THEME == 'dark' else 'plotly_white',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='IBM Plex Sans', size=12, color=TEXT_COLOR),
    margin=dict(t=50, l=40, r=20, b=40)
)

def sf(fig, title="", height=380):
    fig.update_layout(**PLOT_CFG, height=height, title=dict(text=title, font=dict(family='Space Grotesk', size=16, color=H_COLOR)))
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    return fig

# =============================================================================
# PART A: DATA PIPELINE
# =============================================================================
@st.cache_data
def load_data(filepath='processed_agentic_traces.csv'):
    df = pd.read_csv(filepath)
    df = df.sort_values(['session_id', 'turn_number'])
    
    df['delta_tokens'] = df.groupby('session_id')['input_tokens'].diff().fillna(df['input_tokens'])
    df['cum_cost'] = df.groupby('session_id')['turn_cost'].cumsum()
    
    sess_agg = df.groupby('session_id').agg(
        model=('model', 'first'),
        total_cost=('turn_cost', 'sum'),
        turns=('turn_number', 'max'),
        error_rate=('has_error', 'mean'),
        final_tokens=('input_tokens', 'max'),
        is_sys_prompt=('is_system_prompt_present', 'first')
    ).reset_index()
    
    sess_agg['failed'] = (sess_agg['error_rate'] == 1.0).astype(int)
    sess_agg['wasted_cost'] = sess_agg['failed'] * sess_agg['total_cost']
    
    turn5 = df[df['turn_number'] <= 5].groupby('session_id').agg(
        early_err_5=('has_error', 'mean'),
        tokens_5=('input_tokens', 'max'),
        delta_tokens_5=('delta_tokens', 'mean')
    ).reset_index()
    
    turn10 = df[df['turn_number'] == 10].groupby('session_id').agg(
        cost_10=('cum_cost', 'max')
    ).reset_index()
    
    sess_agg = sess_agg.merge(turn5, on='session_id', how='left').merge(turn10, on='session_id', how='left')
    
    # Calculate streak for R1 simulation
    def get_streak(s):
        return s.groupby((s != s.shift()).cumsum()).cumsum()
    df['err_streak'] = df.groupby('session_id')['has_error'].apply(get_streak).reset_index(level=0, drop=True)
    
    return df, sess_agg

df, sess = load_data()

kpi_cost = sess['total_cost'].sum()
kpi_sess = len(sess)
kpi_turns = len(df)
kpi_err = df['has_error'].mean() * 100
kpi_fail = sess['failed'].mean() * 100
kpi_wasted = sess['wasted_cost'].sum()
kpi_sonnet_cost_pct = sess[sess['model'] == 'claude-sonnet-4-6']['total_cost'].sum() / kpi_cost * 100 if kpi_cost > 0 else 0

def wilson_score(p, n, z=1.96):
    if n == 0: return 0,0
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    spread = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return center - spread, center + spread

# =============================================================================
# CHARTS LOGIC
# =============================================================================

# C1: H-BAR total cost
def draw_c1():
    m_cost = sess.groupby('model')['total_cost'].sum().reset_index().sort_values('total_cost')
    fig = px.bar(m_cost, y='model', x='total_cost', orientation='h', color='model', color_discrete_map=PALETTE)
    if 'claude-sonnet-4-6' in m_cost['model'].values:
        fig.add_annotation(x=m_cost['total_cost'].max()*0.8, y='claude-sonnet-4-6', text=f"sonnet ≈{kpi_sonnet_cost_pct:.1f}%", showarrow=True, arrowhead=2, ax=-40, ay=-30, font=dict(color=TEXT_COLOR))
    fig.update_layout(showlegend=False)
    return sf(fig, "C1. Ai ngốn tiền? (Tổng chi phí)", 280)

# C2: BAR Error rate
def draw_c2():
    m_err = df.groupby('model').agg(err=('has_error', 'mean'), n=('has_error', 'count')).reset_index()
    f_rate = sess.groupby('model')['failed'].mean().reset_index()
    m_err = m_err.merge(f_rate, on='model').sort_values('err', ascending=False)
    
    y_vals, err_minus, err_plus = [], [], []
    for _, r in m_err.iterrows():
        p = r['err']
        low, high = wilson_score(p, r['n'])
        y_vals.append(p)
        err_minus.append(p - low)
        err_plus.append(high - p)
        
    fig = go.Figure(go.Bar(
        x=m_err['model'], y=y_vals, error_y=dict(type='data', array=err_plus, arrayminus=err_minus),
        marker_color=[PALETTE.get(m, '#888') for m in m_err['model']],
        text=[f"F:{f*100:.0f}%" for f in m_err['failed']], textposition='outside'
    ))
    if 'claude-sonnet-4-6' in m_err['model'].values:
        fig.add_annotation(x='claude-sonnet-4-6', y=m_err[m_err['model']=='claude-sonnet-4-6']['err'].values[0], text="Ngốn 66.8% ngân sách ↔ Lỗi cao nhất", showarrow=True, arrowhead=2, ax=40, ay=-40, font=dict(color=TEXT_COLOR))
    fig.update_layout(yaxis_tickformat='.0%', showlegend=False)
    return sf(fig, "C2. Tỉ lệ lỗi theo Turn (kèm % Failed Session)", 280)

# C3: LINE cum_cost
def draw_c3():
    fig = go.Figure()
    mean_cost = df.groupby(['model', 'turn_number'])['cum_cost'].mean().reset_index()
    for m in mean_cost['model'].unique():
        m_data = mean_cost[mean_cost['model'] == m]
        fig.add_trace(go.Scatter(x=m_data['turn_number'], y=m_data['cum_cost'], mode='lines', line=dict(color=PALETTE.get(m, '#888'), width=3), name=m))
    fig.add_annotation(x=40, y=mean_cost['cum_cost'].max()*0.8, text="Đường cong quá dốc? -> Xem C4", showarrow=False, font=dict(color=TEXT_COLOR))
    return sf(fig, "C3. Tiền cháy theo turn ra sao?", 320)

# C4: LINE input_tokens ~ turn
def draw_c4():
    sonnet = df[df['model'] == 'claude-sonnet-4-6']
    mean_tok = sonnet.groupby(['is_system_prompt_present', 'turn_number'])['input_tokens'].mean().reset_index()
    fig = go.Figure()
    for status, name, color in [(1, 'Sys Prompt ON', '#F5B544'), (0, 'Sys Prompt OFF', '#5B8DEF')]:
        m_data = mean_tok[mean_tok['is_system_prompt_present'] == status]
        fig.add_trace(go.Scatter(x=m_data['turn_number'], y=m_data['input_tokens'], mode='lines', line=dict(color=color, width=3), name=name))
    fig.add_annotation(x=30, y=20000, text="Vì sao đường cong dốc? Vì context phình", showarrow=False, font=dict(color=TEXT_COLOR))
    return sf(fig, "C4. Context Bloat (Sonnet)", 320)

# C5: SCATTER turn_cost ~ input_tokens
def draw_c5():
    samp = df.sample(min(5000, len(df)))
    fig = px.scatter(samp, x='input_tokens', y='turn_cost', color='model', color_discrete_map=PALETTE)
    
    from sklearn.linear_model import LinearRegression
    for m in samp['model'].unique():
        m_data = samp[samp['model'] == m].dropna(subset=['input_tokens', 'turn_cost'])
        if len(m_data) > 1:
            X = m_data[['input_tokens']].values
            y = m_data['turn_cost'].values
            reg = LinearRegression().fit(X, y)
            x_range = np.linspace(X.min(), X.max(), 100)
            y_pred = reg.predict(x_range.reshape(-1, 1))
            fig.add_trace(go.Scatter(x=x_range, y=y_pred, mode='lines', line=dict(color=PALETTE.get(m), width=2, dash='dash'), showlegend=False))
            
    fig.add_annotation(x=df['input_tokens'].max()*0.5, y=df['turn_cost'].max()*0.8, text="Phình × đơn giá đắt = Cạm bẫy", showarrow=False, font=dict(color=TEXT_COLOR))
    return sf(fig, "C5. Đơn giá biên: Cost ~ Tokens", 320)

# C6: DUMBBELL Sys Prompt ON vs OFF
def draw_c6():
    s_on = sess[(sess['model']=='claude-sonnet-4-6') & (sess['is_sys_prompt']==1)]
    s_off = sess[(sess['model']=='claude-sonnet-4-6') & (sess['is_sys_prompt']==0)]
    
    if len(s_off) == 0: return sf(go.Figure(), "C6. Chưa đủ data OFF")
    
    m_on = {'tok': s_on['final_tokens'].mean(), 'err': df[(df['model']=='claude-sonnet-4-6')&(df['is_system_prompt_present']==1)]['has_error'].mean()*100, 'fail': s_on['failed'].mean()*100, 'turns': s_on['turns'].mean(), 'cost': s_on['total_cost'].mean()}
    m_off = {'tok': s_off['final_tokens'].mean(), 'err': df[(df['model']=='claude-sonnet-4-6')&(df['is_system_prompt_present']==0)]['has_error'].mean()*100, 'fail': s_off['failed'].mean()*100, 'turns': s_off['turns'].mean(), 'cost': s_off['total_cost'].mean()}
    
    metrics = list(m_on.keys())
    fig = go.Figure()
    for i, m in enumerate(metrics):
        val_off, val_on = m_off[m], m_on[m]
        pct = (val_on - val_off)/val_off*100 if val_off else 0
        norm_off = 0
        norm_on = pct
        fig.add_trace(go.Scatter(x=[norm_off, norm_on], y=[m, m], mode='lines+markers', line=dict(color='#5B8DEF', width=3), marker=dict(color=['#34D399', '#FF6B6B'], size=10), showlegend=False))
        fig.add_annotation(x=norm_on, y=m, text=f"+{pct:.0f}%" if pct>0 else f"{pct:.0f}%", showarrow=False, yshift=15, font=dict(color=TEXT_COLOR))
        
    fig.add_annotation(x=50, y='tok', text="Ai bật công tắc phình? System Prompt", showarrow=False, font=dict(color=TEXT_COLOR))
    return sf(fig, "C6. Tác động của System Prompt (Chuẩn hoá %)", 320)

# C7: LINE err rate theo turn ON vs OFF
def draw_c7():
    sonnet = df[df['model'] == 'claude-sonnet-4-6']
    err_t = sonnet.groupby(['is_system_prompt_present', 'turn_number'])['has_error'].mean().reset_index()
    fig = go.Figure()
    for status, name, color in [(1, 'Sys Prompt ON', '#FF6B6B'), (0, 'Sys Prompt OFF', '#34D399')]:
        d = err_t[err_t['is_system_prompt_present'] == status]
        fig.add_trace(go.Scatter(x=d['turn_number'], y=d['has_error'], mode='lines', line=dict(color=color), name=name))
    fig.update_yaxes(tickformat='.0%')
    return sf(fig, "C7. Tốc độ tích luỹ lỗi theo turn", 280)

# C8: ROC Turn 5
def draw_c8():
    valid = sess.dropna(subset=['tokens_5', 'early_err_5'])
    if len(valid) < 10: return sf(go.Figure(), "C8. Thiếu data ROC")
    
    X = valid[['tokens_5', 'early_err_5']]
    y = valid['failed']
    try:
        clf = LogisticRegression(class_weight='balanced').fit(X, y)
        fpr, tpr, _ = roc_curve(y, clf.predict_proba(X)[:, 1])
        roc_auc = auc(fpr, tpr)
        
        fig = px.area(x=fpr, y=tpr, title=f"AUC = {roc_auc:.2f}", labels={'x':'FPR', 'y':'TPR'}, color_discrete_sequence=['#5B8DEF'])
        fig.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, x1=1, y0=0, y1=1)
        fig.add_annotation(x=0.5, y=0.1, text="Đã độc hại thì phát hiện sớm được không?", showarrow=False, font=dict(color=TEXT_COLOR))
    except Exception:
        fig = go.Figure()
    return sf(fig, "C8. ROC - Dự báo thất bại từ Turn 5", 320)

# C9: SCATTER pred cost
def draw_c9():
    valid = sess.dropna(subset=['cost_10'])
    fig = px.scatter(valid, x='cost_10', y='total_cost', log_x=True, log_y=True, color='model', color_discrete_map=PALETTE)
    fig.add_shape(type='line', x0=valid['cost_10'].min(), x1=valid['total_cost'].max(), y0=valid['cost_10'].min(), y1=valid['total_cost'].max(), line=dict(dash='dash', color='gray'))
    return sf(fig, "C9. Cost@10 vs Final Cost", 320)

# C10: SURVIVAL
def draw_c10():
    fig = go.Figure()
    for m in sess['model'].unique():
        s_m = sess[sess['model'] == m]
        max_t = s_m['turns'].max()
        steps = np.arange(1, max_t+1)
        surv = [(s_m['turns'] >= s).sum() / len(s_m) for s in steps]
        fig.add_trace(go.Scatter(x=steps, y=surv, mode='lines', line_shape='hv', name=m, line=dict(color=PALETTE.get(m))))
    
    fig.add_vline(x=30, line_dash='dash', line_color='#FF6B6B')
    fig.add_annotation(x=30, y=0.8, text="Elbow = Hard-cap (R2)", showarrow=True, arrowhead=2, ax=40, font=dict(color=TEXT_COLOR))
    fig.update_yaxes(tickformat='.0%')
    return sf(fig, "C10. % Session còn chạy ~ Turn", 320)

# C11: BAR Simulation
def draw_c11():
    # R1: Dừng sau 3 lỗi
    r1_cut = df[df['err_streak'] >= 3].groupby('session_id')['turn_number'].min().reset_index()
    sess_r1 = sess.merge(r1_cut, on='session_id', how='left')
    sess_r1['r1_cost'] = sess_r1.apply(lambda r: df[(df['session_id']==r['session_id']) & (df['turn_number']<= (r['turn_number'] if pd.notnull(r['turn_number']) else r['turns']))]['turn_cost'].sum(), axis=1)
    
    # R2: Hard-cap tại 30
    sess_r1['r2_cost'] = sess_r1.apply(lambda r: df[(df['session_id']==r['session_id']) & (df['turn_number']<= min(r['turns'], 30))]['turn_cost'].sum(), axis=1)
    
    costs = [sess['total_cost'].sum(), sess_r1['r1_cost'].sum(), sess_r1['r2_cost'].sum()]
    fig = go.Figure(go.Bar(x=['Baseline', 'R1 (3 Fails)', 'R2 (Cap 30)'], y=costs, marker_color=['#5B8DEF', '#34D399', '#38E1D6']))
    fig.add_annotation(x='R2 (Cap 30)', y=costs[2], text="Tiết kiệm chảy vào C13", showarrow=True, arrowhead=2, ay=-40, font=dict(color=TEXT_COLOR))
    return sf(fig, "C11. Mô phỏng chi phí cắt giảm", 320)

# C12: BUBBLE Routing
def draw_c12():
    agg = sess.groupby('model').agg(c=('total_cost', 'mean'), f=('failed', 'mean'), n=('session_id', 'count')).reset_index()
    fig = px.scatter(agg, x='c', y='f', size='n', color='model', text='model', color_discrete_map=PALETTE)
    fig.add_vline(x=agg['c'].median(), line_dash='dash', line_color=GRID_COLOR)
    fig.add_hline(y=agg['f'].median(), line_dash='dash', line_color=GRID_COLOR)
    return sf(fig, "C12. Model Routing Quadrant", 320)

# C13: WATERFALL
def draw_c13():
    r1_cut = df[df['err_streak'] >= 3].groupby('session_id')['turn_number'].min().reset_index()
    sess_r1 = sess.merge(r1_cut, on='session_id', how='left')
    c_r1 = sess_r1.apply(lambda r: df[(df['session_id']==r['session_id']) & (df['turn_number']<= (r['turn_number'] if pd.notnull(r['turn_number']) else r['turns']))]['turn_cost'].sum(), axis=1).sum()
    c_r2 = sess_r1.apply(lambda r: df[(df['session_id']==r['session_id']) & (df['turn_number']<= min(r['turns'], 30))]['turn_cost'].sum(), axis=1).sum()
    
    tot = sess['total_cost'].sum()
    save_r1 = tot - c_r1
    save_r2 = c_r1 - c_r2 # sequential saving assumption
    
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["absolute", "relative", "relative", "total"],
        x=["Tổng ban đầu", "R1 (-3 Lỗi)", "R2 (-Cap30)", "Net Cost"],
        y=[tot, -save_r1, -save_r2, tot - save_r1 - save_r2],
        decreasing={"marker":{"color":"#34D399"}}
    ))
    return sf(fig, "C13. Sổ cái ROI", 320)

# =============================================================================
# SCROLLYTELLING LAYOUT
# =============================================================================
st.sidebar.button("Toggle Light/Dark Theme", on_click=toggle_theme)

st.markdown('<div class="layout-wrapper">', unsafe_allow_html=True)

# 1. STICKY RAIL
st.markdown(f"""
<div class="sticky-rail">
    <div style="font-family:'Space Grotesk'; font-size:1.8rem; font-weight:700; margin-bottom:40px; color:{H_COLOR};">DATA STORY</div>
    <div class="nav-item active">01 MÔ TẢ</div>
    <div class="nav-item">02 CHẨN ĐOÁN</div>
    <div class="nav-item">03 DỰ ĐOÁN</div>
    <div class="nav-item">04 KÊ TOA</div>
</div>
""", unsafe_allow_html=True)

# 2. MAIN CONTENT
with st.container():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    st.markdown(f'<h1 style="font-size:3rem; margin-bottom:10px; color:{H_COLOR}; line-height:1.1;">Cạm Bẫy Chi Phí & Nghịch Lý System Prompt</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:1.2rem; opacity:0.8; margin-bottom:40px;">Phân tích chuyên sâu hệ thống AI Agent dựa trên đo lường Telemetry.</p>', unsafe_allow_html=True)
    
    # KPI Strip
    st.markdown(f"""
    <div class="kpi-strip">
        <div class="kpi-box"><div class="kpi-val">${kpi_cost:.2f}</div><div class="kpi-lbl">Total Cost</div></div>
        <div class="kpi-box"><div class="kpi-val">{kpi_sess:,}</div><div class="kpi-lbl">Sessions</div></div>
        <div class="kpi-box"><div class="kpi-val">{kpi_turns:,}</div><div class="kpi-lbl">Turns</div></div>
        <div class="kpi-box"><div class="kpi-val">{kpi_err:.1f}%</div><div class="kpi-lbl">Turn Err Rate</div></div>
        <div class="kpi-box"><div class="kpi-val">{kpi_fail:.1f}%</div><div class="kpi-lbl">Failed %</div></div>
        <div class="kpi-box wasted-box"><div class="kpi-val">${kpi_wasted:.2f}</div><div class="kpi-lbl">Wasted Cost</div></div>
    </div>
    """, unsafe_allow_html=True)
    
    # ACT 1
    st.markdown('<div class="act-title">HỒI 1: Tiền đi đâu, nhận lại gì?</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(draw_c1(), use_container_width=True, key="c1")
    with c2: st.plotly_chart(draw_c2(), use_container_width=True, key="c2")
    st.plotly_chart(draw_c3(), use_container_width=True, key="c3")
    
    val_err_sonnet = df[df['model']=='claude-sonnet-4-6']['has_error'].mean()*100
    st.markdown(f'<div class="story-box"><strong>66.8%</strong> ngân sách chảy vào Claude-sonnet; đổi lại là turn error-rate <strong>{val_err_sonnet:.1f}%</strong> — cao nhất bảng, với đơn giá tốn kém nhất.</div>', unsafe_allow_html=True)

    # ACT 2
    st.markdown('<div class="act-title">HỒI 2: Giải phẫu cạm bẫy</div>', unsafe_allow_html=True)
    c4, c5 = st.columns(2)
    with c4: st.plotly_chart(draw_c4(), use_container_width=True, key="c4")
    with c5: st.plotly_chart(draw_c5(), use_container_width=True, key="c5")
    
    c6, c7 = st.columns(2)
    with c6: st.plotly_chart(draw_c6(), use_container_width=True, key="c6")
    with c7: st.plotly_chart(draw_c7(), use_container_width=True, key="c7")
    
    st.markdown(f'<div class="story-box">Cạm bẫy = Context phình × Đơn giá đắt. Nghịch lý lớn nhất: <strong>System Prompt</strong> khiến lượng token phình to +149% nhưng tỷ lệ thất bại từ 0% tăng vọt lên >40%.</div>', unsafe_allow_html=True)

    # ACT 3
    st.markdown('<div class="act-title">HỒI 3: Biết trước được gì, dừng ở đâu?</div>', unsafe_allow_html=True)
    c8, c9 = st.columns(2)
    with c8: st.plotly_chart(draw_c8(), use_container_width=True, key="c8")
    with c9: st.plotly_chart(draw_c9(), use_container_width=True, key="c9")
    st.plotly_chart(draw_c10(), use_container_width=True, key="c10")
    
    st.markdown(f'<div class="story-box">Từ turn 5 có thể dự báo sớm thất bại (AUC ~ 0.8). Từ turn 10 có thể nội suy hoá đơn cuối cùng. Đường Survival phẳng sau turn 30 — cố thêm chỉ đốt tiền vô ích.</div>', unsafe_allow_html=True)

    # ACT 4
    st.markdown('<div class="act-title">HỒI 4: Kê toa cắt giảm</div>', unsafe_allow_html=True)
    c11, c12 = st.columns(2)
    with c11: st.plotly_chart(draw_c11(), use_container_width=True, key="c11")
    with c12: st.plotly_chart(draw_c12(), use_container_width=True, key="c12")
    st.plotly_chart(draw_c13(), use_container_width=True, key="c13")
    
    st.markdown(f"""
    <div class="story-box">
        <strong>TÓM TẮT ĐIỀU HÀNH:</strong><br>
        1. <strong>[P0] Circuit Breaker:</strong> Dừng ngay task sau 3 turn lỗi liên tiếp (R1) hoặc chạm mốc turn 30 (R2) để vá lỗ hổng Wasted Cost.<br>
        2. <strong>[P1] System Prompt:</strong> Gỡ bỏ system prompt cồng kềnh trên Claude-sonnet, chuyển về zero-shot.<br>
        3. <strong>[P1] Model Routing:</strong> Định tuyến lại các task đơn giản sang Minimax để tận dụng đơn giá siêu rẻ.
    </div>
    """, unsafe_allow_html=True)

    # Caveat Footer
    st.markdown(f'<div class="footer-caveat">* Caveat: Giá trị <code>pre_gap</code> 10-52s là spike latency của môi trường mạng, KHÔNG TRỘN vào tính toán tiền tệ.<br>Source: <code>processed_agentic_traces.csv</code> · Generated Data Story.</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
