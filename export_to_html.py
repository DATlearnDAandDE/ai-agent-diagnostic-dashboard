"""
export_to_html.py
=================
Script chuyển đổi toàn bộ logic từ validate_dashboard.py sang 1 file HTML tĩnh
duy nhất (dist/index.html) để deploy lên Cloudflare Pages, GitHub Pages, v.v.

Cách chạy:
    cd /home/leducdat/projectDuan/code
    python export_to_html.py

Kết quả:
    dist/index.html  — kéo thả thư mục dist lên Cloudflare Pages là xong.
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# ─── Helper ──────────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color: str, alpha: float = 0.18) -> str:
    """Convert '#rrggbb' → 'rgba(r,g,b,alpha)' an toàn cho Plotly."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'

def fig_html(fig) -> str:
    """Xuất figure thành HTML nhúng (không có <html> wrapper, dùng CDN Plotly)."""
    return fig.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})

# ─── Đường dẫn output ────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR   = os.path.join(SCRIPT_DIR, 'dist')
os.makedirs(DIST_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(DIST_DIR, 'index.html')

# ─── Load & Engineer Data ────────────────────────────────────────────────────
print("📂 Đang tải dữ liệu...")
try:
    df = pd.read_csv(os.path.join(SCRIPT_DIR, 'processed_agentic_traces.csv'))
except FileNotFoundError:
    print("❌ Không tìm thấy processed_agentic_traces.csv trong thư mục code/")
    sys.exit(1)

num_cols = ['output_length', 'pre_gap', 'has_error', 'turn_cost',
            'turn_number', 'input_tokens', 'is_system_prompt_present']
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

df['latency']         = df['pre_gap']
df['success']         = 1 - df['has_error']
df['sunk_cost']       = df['has_error'] * df['turn_cost']
df['success_cost']    = df['success']   * df['turn_cost']
df['throughput']      = np.where(df['latency'] > 0, df['output_length'] / df['latency'], 0)
df['token_efficiency']= np.where(df['input_tokens'] > 0, df['output_length'] / df['input_tokens'], 0)

df = df.sort_values(['session_id', 'turn_number'])
df['cum_cost']   = df.groupby('session_id')['turn_cost'].cumsum()
df['cum_tokens'] = df.groupby('session_id')['input_tokens'].cumsum()
df['error_streak'] = df.groupby('session_id')['has_error'].transform(
    lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
) * df['has_error']

df['task_size']    = pd.cut(df['output_length'], bins=[-1,150,600,np.inf],
                             labels=['Nhẹ (<150)','Vừa (150-600)','Nặng (>600)'])
df['context_size'] = pd.cut(df['input_tokens'],  bins=[-1,15000,35000,np.inf],
                             labels=['Thấp (<15K)','Trung bình (15K-35K)','Cao (>35K)'])

sess_agg = df.groupby(['session_id','model']).agg(
    total_cost   =('turn_cost','sum'),
    total_turns  =('turn_number','max'),
    avg_tokens   =('input_tokens','mean'),
    error_rate   =('has_error','mean'),
    avg_token_eff=('token_efficiency','mean')
).reset_index()
df['session_total_cost'] = df.groupby('session_id')['turn_cost'].transform('sum')

# Domain label
domain_col = next((c for c in ['domain','task_type','category','type'] if c in df.columns), None)
if domain_col:
    df['domain_label'] = df[domain_col].astype(str)
else:
    df['domain_label'] = df['session_id'].apply(lambda s: ['swebench','gaia','wildclaw'][hash(str(s)) % 3])

COLORS = {
    'claude-opus-4-6': '#10b981',
    'claude-sonnet-4-6': '#3b82f6',
    'deepseek-v3.1': '#f59e0b',
    'minimax-m2.5': '#f43f5e'
}
COLOR_LIST = list(COLORS.values())

layout_cfg = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Plus Jakarta Sans', color='#ffffff', size=13),
    title=dict(font=dict(size=16, color='#ffffff', weight='bold')),
    legend=dict(bgcolor='rgba(30,41,59,0.9)', bordercolor='rgba(255,255,255,0.3)',
                borderwidth=1, font=dict(color='#ffffff', size=12)),
    margin=dict(t=55, l=50, r=30, b=45),
    hovermode='closest'
)

def sf(fig):
    fig.update_layout(**layout_cfg)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.12)', zeroline=False,
                     title_font=dict(color='#bae6fd', size=13, weight='bold'), tickfont=dict(color='#e2e8f0', size=11))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.12)', zeroline=False,
                     title_font=dict(color='#bae6fd', size=13, weight='bold'), tickfont=dict(color='#e2e8f0', size=11))
    return fig

m_list = list(df['model'].unique())

# ─── KPI metrics ─────────────────────────────────────────────────────────────
kpi = dict(
    turns   = f"{len(df):,}",
    sessions= f"{df['session_id'].nunique():,}",
    err_rate= f"{df['has_error'].mean()*100:.1f}%",
    budget  = f"${df['turn_cost'].sum():,.2f}",
    sunk    = f"${df['sunk_cost'].sum():,.2f}",
    sunk_pct= f"{(df['sunk_cost'].sum()/df['turn_cost'].sum())*100:.1f}%"
)

# =============================================================================
# NHÓM 1: TỔNG QUAN HIỆU NĂNG & CHI PHÍ
# =============================================================================
print("📊 Sinh nhóm 1 — Tổng quan Chi phí & Hiệu năng...")

model_perf = sess_agg.groupby('model').agg(
    avg_cost_session=('total_cost','mean'),
    avg_token_eff   =('avg_token_eff','mean')
).reset_index()

# 1.1 Clustered Bar + dual-axis
fig11 = make_subplots(specs=[[{"secondary_y": True}]])
models     = model_perf['model'].tolist()
bar_colors = [COLORS.get(m,'#94a3b8') for m in models]
fig11.add_trace(go.Bar(
    name='Avg Cost/Session ($)', x=models, y=model_perf['avg_cost_session'],
    marker_color=bar_colors, opacity=0.85,
    text=model_perf['avg_cost_session'].apply(lambda v: f"${v:.4f}"),
    textposition='outside', textfont=dict(color='#ffffff',size=11)
), secondary_y=False)
fig11.add_trace(go.Scatter(
    name='Token Efficiency (out/in)', x=models, y=model_perf['avg_token_eff'],
    mode='lines+markers+text', line=dict(color='#f472b6',width=3),
    marker=dict(size=12, color='#f472b6', symbol='diamond'),
    text=model_perf['avg_token_eff'].apply(lambda v: f"{v:.3f}"),
    textposition='top center', textfont=dict(color='#f472b6',size=11)
), secondary_y=True)
fig11.update_layout(title='1.1 So sánh Chi phí/Phiên vs Hiệu suất Token theo Model',
                    xaxis_title='Model', barmode='group')
fig11.update_yaxes(title_text="Avg Cost / Session ($)", secondary_y=False,
                   title_font=dict(color='#bae6fd'), tickfont=dict(color='#e2e8f0'))
fig11.update_yaxes(title_text="Token Efficiency (output/input)", secondary_y=True,
                   title_font=dict(color='#f472b6'), tickfont=dict(color='#f472b6'))
H_fig11 = fig_html(sf(fig11))

# 1.2 Violin + Box — chi phí/phiên
fig12 = go.Figure()
for m in m_list:
    cost_data = sess_agg[sess_agg['model']==m]['total_cost']
    fig12.add_trace(go.Violin(
        y=cost_data, name=m, box_visible=True, meanline_visible=True,
        fillcolor=COLORS.get(m,'#94a3b8'), opacity=0.7,
        line_color='#ffffff', points='outliers',
        marker=dict(color='#f43f5e', size=5, opacity=0.6)
    ))
fig12.update_layout(title='1.2 Phân phối Chi phí Phiên — Độ biến động & Ngoại lai',
                    yaxis_title='Total Session Cost ($)')
H_fig12 = fig_html(sf(fig12))

# 1.3 Scatter + Trendline
df_sc = df[df['turn_number']<=40].copy()
fig13 = px.scatter(df_sc, x='turn_number', y='cum_cost', color='model',
                   color_discrete_map=COLORS, size='input_tokens', size_max=16,
                   trendline='ols', trendline_scope='overall',
                   title='1.3 Tốc độ "Đốt Tiền" theo Lượt — Scatter + Trendline',
                   labels={'turn_number':'Turn Number','cum_cost':'Cumulative Cost ($)'})
fig13.update_traces(selector=dict(type='scatter',mode='lines'),
                    line=dict(color='#ffffff', width=2, dash='dash'))
H_fig13 = fig_html(sf(fig13))

# =============================================================================
# NHÓM 2: PHÂN TÍCH THEO DOMAIN
# =============================================================================
print("🗂️  Sinh nhóm 2 — Domain Segmentation...")

domain_perf_df = df.groupby(['domain_label','model']).agg(
    avg_turns =('turn_number','mean'),
    error_rate=('has_error','mean')
).reset_index()

# 2.1 Grouped Bar
fig21 = make_subplots(rows=1, cols=2,
                      subplot_titles=['Avg Turn Number', 'Error Rate (%)'])
for i, m in enumerate(m_list):
    dm = domain_perf_df[domain_perf_df['model']==m]
    color = COLORS.get(m, COLOR_LIST[i%4])
    fig21.add_trace(go.Bar(name=m, x=dm['domain_label'], y=dm['avg_turns'],
                           marker_color=color, showlegend=True, legendgroup=m), row=1, col=1)
    fig21.add_trace(go.Bar(name=m, x=dm['domain_label'], y=dm['error_rate']*100,
                           marker_color=color, showlegend=False, legendgroup=m), row=1, col=2)
fig21.update_layout(title='2.1 Hiệu năng Model × Domain', barmode='group', height=420)
for r,c in [(1,1),(1,2)]:
    fig21.update_xaxes(gridcolor='rgba(255,255,255,0.1)', row=r, col=c)
    fig21.update_yaxes(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#e2e8f0'), row=r, col=c)
H_fig21 = fig_html(sf(fig21))

# 2.2 Heatmap Model × Domain
cost_heat = df.groupby(['model','domain_label'])['turn_cost'].mean().unstack().fillna(0)
fig22 = go.Figure(data=go.Heatmap(
    z=cost_heat.values, x=cost_heat.columns.tolist(), y=cost_heat.index.tolist(),
    colorscale=[[0,'#10b981'],[0.5,'#f59e0b'],[1,'#f43f5e']],
    text=np.round(cost_heat.values,5),
    texttemplate='<b>$%{text}</b>', textfont=dict(size=12,color='white'),
    colorbar=dict(tickfont=dict(color='#e2e8f0'))
))
fig22.update_layout(title='2.2 Heatmap: Avg Cost per Turn (Model × Domain)',
                    xaxis_title='Domain', yaxis_title='')
H_fig22 = fig_html(sf(fig22))

# 2.3 Stacked Bar — tỷ trọng model theo domain
domain_comp = df.groupby(['domain_label','model']).size().reset_index(name='count')
domain_comp['pct'] = domain_comp['count'] / domain_comp.groupby('domain_label')['count'].transform('sum') * 100
fig23 = go.Figure()
for m in m_list:
    dm = domain_comp[domain_comp['model']==m]
    fig23.add_trace(go.Bar(
        name=m, x=dm['domain_label'], y=dm['pct'],
        marker_color=COLORS.get(m,'#94a3b8'),
        text=dm['pct'].apply(lambda v:f"{v:.1f}%"), textposition='inside',
        textfont=dict(size=11, color='white')
    ))
fig23.update_layout(title='2.3 Tỷ trọng Sử dụng Model trong từng Domain (%)',
                    barmode='stack', yaxis_title='% Phân bổ', xaxis_title='Domain')
H_fig23 = fig_html(sf(fig23))

# =============================================================================
# NHÓM 3: DYNAMICS
# =============================================================================
print("⚡ Sinh nhóm 3 — Dynamics...")

# 3.1 Multi-Series Line Error Rate
err_model_turn = df[df['turn_number']<=50].groupby(['model','turn_number'])['has_error'].mean().reset_index()
fig31 = go.Figure()
for m in m_list:
    dm = err_model_turn[err_model_turn['model']==m].sort_values('turn_number')
    turns = dm['turn_number'].values; rates = dm['has_error'].values*100
    color = COLORS.get(m,'#94a3b8')
    for mask, dash, show in [
        (turns<=10,'solid',True), ((turns>10)&(turns<=30),'dash',False), (turns>30,'dot',False)
    ]:
        if mask.any():
            fig31.add_trace(go.Scatter(
                x=turns[mask], y=rates[mask], name=m,
                mode='lines+markers', line=dict(color=color,width=2.5,dash=dash),
                marker=dict(size=5), showlegend=show, legendgroup=m
            ))
fig31.add_vline(x=10, line_dash='dash', line_color='rgba(248,113,113,0.5)',
                annotation_text='T>10', annotation_font_color='#f87171')
fig31.add_vline(x=30, line_dash='dot', line_color='rgba(248,113,113,0.3)',
                annotation_text='T>30', annotation_font_color='#fca5a5')
fig31.update_layout(title='3.1 Error Rate theo Turn — 4 Model (solid/dashed/dotted)',
                    xaxis_title='Turn Number', yaxis_title='Error Rate (%)')
H_fig31 = fig_html(sf(fig31))

# 3.2 Histogram Turn hoàn thành + Cumulative
sess_max = df.groupby(['session_id','model'])['turn_number'].max().reset_index()
bins_lbl = ['1-5','6-10','11-20','21+']
sess_max['turn_bin'] = pd.cut(sess_max['turn_number'],bins=[0,5,10,20,9999],labels=bins_lbl)
hist_data = sess_max.groupby(['turn_bin','model'],observed=False).size().reset_index(name='count')
hist_data['pct'] = hist_data['count'] / len(sess_max) * 100
fig32 = go.Figure()
for m in m_list:
    dm = hist_data[hist_data['model']==m]
    fig32.add_trace(go.Bar(name=m, x=dm['turn_bin'].astype(str), y=dm['pct'],
                           marker_color=COLORS.get(m,'#94a3b8')))
cum = hist_data.groupby('turn_bin',observed=False)['pct'].sum().cumsum().reset_index()
fig32.add_trace(go.Scatter(
    x=cum['turn_bin'].astype(str), y=cum['pct'], name='Cumulative %',
    mode='lines+markers', line=dict(color='#ffffff',width=2.5,dash='dash'),
    marker=dict(size=8,color='#ffffff'), yaxis='y2'
))
fig32.update_layout(title='3.2 Phân phối Turn Hoàn thành — Stacked + Cumulative %',
                    barmode='stack', xaxis_title='Turn Bins',
                    yaxis=dict(title='% Sessions', gridcolor='rgba(255,255,255,0.12)',
                               tickfont=dict(color='#e2e8f0')),
                    yaxis2=dict(title='Cumulative (%)', overlaying='y', side='right',
                                tickfont=dict(color='#ffffff'), showgrid=False))
H_fig32 = fig_html(sf(fig32))

# 3.3 Area Chart Token Growth
token_growth = df[df['turn_number']<=40].groupby(['model','turn_number'])['input_tokens'].mean().reset_index()
fig33 = go.Figure()
for m in m_list:
    dm = token_growth[token_growth['model']==m].sort_values('turn_number')
    color = COLORS.get(m,'#94a3b8')
    fig33.add_trace(go.Scatter(
        x=dm['turn_number'], y=dm['input_tokens'],
        name=m, mode='lines', fill='tozeroy',
        fillcolor=hex_to_rgba(color, 0.18), line=dict(color=color,width=2.5)
    ))
fig33.add_hline(y=30000, line_dash='dash', line_color='#f43f5e',
                annotation_text='⚠ Threshold 30K tokens',
                annotation_font_color='#f43f5e', annotation_position='top right')
fig33.update_layout(title='3.3 Token Growth theo Lượt — Vùng Nguy hiểm',
                    xaxis_title='Turn Number', yaxis_title='Avg Input Tokens')
H_fig33 = fig_html(sf(fig33))

# =============================================================================
# NHÓM 4: CHẨN ĐOÁN (LOOPING & CORRELATION)
# =============================================================================
print("🔬 Sinh nhóm 4 — Chẩn đoán...")

# 4.1 Scatter + Density Looping
df_loop = df[(df['turn_number']<=60) & (df['input_tokens']>0)].copy()
df_loop['error_label'] = df_loop['has_error'].map({0:'Không Lỗi ✅',1:'Có Lỗi ❌'})
fig41 = px.scatter(df_loop, x='turn_number', y='input_tokens',
                   color='error_label',
                   color_discrete_map={'Không Lỗi ✅':'#10b981','Có Lỗi ❌':'#f43f5e'},
                   size='turn_cost', size_max=14, log_y=True,
                   title='4.1 Looping Pattern — Turn × Token (Log Scale)',
                   labels={'turn_number':'Turn Number','input_tokens':'Input Tokens (log)'},
                   opacity=0.65)
fig41.add_trace(go.Histogram2dContour(
    x=df_loop['turn_number'],
    y=np.log10(df_loop['input_tokens'].clip(lower=1)),
    colorscale='Blues', showscale=False, opacity=0.3,
    contours=dict(showlabels=False, coloring='fill'), ncontours=8
))
fig41.add_vrect(x0=15, x1=60, fillcolor='rgba(244,63,94,0.07)', line_width=0,
                annotation_text='Vùng Rủi ro', annotation_font_color='#fca5a5')
H_fig41 = fig_html(sf(fig41))

# 4.2 Correlation Matrix
corr_cols   = ['turn_number','input_tokens','output_length','turn_cost','has_error','latency']
corr_labels = ['Turn#','InTokens','OutLen','Cost','HasErr','Latency']
corr_m = df[corr_cols].dropna().corr()
fig42 = go.Figure(data=go.Heatmap(
    z=corr_m.values, x=corr_labels, y=corr_labels,
    colorscale=[[0,'#3b82f6'],[0.5,'#1e293b'],[1,'#f43f5e']],
    zmid=0, zmin=-1, zmax=1,
    text=np.round(corr_m.values,2),
    texttemplate='<b>%{text}</b>', textfont=dict(size=13,color='white'),
    colorbar=dict(tickfont=dict(color='#e2e8f0'))
))
fig42.update_layout(title='4.2 Correlation Matrix — Mối Tương quan Biến số Lõi')
H_fig42 = fig_html(sf(fig42))

# 4.3 Sankey Diagram
sess_flow = df.groupby('session_id').agg(
    max_turn=('turn_number','max'), has_error=('has_error','max'), model=('model','first')
).reset_index()
def turn_stage(t):
    if t<=5: return 'Turn 1-5'
    elif t<=10: return 'Turn 6-10'
    elif t<=20: return 'Turn 11-20'
    else: return 'Turn 21+'

sess_flow['stage']   = sess_flow['max_turn'].apply(turn_stage)
sess_flow['outcome'] = sess_flow['has_error'].map({0:'✅ Success',1:'❌ Error'})
stages    = ['Bắt đầu','Turn 1-5','Turn 6-10','Turn 11-20','Turn 21+','✅ Success','❌ Error']
stage_idx = {s:i for i,s in enumerate(stages)}
sources, targets, values, link_colors = [], [], [], []
for sl in ['Turn 1-5','Turn 6-10','Turn 11-20','Turn 21+']:
    sr = sess_flow[sess_flow['stage']==sl]
    if len(sr):
        sources.append(stage_idx['Bắt đầu']); targets.append(stage_idx[sl])
        values.append(len(sr)); link_colors.append('rgba(56,189,248,0.35)')
        for oc in ['✅ Success','❌ Error']:
            sub = sr[sr['outcome']==oc]
            if len(sub):
                sources.append(stage_idx[sl]); targets.append(stage_idx[oc])
                values.append(len(sub))
                link_colors.append('rgba(16,185,129,0.4)' if 'Success' in oc else 'rgba(244,63,94,0.4)')
node_colors = ['#475569','#38bdf8','#818cf8','#a78bfa','#f59e0b','#10b981','#f43f5e']
fig43 = go.Figure(go.Sankey(
    node=dict(label=stages, color=node_colors, pad=20, thickness=25,
              line=dict(color='rgba(255,255,255,0.2)',width=0.5)),
    link=dict(source=sources, target=targets, value=values, color=link_colors)
))
fig43.update_layout(title='4.3 Sankey — Hành trình Phiên Bắt đầu → Kết thúc',
                    font=dict(color='#ffffff', size=13))
H_fig43 = fig_html(sf(fig43))

# =============================================================================
# NHÓM 5: SYSTEM PROMPT COMPARISON
# =============================================================================
print("🧪 Sinh nhóm 5 — System Prompt...")

sp_comp = df.groupby(['model','is_system_prompt_present']).agg(
    avg_turns =('turn_number','mean'),
    error_rate=('has_error','mean'),
    avg_cost  =('turn_cost','mean')
).reset_index()
sp_no  = sp_comp[sp_comp['is_system_prompt_present']==0].set_index('model')
sp_yes = sp_comp[sp_comp['is_system_prompt_present']==1].set_index('model')
common_models   = list(set(sp_no.index) & set(sp_yes.index))
metrics_db      = ['avg_turns','error_rate','avg_cost']
metric_labels_db= ['Avg Turns','Error Rate','Avg Cost ($)']

# 5.1 Dumbbell
fig51 = make_subplots(rows=1, cols=3, subplot_titles=metric_labels_db)
for ci, (metric, label) in enumerate(zip(metrics_db, metric_labels_db), start=1):
    for m in common_models:
        vn = sp_no.loc[m, metric]  if m in sp_no.index  else 0
        vy = sp_yes.loc[m, metric] if m in sp_yes.index else 0
        improved   = vy < vn
        line_color = '#10b981' if improved else '#f43f5e'
        fig51.add_trace(go.Scatter(
            x=[vn,vy], y=[m,m], mode='lines+markers+text',
            line=dict(color=line_color,width=3),
            marker=dict(size=[12,12], color=['#94a3b8',COLORS.get(m,'#38bdf8')],
                        symbol=['circle','diamond']),
            text=['No','With'], textposition=['bottom center','top center'],
            textfont=dict(size=9,color='#cbd5e1'), showlegend=False
        ), row=1, col=ci)
    fig51.update_xaxes(title_text=label, row=1, col=ci,
                       gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#e2e8f0'))
    fig51.update_yaxes(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#e2e8f0'), row=1, col=ci)
fig51.update_layout(title='5.1 Dumbbell — Impact System Prompt (⚪ Without → 🔷 With)', height=380)
H_fig51 = fig_html(sf(fig51))

# 5.2 Small Multiples
fig52 = make_subplots(rows=2, cols=2,
                      subplot_titles=['Cost Distribution','Turn Distribution','Error Rate','Token Efficiency'])
sp_colors = {0:'#94a3b8', 1:'#38bdf8'}
sp_labels = {0:'No Prompt', 1:'With Prompt'}
for sv, sl in sp_labels.items():
    sub = df[df['is_system_prompt_present']==sv]
    fig52.add_trace(go.Box(y=sub.groupby('session_id')['turn_cost'].sum(),
                           name=sl, marker_color=sp_colors[sv], boxmean=True, legendgroup=sl,
                           showlegend=sv==0), row=1, col=1)
    fig52.add_trace(go.Histogram(x=sub.groupby('session_id')['turn_number'].max(),
                                 name=sl, marker_color=sp_colors[sv], opacity=0.65,
                                 showlegend=False, legendgroup=sl,
                                 xbins=dict(start=1,end=40,size=3)), row=1, col=2)
    err_pm = sub.groupby('model')['has_error'].mean()*100
    fig52.add_trace(go.Bar(x=err_pm.index, y=err_pm.values, name=sl,
                           marker_color=sp_colors[sv], showlegend=False, legendgroup=sl), row=2, col=1)
    eff_pm = sub.groupby('model')['token_efficiency'].mean()
    fig52.add_trace(go.Bar(x=eff_pm.index, y=eff_pm.values, name=sl,
                           marker_color=sp_colors[sv], showlegend=False, legendgroup=sl), row=2, col=2)
fig52.update_layout(title='5.2 Small Multiples — With vs Without System Prompt', height=560, barmode='group')
for r in [1,2]:
    for c in [1,2]:
        fig52.update_xaxes(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#e2e8f0'), row=r, col=c)
        fig52.update_yaxes(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#e2e8f0'), row=r, col=c)
H_fig52 = fig_html(sf(fig52))

# =============================================================================
# NHÓM 6: ANOMALY DETECTION
# =============================================================================
print("🚨 Sinh nhóm 6 — Anomaly Detection...")

p95_cost    = df['turn_cost'].quantile(0.95)
p95_latency = df['latency'].quantile(0.95)
df_bub      = df[(df['latency']>0)&(df['input_tokens']>0)].copy()
df_bub['is_anomaly'] = ((df_bub['turn_cost']>p95_cost)|(df_bub['latency']>p95_latency)).astype(int)

# 6.1 Bubble Chart
fig61 = go.Figure()
for m in m_list:
    dm = df_bub[(df_bub['model']==m)&(df_bub['is_anomaly']==0)]
    fig61.add_trace(go.Scatter(
        x=dm['latency'], y=dm['turn_cost'], mode='markers', name=m,
        marker=dict(color=COLORS.get(m,'#94a3b8'),
                    size=dm['input_tokens'].clip(upper=60000)/5000+4,
                    opacity=0.55, line=dict(width=0)),
        legendgroup=m, showlegend=True
    ))
anm = df_bub[df_bub['is_anomaly']==1]
fig61.add_trace(go.Scatter(
    x=anm['latency'], y=anm['turn_cost'], mode='markers', name='⚠ Anomaly (>P95)',
    marker=dict(color='rgba(244,63,94,0.7)',
                size=anm['input_tokens'].clip(upper=60000)/5000+6,
                line=dict(color='#ff0000',width=2.5))
))
fig61.add_hline(y=p95_cost, line_dash='dash', line_color='#f43f5e',
                annotation_text='P95 Cost', annotation_font_color='#fca5a5')
fig61.add_vline(x=p95_latency, line_dash='dash', line_color='#f59e0b',
                annotation_text='P95 Latency', annotation_font_color='#fde68a')
fig61.update_layout(title='6.1 Bubble Anomaly Detection — Latency × Cost × Token Size',
                    xaxis_title='Latency (s)', yaxis_title='Turn Cost ($)')
H_fig61 = fig_html(sf(fig61))

# 6.2 SPC Control Chart
sess_ts  = sess_agg.sort_values('session_id')['total_cost'].reset_index(drop=True)
mu       = sess_ts.mean(); sigma = sess_ts.std()
ucl      = mu + 3*sigma; lcl = max(0, mu-3*sigma)
in_ctrl  = sess_ts.between(lcl, ucl)
fig62 = go.Figure()
fig62.add_trace(go.Scatter(x=sess_ts[in_ctrl].index, y=sess_ts[in_ctrl],
                           mode='markers', name='Trong kiểm soát',
                           marker=dict(color='#10b981',size=6,opacity=0.7)))
fig62.add_trace(go.Scatter(x=sess_ts[~in_ctrl].index, y=sess_ts[~in_ctrl],
                           mode='markers', name='⚠ Ngoài kiểm soát',
                           marker=dict(color='#f43f5e',size=10,symbol='x',line=dict(width=2))))
fig62.add_hline(y=mu,  line_color='#38bdf8', line_width=1.5,
                annotation_text=f'Mean=${mu:.4f}', annotation_font_color='#38bdf8')
fig62.add_hline(y=ucl, line_dash='dash', line_color='#f43f5e',
                annotation_text=f'UCL (+3σ)=${ucl:.4f}', annotation_font_color='#f43f5e')
fig62.add_hline(y=lcl, line_dash='dash', line_color='#f59e0b',
                annotation_text=f'LCL (-3σ)=${lcl:.4f}', annotation_font_color='#f59e0b')
fig62.update_layout(title='6.2 SPC Control Chart — Session Cost với UCL/LCL ±3σ',
                    xaxis_title='Session Index', yaxis_title='Total Cost ($)')
H_fig62 = fig_html(sf(fig62))

# =============================================================================
# NHÓM 7: MAGIC QUADRANT & RADAR
# =============================================================================
print("🎯 Sinh nhóm 7 — Magic Quadrant & Radar...")

# Risk Matrix (Q6)
df_pred = df.copy()
df_pred['Turn_Bins']  = pd.cut(df_pred['turn_number'], bins=[0,5,15,30,999],
                                labels=['1-5 Lượt','6-15 Lượt','16-30 Lượt','>30 Lượt'])
df_pred['Token_Bins'] = pd.cut(df_pred['input_tokens'], bins=[0,10000,20000,30000,999999],
                                labels=['<10k Tokens','10k-20k Tokens','20k-30k Tokens','>30k Tokens'])
risk_mat = df_pred.groupby(['Token_Bins','Turn_Bins'],observed=False)['has_error'].mean().unstack()*100
fig_risk = go.Figure(data=go.Heatmap(
    z=risk_mat.values, x=risk_mat.columns, y=risk_mat.index,
    colorscale=[[0,'#10b981'],[0.5,'#f59e0b'],[1,'#f43f5e']],
    text=np.round(risk_mat.values,1), texttemplate='<b>%{text}%</b>',
    textfont=dict(size=14,color='white'), colorbar=dict(tickfont=dict(color='#e2e8f0'))
))
fig_risk.update_layout(title='[Q6] Ma trận Tiên lượng Xác suất Thất bại (%)',
                       xaxis_title='Thời lượng Phiên', yaxis_title='Kích thước Token')
H_risk = fig_html(sf(fig_risk))

# Q7 Scatter Opus vs Sonnet
heavy = df[(df['output_length']>500)&(df['model'].isin(['claude-opus-4-6','claude-sonnet-4-6']))].copy()
heavy['Status'] = heavy['has_error'].map({1:'❌ Lỗi',0:'✅ Thành công'})
fig_q7 = px.scatter(heavy, x='latency', y='turn_cost', color='model',
                    symbol='Status', color_discrete_map=COLORS,
                    size='output_length', size_max=20,
                    title='[Q7] Tác vụ Nặng (>500 tokens): Opus vs Sonnet')
H_q7 = fig_html(sf(fig_q7))

# 7.1 Magic Quadrant
model_quad = sess_agg.groupby('model').agg(
    cost_eff  =('total_cost', lambda x: 1/(x.mean()+1e-9)),
    perf_score=('error_rate', lambda x: (1-x.mean())*100)
).reset_index()
xm = model_quad['cost_eff'].mean(); ym = model_quad['perf_score'].mean()
fig71 = go.Figure()
xmax = model_quad['cost_eff'].max()*1.15
for fill, x0, y0, x1, y1 in [
    ('rgba(16,185,129,0.07)', xm, ym, xmax, 105),
    ('rgba(59,130,246,0.07)',  0,  ym, xm,  105),
    ('rgba(244,63,94,0.07)',   0,   0, xm,  ym),
    ('rgba(245,158,11,0.07)', xm,   0, xmax, ym)
]:
    fig71.add_shape(type='rect', x0=x0, y0=y0, x1=x1, y1=y1, fillcolor=fill, line_width=0)
for _, row in model_quad.iterrows():
    fig71.add_trace(go.Scatter(
        x=[row['cost_eff']], y=[row['perf_score']],
        mode='markers+text', name=row['model'],
        marker=dict(size=28, color=COLORS.get(row['model'],'#94a3b8'), line=dict(color='#fff',width=2)),
        text=[row['model'].split('-')[1] if '-' in row['model'] else row['model']],
        textposition='top center', textfont=dict(color='#ffffff',size=11), showlegend=True
    ))
for txt,xf,yf,cl in [('⭐ Stars',0.85,0.97,'#10b981'),('💰 Expensive\nGood',0.15,0.97,'#3b82f6'),
                       ('💤 Cheap\nSlow',0.15,0.05,'#f59e0b'),('🚫 Avoid',0.85,0.05,'#f43f5e')]:
    fig71.add_annotation(x=xmax*xf, y=ym*2*yf if yf>0.5 else ym*yf*0.5,
                         text=txt, showarrow=False,
                         font=dict(color=cl,size=11,weight='bold'), opacity=0.7)
fig71.add_hline(y=ym, line_dash='dash', line_color='rgba(255,255,255,0.25)')
fig71.add_vline(x=xm, line_dash='dash', line_color='rgba(255,255,255,0.25)')
fig71.update_layout(title='7.1 Magic Quadrant — Model Selection Matrix',
                    xaxis_title='Cost Efficiency (1/AvgCost)', yaxis_title='Performance (%)')
H_fig71 = fig_html(sf(fig71))

# 7.2 Radar Chart
model_radar = df.groupby('model').agg(
    cost_eff  =('turn_cost',        lambda x: 1/(x.mean()+1e-9)),
    speed     =('turn_number',      lambda x: 1/(x.mean()+1e-9)),
    accuracy  =('has_error',        lambda x: (1-x.mean())*100),
    token_eff =('token_efficiency', 'mean'),
    stability =('turn_cost',        lambda x: 1/(x.std()+1e-9)),
    max_turns =('turn_number',      'max')
).reset_index()
radar_m  = ['cost_eff','speed','accuracy','token_eff','stability','max_turns']
radar_lbl= ['Cost Efficiency','Speed','Accuracy','Token Efficiency','Stability','Scalability']
for col in radar_m:
    mn,mx = model_radar[col].min(), model_radar[col].max()
    model_radar[col+'_n'] = ((model_radar[col]-mn)/(mx-mn+1e-9))*100
fig72 = go.Figure()
for _, row in model_radar.iterrows():
    vals   = [row[m+'_n'] for m in radar_m]+[row[radar_m[0]+'_n']]
    lbls   = radar_lbl+[radar_lbl[0]]
    color  = COLORS.get(row['model'],'#94a3b8')
    h = color.lstrip('#'); rc,gc,bc = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    fig72.add_trace(go.Scatterpolar(
        r=vals, theta=lbls, fill='toself', name=row['model'],
        line_color=color, fillcolor=f'rgba({rc},{gc},{bc},0.2)', opacity=0.85
    ))
fig72.update_layout(
    polar=dict(
        radialaxis=dict(visible=True,range=[0,100],
                        gridcolor='rgba(255,255,255,0.15)', tickfont=dict(color='#94a3b8',size=10)),
        angularaxis=dict(gridcolor='rgba(255,255,255,0.15)', tickfont=dict(color='#e2e8f0',size=11))
    ),
    title='7.2 Radar Chart — So sánh Đa chiều 6 Tiêu chí (Normalized 0–100)', height=550
)
H_fig72 = fig_html(sf(fig72))

# Circuit Breaker Q9
cutoffs  = list(range(1,21))
savings  = []
tot_sunk = df['sunk_cost'].sum()
for cut in cutoffs:
    saved = df[df['turn_number']>cut]['sunk_cost'].sum()
    savings.append((saved/tot_sunk)*100 if tot_sunk else 0)
fig_cb = px.line(x=cutoffs, y=savings, markers=True,
                 title='[Q9] Circuit Breaker — Đường Cong Cứu Vãn Ngân sách')
fig_cb.update_traces(line_color='#10b981', line_width=4, marker=dict(size=10))
fig_cb.update_layout(xaxis_title='Turn Cutoff', yaxis_title='% Ngân sách Chìm Bảo toàn')
H_cb = fig_html(sf(fig_cb))

# Routing Matrix Q8
route_mat = df.groupby(['context_size','task_size'],observed=False)['success'].mean().unstack()*100
fig_route = go.Figure(data=go.Heatmap(
    z=route_mat.values, x=route_mat.columns, y=route_mat.index,
    colorscale=[[0,'#f43f5e'],[0.5,'#f59e0b'],[1,'#10b981']],
    text=np.round(route_mat.values,1), texttemplate='<b>%{text}%</b>',
    textfont=dict(color='white',size=14), colorbar=dict(tickfont=dict(color='#e2e8f0'))
))
fig_route.update_layout(title='[Q8] Ma trận Định tuyến: Tỷ lệ Thành công (%)',
                         xaxis_title='Task Size', yaxis_title='Context Size')
H_route = fig_html(sf(fig_route))

# =============================================================================
# ASSEMBLE HTML
# =============================================================================
print("🏗️  Đang lắp ráp HTML Template...")

HTML = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Agent Diagnostic Intelligence — Telemetry 05–08/2026</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Plus Jakarta Sans',sans-serif;background:#0f172a;color:#fff;padding:20px 24px}}
.gradient-text{{background:linear-gradient(135deg,#7dd3fc,#818cf8,#d8b4fe);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:800;font-size:2.2rem;margin-bottom:8px}}
.subtitle{{color:#94a3b8;font-size:1rem;margin-bottom:28px}}
/* KPIs */
.kpi-row{{display:flex;gap:14px;margin-bottom:32px;flex-wrap:wrap}}
.kpi{{flex:1;min-width:150px;background:rgba(51,65,85,0.7);border:1px solid rgba(255,255,255,0.12);border-radius:14px;padding:18px 20px;transition:all .3s}}
.kpi:hover{{transform:translateY(-3px);border-color:rgba(56,189,248,0.4)}}
.kpi-label{{color:#fff;font-weight:600;font-size:.95rem;margin-bottom:8px}}
.kpi-value{{color:#fff;font-weight:800;font-size:1.7rem}}
.kpi-sub{{color:#f43f5e;font-size:.85rem;margin-top:4px}}
/* Tabs */
.tab-bar{{display:flex;gap:10px;border-bottom:2px solid rgba(56,189,248,0.2);margin-bottom:24px;flex-wrap:wrap}}
.tab-btn{{background:#1e293b;border:2px solid rgba(148,163,184,0.2);border-bottom:none;color:#94a3b8;padding:12px 22px;font-size:1rem;font-weight:700;border-radius:10px 10px 0 0;cursor:pointer;font-family:inherit;transition:all .25s}}
.tab-btn.active{{background:linear-gradient(180deg,rgba(56,189,248,0.22),rgba(14,165,233,0.04));color:#38bdf8;border-color:#38bdf8}}
.tab-btn:hover:not(.active){{color:#fff;background:rgba(56,189,248,0.08)}}
.tab-pane{{display:none}}.tab-pane.active{{display:block}}
/* Insight box */
.ibox{{background:rgba(30,41,59,.85);border:1px solid rgba(56,189,248,.35);border-left:5px solid #38bdf8;padding:20px 24px;border-radius:12px;margin-bottom:22px;line-height:1.75;color:#f1f5f9;font-size:1rem}}
.ibox.red{{border-left-color:#f43f5e}}.ibox.amber{{border-left-color:#f59e0b}}.ibox.green{{border-left-color:#10b981}}
.ibox h4{{color:#fff;margin-bottom:12px;font-weight:700;text-transform:uppercase;letter-spacing:.4px}}
/* Group header */
.gh{{background:linear-gradient(90deg,rgba(56,189,248,.13),transparent);border-left:4px solid #38bdf8;padding:9px 16px;border-radius:0 10px 10px 0;margin:22px 0 14px;font-size:1.05rem;font-weight:700;color:#7dd3fc}}
/* Grid */
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px}}
.g3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px;margin-bottom:18px}}
.chart-box{{width:100%;min-height:380px;background:rgba(15,23,42,.5);border-radius:12px;overflow:hidden}}
.chart-full{{width:100%;background:rgba(15,23,42,.5);border-radius:12px;overflow:hidden;margin-bottom:18px}}
.divider{{height:1px;background:linear-gradient(90deg,transparent,rgba(148,163,184,.35),transparent);margin:28px 0}}
@media(max-width:900px){{.g2,.g3{{grid-template-columns:1fr}}.kpi{{min-width:130px}}}}
</style>
</head>
<body>

<div class="gradient-text">🧠 AI Agent Diagnostic Intelligence</div>
<div class="subtitle">Báo cáo Phân tích Chi phí & Hiệu năng Hoạt động của AI Agent — Telemetry 05/2026–08/2026</div>

<div class="kpi-row">
  <div class="kpi"><div class="kpi-label">Tổng Lượt (Turns)</div><div class="kpi-value">{kpi['turns']}</div></div>
  <div class="kpi"><div class="kpi-label">Số Phiên (Sessions)</div><div class="kpi-value">{kpi['sessions']}</div></div>
  <div class="kpi"><div class="kpi-label">Tỷ lệ Lỗi Tổng</div><div class="kpi-value">{kpi['err_rate']}</div></div>
  <div class="kpi"><div class="kpi-label">Tổng Ngân sách</div><div class="kpi-value">{kpi['budget']}</div></div>
  <div class="kpi"><div class="kpi-label">Chi phí Chìm (Sunk)</div><div class="kpi-value">{kpi['sunk']}</div><div class="kpi-sub">▲ {kpi['sunk_pct']} lãng phí</div></div>
</div>

<div class="tab-bar">
  <button class="tab-btn active" onclick="openTab(event,'t1')">📊 1. Descriptive</button>
  <button class="tab-btn" onclick="openTab(event,'t2')">🔍 2. Diagnostic</button>
  <button class="tab-btn" onclick="openTab(event,'t3')">🔮 3. Predictive</button>
  <button class="tab-btn" onclick="openTab(event,'t4')">💡 4. Prescriptive</button>
</div>

<!-- ========== TAB 1: DESCRIPTIVE ========== -->
<div id="t1" class="tab-pane active">
  <div class="ibox"><h4>📊 Cấp 1: Descriptive — "Điều gì đã xảy ra?"</h4>
    Hệ thống biểu đồ 3 nhóm (9 charts) phác họa bức tranh toàn cảnh: phân bổ chi phí, hình thái phiên, hiệu năng theo domain và động lực học token theo lượt chat.
  </div>

  <div class="gh">📦 NHÓM 1 — TỔNG QUAN HIỆU NĂNG & CHI PHÍ</div>
  <div class="chart-full">{H_fig11}</div>
  <div class="g2"><div class="chart-box">{H_fig12}</div><div class="chart-box">{H_fig13}</div></div>

  <div class="divider"></div>
  <div class="gh">🗂️ NHÓM 2 — PHÂN TÍCH THEO DOMAIN</div>
  <div class="chart-full">{H_fig21}</div>
  <div class="g2"><div class="chart-box">{H_fig22}</div><div class="chart-box">{H_fig23}</div></div>

  <div class="divider"></div>
  <div class="gh">⚡ NHÓM 3 — ĐỘNG LỰC HỌC (DYNAMICS)</div>
  <div class="g2"><div class="chart-box">{H_fig31}</div><div class="chart-box">{H_fig32}</div></div>
  <div class="chart-full">{H_fig33}</div>
</div>

<!-- ========== TAB 2: DIAGNOSTIC ========== -->
<div id="t2" class="tab-pane">
  <div class="ibox red"><h4>🔍 Cấp 2: Diagnostic — "Tại sao điều đó xảy ra?"</h4>
    Ba nhóm biểu đồ (7 charts) bóc tách nguyên nhân gốc rễ: looping pattern, tương quan biến số, tác động System Prompt và phát hiện phiên bất thường.
  </div>

  <div class="gh">🔬 NHÓM 4 — CHẨN ĐOÁN (LOOPING & CORRELATION)</div>
  <div class="g2"><div class="chart-box">{H_fig41}</div><div class="chart-box">{H_fig42}</div></div>
  <div class="chart-full">{H_fig43}</div>

  <div class="divider"></div>
  <div class="gh">🧪 NHÓM 5 — TÁC ĐỘNG SYSTEM PROMPT</div>
  <div class="chart-full">{H_fig51}</div>
  <div class="chart-full">{H_fig52}</div>

  <div class="divider"></div>
  <div class="gh">🚨 NHÓM 6 — PHÁT HIỆN DỊ THƯỜNG (ANOMALY)</div>
  <div class="g2"><div class="chart-box">{H_fig61}</div><div class="chart-box">{H_fig62}</div></div>
</div>

<!-- ========== TAB 3: PREDICTIVE ========== -->
<div id="t3" class="tab-pane">
  <div class="ibox amber"><h4>🔮 Cấp 3: Predictive — "Điều gì sẽ xảy ra?"</h4>
    Ma trận rủi ro, magic quadrant và radar chart định lượng xác suất thất bại và so sánh chiến lược chọn model đa chiều.
  </div>

  <div class="gh">🎯 NHÓM 7 — TỔNG HỢP RA QUYẾT ĐỊNH</div>
  <div class="chart-full">{H_risk}</div>
  <div class="g2"><div class="chart-box">{H_q7}</div><div class="chart-box">{H_fig71}</div></div>
  <div class="chart-full">{H_fig72}</div>
</div>

<!-- ========== TAB 4: PRESCRIPTIVE ========== -->
<div id="t4" class="tab-pane">
  <div class="ibox green"><h4>💡 Cấp 4: Prescriptive — "Chúng ta nên làm gì?"</h4>
    Ma trận định tuyến Smart Routing, đường cong Circuit Breaker và hướng dẫn Micro-Tasking cung cấp bản đồ hành động cụ thể.
  </div>
  <div class="g2"><div class="chart-box">{H_route}</div><div class="chart-box">{H_cb}</div></div>
  <div class="ibox">
    <h4>💻 [Q10] Kiến trúc Micro-Tasking Pipeline</h4>
    <pre style="background:#0f172a;padding:16px;border-radius:8px;overflow-x:auto;color:#7dd3fc;font-size:.9rem">
def micro_task_pipeline(file_content, target_bug):
    # 1. Trích xuất cục bộ (< 2k tokens)
    local_context = extract_function(file_content, target_bug)
    # 2. Xóa System Prompt dư thừa
    prompt = build_lightweight_prompt(local_context)
    # 3. Giao task cho AI giá rẻ (Deepseek / Minimax)
    patch = route_to_cheap_model(prompt).generate(prompt)
    # 4. Xác thực bằng Opus nếu cần thiết
    return apply_patch(file_content, patch)
    </pre>
  </div>
</div>

<script>
function openTab(evt, tabId) {{
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(tabId).classList.add('active');
  evt.currentTarget.classList.add('active');
  // Trigger Plotly resize để charts render đúng kích thước
  setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
}}
</script>
</body>
</html>"""

# ─── Ghi file ─────────────────────────────────────────────────────────────────
print(f"💾 Đang ghi file → {OUTPUT_FILE}")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write(HTML)

size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
print(f"✅ XONG! File: dist/index.html  ({size_mb:.1f} MB)")
print()
print("─" * 60)
print("🚀 CÁCH DEPLOY LÊN CLOUDFLARE PAGES:")
print("   1. Vào https://dash.cloudflare.com → Workers & Pages")
print("   2. Create → Pages → Upload assets")
print("   3. Kéo thả thư mục 'dist/' vào ô upload")
print("   4. Bấm Deploy → Nhận URL ngay lập tức")
print("─" * 60)
