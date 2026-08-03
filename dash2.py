import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Executive Dashboard", layout="wide", initial_sidebar_state="expanded")

with st.sidebar:
    st.markdown("### ⚙️ Cài đặt")
    dark_mode = st.toggle("🌙 Giao diện Tối", value=False)

if dark_mode:
    THEME = {
        'bg': '#0E1117',
        'card_bg': '#1E1E2E',
        'text_main': '#FFFFFF',
        'text_sub': '#9CA3AF',
        'border': '#334155',
        'title': '#93C5FD',
        'story_bg': '#1E293B',
        'exec_bg': '#1E1E2E',
        'grid': '#334155',
        'table_row': '#1E1E2E',
        'table_row_alt': '#0F172A',
        'kpi_val': '#F8FAFC'
    }
else:
    THEME = {
        'bg': '#F4F5F7',
        'card_bg': '#FFFFFF',
        'text_main': '#111827',
        'text_sub': '#6B7280',
        'border': '#E3E6EA',
        'title': '#1F4E79',
        'story_bg': '#F8FAFC',
        'exec_bg': '#FFFFFF',
        'grid': '#ECEEF1',
        'table_row': '#FFFFFF',
        'table_row_alt': '#F8F9FA',
        'kpi_val': '#1F2937'
    }

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600;700&display=swap');
    
    html, body, [class*="css"], .stMarkdown {{ font-family: 'Segoe UI', system-ui, sans-serif !important; }}
    
    .stApp {{ background-color: {THEME['bg']}; color: {THEME['text_main']}; }}
    
    .kpi-row {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 16px; margin-bottom: 24px; }}
    .kpi-card {{ background: {THEME['card_bg']}; border: 1px solid {THEME['border']}; border-radius: 4px; padding: 16px 20px; height: 96px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); transition: transform 0.2s, box-shadow 0.2s; }}
    .kpi-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); border-color: {THEME['text_sub']}; }}
    .kpi-label {{ color: {THEME['text_sub']}; font-size: 13px; text-transform: uppercase; font-weight: 600; margin-bottom: 4px; }}
    .kpi-val {{ color: {THEME['kpi_val']}; font-size: 30px; font-weight: 600; font-variant-numeric: tabular-nums; line-height: 1.1; }}
    .kpi-sub {{ color: {THEME['text_sub']}; font-size: 11px; margin-top: 4px; }}
    .text-red {{ color: #E15759 !important; font-weight: 600; }}
    
    .story-box {{ background: {THEME['story_bg']}; border: 1px solid {THEME['border']}; border-left: 3px solid #1F77B4; border-radius: 4px; padding: 20px; font-size: 14px; color: {THEME['text_main']}; line-height: 1.6; margin-top: 16px; margin-bottom: 16px;}}
    .exec-box {{ background: {THEME['exec_bg']}; border: 1px solid {THEME['border']}; border-top: 4px solid {THEME['title']}; border-radius: 4px; padding: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-top: 16px; }}
    
    .pbi-header {{
        background: {THEME['card_bg']};
        padding: 24px 24px;
        border-bottom: 1px solid {THEME['border']};
        margin: -4rem -4rem 2rem -4rem;
    }}
    .pbi-title {{ color: {THEME['title']}; font-size: 24px; font-weight: 600; margin: 0; }}
    .pbi-subtitle {{ color: {THEME['text_sub']}; font-size: 13px; margin-top: 2px; }}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{ gap: 24px; }}
    .stTabs [data-baseweb="tab"] {{ height: 50px; white-space: pre-wrap; background-color: transparent; border-radius: 4px 4px 0 0; gap: 1px; padding-top: 10px; padding-bottom: 10px; color: {THEME['text_sub']}; font-weight: 600; font-size: 14px; }}
    .stTabs [aria-selected="true"] {{ color: {THEME['title']} !important; border-bottom: 3px solid {THEME['title']}; }}
    
    /* Table styling */
    table {{ width: 100%; text-align: left; font-size: 14px; border-collapse: collapse; color: {THEME['text_main']}; }}
    th {{ padding: 8px 0; color: {THEME['text_sub']}; border-bottom: 1px solid {THEME['border']}; }}
    td {{ padding: 12px 0; border-bottom: 1px solid {THEME['border']}; }}
</style>
""", unsafe_allow_html=True)

COLORS = { 'claude-sonnet-4-6': '#1F77B4', 'claude-opus-4-6': '#2CA089', 'deepseek-v3.1': '#EDB120', 'minimax-m2.5': '#E15759', 'muted': '#9CA3AF' }
PLOT_CONFIG = dict(displayModeBar=False)

def format_currency(v): return f"${v:,.2f}"
def format_percent(v): return f"{v:.1f}%"
def format_int(v): return f"{int(v):,}"

def apply_layout(fig, title, height=350):
    fig.update_layout(
        height=height,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Segoe UI', color=THEME['text_main'], size=15),
        title=dict(text=title, font=dict(color=THEME['title'], size=22, weight='bold'), x=0, y=0.98),
        margin=dict(l=24, r=24, t=90, b=24),
        xaxis=dict(showgrid=False, zeroline=False, showline=False, color=THEME['text_main'], tickfont=dict(size=15, weight='bold', color=THEME['text_main']), title_font=dict(size=16, weight='bold')),
        yaxis=dict(showgrid=True, gridcolor=THEME['grid'], zeroline=False, showline=False, color=THEME['text_main'], tickfont=dict(size=15, weight='bold', color=THEME['text_main']), title_font=dict(size=16, weight='bold')),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=15, weight='bold', color=THEME['text_main'])),
        hovermode='closest'
    )
    return fig

@st.cache_data
def load_data():
    df = pd.read_csv('processed_agentic_traces.csv')
    cols = ['output_length', 'pre_gap', 'has_error', 'turn_cost', 'turn_number', 'input_tokens', 'is_system_prompt_present']
    for c in cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    if 'session_id' in df.columns:
        parts = df['session_id'].str.split('__', expand=True)
        df['benchmark'] = parts[0] if 0 in parts.columns else 'unknown'
        df['project'] = parts[1] if 1 in parts.columns else 'unknown'
    else:
        df['session_id'] = df['task_id']
        parts = df['session_id'].str.split('__', expand=True)
        df['benchmark'] = parts[0] if 0 in parts.columns else 'unknown'
        df['project'] = parts[1] if 1 in parts.columns else 'unknown'
        df['is_system_prompt_present'] = df['flag']
        df['has_error'] = 1 - df['resolved']
        df['turn_cost'] = df['cost']
        df['turn_number'] = df['step']
        
    df = df.sort_values(['session_id', 'turn_number'])
    df['cum_cost'] = df.groupby('session_id')['turn_cost'].cumsum()
    
    sess = df.groupby('session_id').agg(
        model=('model', 'first'), benchmark=('benchmark', 'first'), project=('project', 'first'),
        prompt_on=('is_system_prompt_present', 'first'), turns=('turn_number', 'max'),
        total_cost=('turn_cost', 'sum'), error_share=('has_error', 'mean')
    ).reset_index()

    sess['failed'] = (sess['error_share'] == 1.0).astype(int)
    sess['resolved'] = 1 - sess['failed']
    sess['cost_per_turn'] = np.where(sess['turns']>0, sess['total_cost']/sess['turns'], 0)

    mod = sess.groupby('model').agg(
        n_sessions=('session_id', 'count'), total_cost=('total_cost', 'sum'),
        total_turns=('turns', 'sum'), failed_rate=('failed', 'mean'), avg_turns=('turns', 'mean')
    ).reset_index()
    mod['avg_cost_session'] = np.where(mod['n_sessions']>0, mod['total_cost'] / mod['n_sessions'], 0)

    turn_err = df.groupby('model').agg(err=('has_error','sum'), n=('has_error','count')).reset_index()
    z = 1.96
    turn_err['p'] = np.where(turn_err['n']>0, turn_err['err'] / turn_err['n'], 0)
    turn_err['ci_hw'] = np.where(turn_err['n']>0, z * np.sqrt((turn_err['p']*(1-turn_err['p']))/turn_err['n'] + z**2/(4*turn_err['n']**2)) / (1 + z**2/turn_err['n']), 0)
    
    return df, sess, mod, turn_err

df, sess, mod, turn_err = load_data()

tot_cost = sess['total_cost'].sum()
tot_sess = len(sess)
tot_turns = sess['turns'].sum()
tot_waste = sess[sess['failed']==1]['total_cost'].sum()
avg_t = (tot_turns/tot_sess) if tot_sess > 0 else 0
res_rate = (1 - sess['failed'].mean()) * 100 if tot_sess > 0 else 0
avg_c = (tot_cost/tot_sess) if tot_sess > 0 else 0
waste_pct = (tot_waste/tot_cost)*100 if tot_cost > 0 else 0
s_mod = mod[mod['model']=='claude-sonnet-4-6']
son_cost = s_mod['avg_cost_session'].values[0] if len(s_mod) > 0 else 0
top_proj = sess['project'].value_counts().index[0] if len(sess['project']) else "N/A"

st.markdown(f"""
<div class="pbi-header">
    <h1 class="pbi-title">Cạm Bẫy Chi Phí & Nghịch Lý System Prompt</h1>
    <div class="pbi-subtitle">Source: processed_agentic_traces.csv · Đã hiển thị {tot_sess} sessions · Tổng ngân sách {format_currency(tot_cost)}</div>
</div>
""", unsafe_allow_html=True)

# TAB NATIVE
tab1, tab2, tab3, tab4 = st.tabs(["01 MÔ TẢ", "02 CHẨN ĐOÁN", "03 DỰ BÁO", "04 KHUYẾN NGHỊ"])

with tab1:
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card"><div class="kpi-label">Tổng Ngân Sách</div><div class="kpi-val">{format_currency(tot_cost)}</div><div class="kpi-sub">4 models · SWE-bench</div></div>
        <div class="kpi-card"><div class="kpi-label">Sessions</div><div class="kpi-val">{format_int(tot_sess)}</div><div class="kpi-sub">Top project: {top_proj}</div></div>
        <div class="kpi-card"><div class="kpi-label">Turns</div><div class="kpi-val">{format_int(tot_turns)}</div><div class="kpi-sub">TB ~{avg_t:.1f} turns/session</div></div>
        <div class="kpi-card"><div class="kpi-label">Resolve Rate</div><div class="kpi-val"><span class="{'text-red' if res_rate<50 else ''}">{'▼ ' if res_rate<50 else ''}{format_percent(res_rate)}</span></div><div class="kpi-sub">Sessions không fail 100%</div></div>
        <div class="kpi-card"><div class="kpi-label">Cost / Session</div><div class="kpi-val">{format_currency(avg_c)}</div><div class="kpi-sub">Sonnet <span class="text-red">~{format_currency(son_cost)}</span></div></div>
        <div class="kpi-card" style="border-left: 3px solid #E15759;"><div class="kpi-label">Wasted Cost</div><div class="kpi-val text-red">{format_currency(tot_waste)}</div><div class="kpi-sub">≈{waste_pct:.1f}% tổng ngân sách</div></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([6, 6])
    with c1:
        fig1 = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "domain"}]], column_widths=[0.6, 0.4])
        if len(mod) > 0:
            b_df = mod.sort_values('total_cost', ascending=True)
            fig1.add_trace(go.Bar(x=b_df['total_cost'], y=b_df['model'], orientation='h', marker_color=[COLORS.get(m, '#9CA3AF') for m in b_df['model']], text=b_df['total_cost'].apply(lambda x: f"${x:.0f}"), textposition='auto', showlegend=False), row=1, col=1)
            fig1.add_trace(go.Pie(labels=mod['model'], values=mod['total_cost'], marker_colors=[COLORS.get(m, '#9CA3AF') for m in mod['model']], hole=0.6, textinfo='none', showlegend=False), row=1, col=2)
            son_pct = (son_cost * s_mod['n_sessions'].values[0] / tot_cost * 100) if (len(s_mod)>0 and tot_cost>0) else 0
            fig1.add_annotation(text=f"Sonnet<br>{son_pct:.1f}%", x=0.88, y=0.5, showarrow=False, xref="paper", yref="paper", font=dict(size=12, color=THEME['title'], weight='bold'))
        st.plotly_chart(apply_layout(fig1, "MT1: Phân bổ Ngân sách theo Model", 350), use_container_width=True, config=PLOT_CONFIG)
    with c2:
        tbl = mod[['model', 'n_sessions', 'total_turns', 'avg_cost_session']].sort_values('total_turns', ascending=False)
        html = f"""<div style="padding: 16px; background: {THEME['card_bg']}; border: 1px solid {THEME['border']}; border-radius: 4px; height: 350px;">
        <div style='color: {THEME['title']}; font-size: 16px; font-weight: bold; margin-bottom: 12px;'>MT2: Hiệu suất Tổng quan</div>
        <table>
        <tr><th>Model</th><th>Sessions</th><th>Turns</th><th>$/Sess</th></tr>"""
        for _, r in tbl.iterrows(): html += f"<tr><td style='color: {COLORS.get(r['model'], THEME['text_main'])}; font-weight: 600;'>{r['model']}</td><td>{r['n_sessions']:,}</td><td>{r['total_turns']:,}</td><td>${r['avg_cost_session']:.2f}</td></tr>"
        html += "</table></div>"
        st.markdown(html, unsafe_allow_html=True)
        
    c3, c4 = st.columns([7, 5])
    with c3:
        fig5 = go.Figure()
        for s_id in sess['session_id'].unique()[:100]: 
            sdf = df[df['session_id'] == s_id]
            fig5.add_trace(go.Scatter(x=sdf['turn_number'], y=sdf['cum_cost'], mode='lines', line=dict(color=COLORS.get(sdf['model'].iloc[0], '#9CA3AF'), width=1), opacity=0.12, showlegend=False))
        mean_cost = df.groupby(['model', 'turn_number'])['cum_cost'].mean().reset_index()
        for m in mean_cost['model'].unique():
            mdf = mean_cost[mean_cost['model']==m]
            fig5.add_trace(go.Scatter(x=mdf['turn_number'], y=mdf['cum_cost'], mode='lines', line=dict(color=COLORS.get(m, '#9CA3AF'), width=3), name=m))
        st.plotly_chart(apply_layout(fig5, "MT5: Chi phí Tích luỹ Phiên điển hình", 350), use_container_width=True, config=PLOT_CONFIG)
    with c4:
        fig3 = go.Figure()
        tdf = turn_err.sort_values('p', ascending=False)
        fig3.add_trace(go.Bar(x=tdf['model'], y=tdf['p'], error_y=dict(type='data', array=tdf['ci_hw'], visible=True), marker_color=[COLORS.get(m, '#9CA3AF') for m in tdf['model']], text=(tdf['p']*100).round(1).astype(str)+'%', textposition='outside'))
        fig3.update_layout(yaxis_tickformat='.0%')
        st.plotly_chart(apply_layout(fig3, "MT3: Tỷ lệ Lỗi (Error Rate per Turn) + 95% CI", 350), use_container_width=True, config=PLOT_CONFIG)

    c5, c6 = st.columns([7, 5])
    with c5:
        fig4 = px.box(df, x="turn_cost", y="model", color="model", log_x=True, color_discrete_map=COLORS, orientation='h')
        st.plotly_chart(apply_layout(fig4, "MT4: Phân phối Chi phí mỗi Lượt (Log Scale)", 300), use_container_width=True, config=PLOT_CONFIG)
    with c6:
        op_cost = mod[mod['model']=='claude-opus-4-6']['avg_cost_session'].values[0] if len(mod[mod['model']=='claude-opus-4-6']) > 0 else 0
        st.markdown(f"""<div class="story-box">
            <b style="color: {THEME['title']};">STORY BOX 1: Tương phản đắt-rẻ</b><br><br>
            Sự tương phản gay gắt: Mô hình chiếm {son_pct:.1f}% ngân sách lại gây lãng phí tới <span class="text-red">${tot_waste:,.2f}</span> vì hội chứng vòng lặp.
            Ngược lại, Opus dù đơn giá cao nhưng giải quyết gọn gàng, tổng chi phí chỉ khoảng ${op_cost:.2f}/phiên. 
        </div>""", unsafe_allow_html=True)

with tab2:
    c1, c2 = st.columns([6, 6])
    with c1:
        fig_cd1 = px.scatter(sess, x="cost_per_turn", y="error_share", size="turns", color="model", color_discrete_map=COLORS, opacity=0.7, hover_data=["session_id"])
        if not sess.empty:
            med_cost = sess['cost_per_turn'].median()
            fig_cd1.add_shape(type="rect", x0=0, x1=med_cost, y0=0.5, y1=1.0, fillcolor="#E15759", opacity=0.1, line_width=0)
            fig_cd1.add_annotation(x=med_cost/2, y=0.8, text="Bẫy giá rẻ", showarrow=False, font=dict(color="#E15759", weight="bold"))
        st.plotly_chart(apply_layout(fig_cd1, "CD1: 'Bẫy giá rẻ' (X=Cost/Turn, Y=Error%, Size=Turns)", 350), use_container_width=True, config=PLOT_CONFIG)
    with c2:
        sonnet = sess[sess['model']=='claude-sonnet-4-6']
        if not sonnet.empty and len(sonnet['prompt_on'].unique()) == 2:
            s_grp = sonnet.groupby('prompt_on').agg(err=('error_share', 'mean'), fail=('failed', 'mean'), tns=('turns', 'mean'), cst=('total_cost', 'mean')).reset_index()
            tk_mean = df[df['model']=='claude-sonnet-4-6'].groupby('is_system_prompt_present')['input_tokens'].mean()
            y_metrics = ['Tokens', 'Error Rate', 'Failed Rate', 'Turns', 'Total Cost']
            def norm(v0, v1):
                mx = max(v0, v1)
                return (v0/mx*100 if mx else 0), (v1/mx*100 if mx else 0)
            t0, t1 = norm(tk_mean.get(0,0), tk_mean.get(1,0))
            e0, e1 = norm(s_grp[s_grp['prompt_on']==0]['err'].iloc[0], s_grp[s_grp['prompt_on']==1]['err'].iloc[0])
            f0, f1 = norm(s_grp[s_grp['prompt_on']==0]['fail'].iloc[0], s_grp[s_grp['prompt_on']==1]['fail'].iloc[0])
            n0, n1 = norm(s_grp[s_grp['prompt_on']==0]['tns'].iloc[0], s_grp[s_grp['prompt_on']==1]['tns'].iloc[0])
            c0, c1 = norm(s_grp[s_grp['prompt_on']==0]['cst'].iloc[0], s_grp[s_grp['prompt_on']==1]['cst'].iloc[0])
            fig_cd2 = go.Figure()
            for i, (v0, v1, lbl) in enumerate(zip([t0,e0,f0,n0,c0], [t1,e1,f1,n1,c1], y_metrics)):
                fig_cd2.add_trace(go.Scatter(x=[v0, v1], y=[lbl, lbl], mode='lines+markers', marker=dict(size=12, color=['#9CA3AF', '#E15759']), line=dict(color='#9CA3AF', width=2), showlegend=False))
            st.plotly_chart(apply_layout(fig_cd2, "CD2: Tác động System Prompt trên Sonnet (OFF → ON)", 350), use_container_width=True, config=PLOT_CONFIG)
        else: st.info("Cần dữ liệu Sonnet đủ 2 trạng thái Prompt.")

    c3, c4 = st.columns([6, 6])
    with c3:
        if 'claude-sonnet-4-6' in mod['model'].values and 'claude-opus-4-6' in mod['model'].values:
            c_son = mod[mod['model']=='claude-sonnet-4-6']['avg_cost_session'].values[0]
            c_op = mod[mod['model']=='claude-opus-4-6']['avg_cost_session'].values[0]
            t_son = mod[mod['model']=='claude-sonnet-4-6']['avg_turns'].values[0]
            t_op = mod[mod['model']=='claude-opus-4-6']['avg_turns'].values[0]
            diff = c_son - c_op
            d_turn = (t_son - t_op) * (c_son/t_son) if t_son > 0 else 0
            d_cpt = diff - d_turn
            fig_cd4 = go.Figure(go.Waterfall(orientation="v", measure=["absolute", "relative", "relative", "absolute"], x=["Sonnet", "-ΔTurns", "-ΔCost", "Opus"], y=[c_son, -d_turn, -d_cpt, c_op], connector={"line":{"color":THEME['grid']}}, decreasing={"marker":{"color":"#2CA089"}}, increasing={"marker":{"color":"#E15759"}}, totals={"marker":{"color":THEME['title']}}))
            st.plotly_chart(apply_layout(fig_cd4, "CD4: Phân rã Cost (Vì sao Opus rẻ hơn Sonnet)", 350), use_container_width=True, config=PLOT_CONFIG)
    with c4:
        if not sonnet.empty:
            top_proj_sonnet = sonnet['project'].value_counts().nlargest(10).index
            df_cd3 = sonnet[sonnet['project'].isin(top_proj_sonnet)]
            cd3_agg = df_cd3.groupby(['prompt_on', 'project']).size().reset_index(name='count')
            cd3_agg['prompt_label'] = cd3_agg['prompt_on'].map({0: 'Prompt OFF', 1: 'Prompt ON'})
            fig_cd3 = px.bar(cd3_agg, x='project', y='count', color='prompt_label', barmode='group', color_discrete_map={'Prompt OFF': '#9CA3AF', 'Prompt ON': '#E15759'})
            fig_cd3.update_xaxes(tickangle=-45, title_text="")
            fig_cd3.update_yaxes(title_text="Sessions")
            st.plotly_chart(apply_layout(fig_cd3, "CD3: Top 10 Projects (Sessions theo Prompt)", 350), use_container_width=True, config=PLOT_CONFIG)

    c5, c6 = st.columns([6, 6])
    with c5:
        top_projects_all = sess['project'].value_counts().nlargest(20).index
        hm_df = sess[sess['project'].isin(top_projects_all)].groupby(['model', 'project'])['error_share'].mean().unstack().fillna(0) * 100
        fig_cd5 = px.imshow(hm_df, text_auto=".0f", color_continuous_scale="YlOrRd", aspect="auto")
        fig_cd5.update_xaxes(showticklabels=True, tickangle=-45, tickfont=dict(size=11), title_text="")
        fig_cd5.update_yaxes(title_text="")
        st.plotly_chart(apply_layout(fig_cd5, "CD5: Heatmap Tỷ lệ Lỗi (%) - Top 20 Projects", 450), use_container_width=True, config=PLOT_CONFIG)
    with c6:
        st.markdown(f"""<div class="story-box">
            <b style="color: {THEME['title']};">STORY BOX 2: Nghịch lý Prompt</b><br><br>
            Việc ép AI tuân thủ System Prompt khắt khe đã tạo ra tác dụng ngược thảm hại. Khi bật System Prompt (CD2), lượng ngữ cảnh tăng vọt khiến mô hình bị ngộ độc. 
            Kết quả: lỗi tăng phi mã, tỷ lệ thất bại vọt lên, kéo theo thời gian xử lý và chi phí nhân gấp nhiều lần. Phân rã (CD4) chứng minh Opus tiết kiệm vì giải quyết xong nghỉ sớm.
        </div>""", unsafe_allow_html=True)

with tab3:
    c1, c2 = st.columns([6, 6])
    with c1:
        surv = []
        for m in sess['model'].unique():
            s_m = sess[sess['model']==m]['session_id']
            max_t = int(df['turn_number'].max())
            for t in range(1, max_t+1):
                active_s = df[(df['session_id'].isin(s_m)) & (df['turn_number']>=t)]['session_id'].nunique()
                if active_s > 0: surv.append({'model': m, 'turn': t, 'p': active_s/len(s_m)})
        if surv:
            surv_df = pd.DataFrame(surv)
            fig_db1 = px.line(surv_df, x="turn", y="p", color="model", color_discrete_map=COLORS, line_shape='hv')
            fig_db1.add_vrect(x0=10, x1=15, fillcolor="#E15759", opacity=0.1, line_width=0, annotation_text="Circuit Breaker")
            st.plotly_chart(apply_layout(fig_db1, "DB1: Survival Curve - Phiên còn chạy theo Turn", 350), use_container_width=True, config=PLOT_CONFIG)
    with c2:
        tk_line = df.groupby(['model', 'turn_number'])['input_tokens'].mean().reset_index()
        fig_db4 = px.line(tk_line, x="turn_number", y="input_tokens", color="model", color_discrete_map=COLORS)
        st.plotly_chart(apply_layout(fig_db4, "DB4: Sự phình to Tokens (Context Bloat)", 350), use_container_width=True, config=PLOT_CONFIG)

    c3, c4 = st.columns([6, 6])
    with c3:
        cost10 = df[df['turn_number']<=10].groupby('session_id')['turn_cost'].sum().reset_index().rename(columns={'turn_cost':'cost_10'})
        db2_df = pd.merge(sess, cost10, on='session_id')
        fig_db2 = px.scatter(db2_df, x="cost_10", y="total_cost", color="model", log_x=True, log_y=True, color_discrete_map=COLORS, opacity=0.7)
        fig_db2.add_shape(type="line", x0=0.01, y0=0.01, x1=db2_df['total_cost'].max(), y1=db2_df['total_cost'].max(), line=dict(color="#9CA3AF", dash="dash"))
        st.plotly_chart(apply_layout(fig_db2, "DB2: Dự báo Chi phí cuối cùng từ Lượt 10", 350), use_container_width=True, config=PLOT_CONFIG)
    with c4:
        s5 = df[df['turn_number']<=5].groupby('session_id').agg(err=('has_error', 'sum')).reset_index()
        s5 = pd.merge(s5, sess[['session_id', 'failed']], on='session_id')
        auc_val = 0
        if len(s5['failed'].unique()) > 1:
            lr = LogisticRegression()
            lr.fit(s5[['err']], s5['failed'])
            fpr, tpr, _ = roc_curve(s5['failed'], lr.predict_proba(s5[['err']])[:,1])
            auc_val = auc(fpr, tpr)
            fig_db3 = px.area(x=fpr, y=tpr, title='', labels=dict(x='FPR', y='TPR'), color_discrete_sequence=['#2CA089'])
            fig_db3.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(apply_layout(fig_db3, f"DB3: Cảnh báo sớm Thất bại từ 5 turns (AUC={auc_val:.2f})", 350), use_container_width=True, config=PLOT_CONFIG)

    st.markdown(f"""<div class="story-box">
        <b style="color: {THEME['title']};">STORY BOX 3: Biết trước tương lai</b> — Chỉ với 5 lượt đầu tiên, thuật toán phân loại sớm đạt độ nhạy ROC AUC {auc_val:.2f}. Kết hợp với quy luật sống sót (DB1), tỷ lệ xử lý thành công tiệm cận số 0 sau lượt thứ 15. Đây là căn cứ vững chắc để kích hoạt các biện pháp dừng cứng tự động, bảo vệ ngân sách trước tình trạng "phình to ngữ cảnh" (DB4) theo hàm mũ của hệ thống.
    </div>""", unsafe_allow_html=True)

with tab4:
    c1, c2 = st.columns([6, 6])
    with c1:
        fig_kn3 = px.scatter(mod, x="avg_cost_session", y="failed_rate", size="n_sessions", color="model", color_discrete_map=COLORS, hover_name="model", size_max=40)
        if not mod.empty:
            fig_kn3.add_hline(y=mod['failed_rate'].median(), line_dash="dash", line_color="#9CA3AF")
            fig_kn3.add_vline(x=mod['avg_cost_session'].median(), line_dash="dash", line_color="#9CA3AF")
        st.plotly_chart(apply_layout(fig_kn3, "KN3: Định tuyến Mô hình", 350), use_container_width=True, config=PLOT_CONFIG)
    with c2:
        cut15_cost = df[df['turn_number']<=15]['turn_cost'].sum()
        fig_kn1 = go.Figure(data=[
            go.Bar(name='Baseline', x=['Tổng Chi Phí'], y=[tot_cost], marker_color='#9CA3AF'),
            go.Bar(name='Ngắt Lượt 15', x=['Tổng Chi Phí'], y=[cut15_cost], marker_color='#2CA089')
        ])
        st.plotly_chart(apply_layout(fig_kn1, "KN1: Mô phỏng Ngắt mạch Turn 15", 350), use_container_width=True, config=PLOT_CONFIG)

    c3, c4 = st.columns([6, 6])
    with c3:
        if 'claude-sonnet-4-6' in sess['model'].values and len(sess[sess['model']=='claude-sonnet-4-6']['prompt_on'].unique()) == 2:
            s_grp = sess[sess['model']=='claude-sonnet-4-6'].groupby('prompt_on')['total_cost'].mean().reset_index()
            c_on = s_grp[s_grp['prompt_on']==1]['total_cost'].iloc[0]
            c_off = s_grp[s_grp['prompt_on']==0]['total_cost'].iloc[0]
            n_on = len(sess[(sess['model']=='claude-sonnet-4-6') & (sess['prompt_on']==1)])
            sv = (c_on - c_off) * n_on if c_on > c_off else 0
        else: sv = 0
        fig_kn2 = go.Figure(go.Waterfall(orientation="v", measure=["absolute", "relative", "total"], x=["Hiện tại", "Prune Prompt", "Kỳ vọng"], y=[tot_cost, -sv, tot_cost-sv], connector={"line":{"color":THEME['grid']}}, decreasing={"marker":{"color":"#2CA089"}}, totals={"marker":{"color":THEME['title']}}))
        st.plotly_chart(apply_layout(fig_kn2, "KN2: Tiết kiệm nhờ Tối giản Prompt", 350), use_container_width=True, config=PLOT_CONFIG)
    with c4:
        fig_kn4 = go.Figure(go.Waterfall(orientation="v", measure=["absolute", "relative", "relative", "total"], x=["Ngân sách Wasted", "Ngắt mạch T15", "Tối giản Prompt", "Lãng phí còn lại"], y=[tot_waste, -(tot_cost-cut15_cost), -sv, tot_waste - (tot_cost-cut15_cost) - sv], connector={"line":{"color":THEME['grid']}}, decreasing={"marker":{"color":"#2CA089"}}, totals={"marker":{"color":"#E15759"}}))
        st.plotly_chart(apply_layout(fig_kn4, "KN4: Sổ cái Thu hồi Lãng phí (ROI Ledger)", 350), use_container_width=True, config=PLOT_CONFIG)

    st.markdown(f"""
    <div class="exec-box">
        <h3 style="color: {THEME['title']}; margin-top: 0;">TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)</h3>
        <p style="color: {THEME['text_sub']}; font-size: 14px; margin-bottom: 20px;">
            Ngân sách lãng phí lên tới <b style="color: {THEME['text_main']};">${tot_waste:,.2f}</b> ({(tot_waste/tot_cost*100) if tot_cost>0 else 0:.1f}%). 
            Ngắt mạch vòng lặp (Lượt 15) và tinh giản System Prompt là 2 phương án cấp thiết để chấn chỉnh hiệu năng.
        </p>
        <table>
            <tr style="background: {THEME['table_row_alt']};"><th>Ưu tiên</th><th>Hành động Đề xuất</th><th>Tiết kiệm (ước tính)</th><th>Lead-time</th></tr>
            <tr><td style="color: #E15759; font-weight: bold;">P0</td><td>Circuit Breaker: Ngắt cứng tại lượt 15</td><td>${(tot_cost-cut15_cost):.2f}</td><td>1 Tuần</td></tr>
            <tr><td style="color: #EDB120; font-weight: bold;">P1</td><td>Pruning Prompt: Bỏ ràng buộc state-focused</td><td>${sv:.2f}</td><td>2 Tuần</td></tr>
            <tr><td style="color: #1F77B4; font-weight: bold;">P2</td><td>Smart Routing: Ưu tiên Opus cho task lớn</td><td>Chất lượng cao hơn</td><td>1 Tháng</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
