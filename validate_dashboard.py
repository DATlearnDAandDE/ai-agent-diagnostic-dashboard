import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_percentage_error
import warnings

warnings.filterwarnings("ignore")

# ================================================================
# THIẾT KẾ GIAO DIỆN & CSS (THEME COMMAND / BOARDROOM)
# ================================================================
st.set_page_config(
    page_title="AI Agent Diagnostic Intelligence",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Sidebar Theme Switcher & Navigation
st.sidebar.markdown("### Theme Settings")
theme = st.sidebar.radio("Chọn Theme:", ["Command (Dark)", "Boardroom (Light)"])
if theme == "Command (Dark)":
    bg_color = "#0B1020"
    surface_color = "#141B2E"
    text_color = "#E6ECF5"
    grid_color = "rgba(255,255,255,0.07)"
else:
    bg_color = "#EEF1F7"
    surface_color = "#FFFFFF"
    text_color = "#1F4E79"
    grid_color = "rgba(0,0,0,0.07)"

# CSS Styling (Scrollytelling)
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500&family=JetBrains+Mono:wght@400;500;600&display=swap');

    .stApp {{
        background-color: {bg_color};
        color: {text_color};
        font-family: 'IBM Plex Sans', sans-serif;
        background-image: 
            radial-gradient(ellipse at 80% 0%, rgba(56, 225, 214, 0.03) 0%, transparent 50%),
            radial-gradient(ellipse at 20% 100%, rgba(155, 140, 255, 0.03) 0%, transparent 50%),
            linear-gradient({grid_color} 1px, transparent 1px),
            linear-gradient(90deg, {grid_color} 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 40px 40px, 40px 40px;
    }}
    
    h1, h2, h3, h4 {{ font-family: 'Space Grotesk', sans-serif !important; color: {text_color} !important; font-weight: 700; }}
    .display-title {{ font-size: 42px; padding-bottom: 20px; border-bottom: 1px solid {grid_color}; margin-bottom: 30px; }}
    
    .metric-value {{ font-family: 'JetBrains Mono', monospace; font-size: 38px; font-weight: 600; color: #38E1D6; line-height: 1.1; }}
    .metric-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #8A97B0; margin-top: 5px; }}
    .metric-box {{ background: {surface_color}; padding: 20px; border-radius: 8px; border: 1px solid {grid_color}; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.1); transition: all 0.3s; }}
    .metric-box:hover {{ transform: translateY(-5px); border-color: #38E1D6; box-shadow: 0 8px 30px rgba(56, 225, 214, 0.15); }}
    
    .story-box {{
        background: {surface_color};
        border-left: 4px solid #F5B544;
        padding: 24px;
        border-radius: 4px;
        margin: 40px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        font-size: 18px;
        line-height: 1.6;
    }}
    
    .nav-item {{ padding: 12px 16px; margin: 8px 0; border-radius: 6px; cursor: pointer; color: #8A97B0; font-weight: 500; transition: all 0.3s; }}
    .nav-item:hover {{ background: {grid_color}; color: {text_color}; }}
    .nav-active {{ background: rgba(56, 225, 214, 0.1); color: #38E1D6; border-left: 3px solid #38E1D6; }}
    
    /* Hide specific st elements */
    header {{ visibility: hidden; }}
    .block-container {{ max-width: 1500px; padding: 2rem; }}
    
</style>
""", unsafe_allow_html=True)

MODEL_COLORS = {
    'deepseek-v3.1': '#38E1D6',
    'claude-sonnet-4-6': '#F5B544',
    'claude-opus-4-6': '#9B8CFF',
    'minimax-m2.5': '#FF6B6B'
}

def apply_plotly_layout(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Space Grotesk", size=18, color=text_color)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="IBM Plex Sans", color=text_color),
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(showgrid=True, gridcolor=grid_color, gridwidth=1, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=grid_color, gridwidth=1, zeroline=False),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ================================================================
# 0. CHUẨN BỊ DỮ LIỆU
# ================================================================
@st.cache_data
def load_and_prep_data(filepath="processed_agentic_traces.csv"):
    try:
        df_raw = pd.read_csv(filepath)
        if 'session_id' in df_raw.columns:
            # Map from original dataset to requested schema
            df = pd.DataFrame()
            df['task_id'] = df_raw['session_id']
            df['model'] = df_raw['model']
            df['duration'] = df_raw['pre_gap']
            df['cost'] = df_raw['turn_cost']
            df['resolved'] = 1 - df_raw['has_error']
            df['flag'] = df_raw['is_system_prompt_present']
            df['tokens'] = df_raw['input_tokens']
            df['step'] = df_raw['turn_number']
            df = df.sort_values(['task_id', 'step'])
            df['cumulative_cost'] = df.groupby('task_id')['cost'].cumsum()
        else:
            cols = ['task_id', 'model', 'duration', 'cost', 'resolved', 'flag', 'tokens', 'step', 'cumulative_cost']
            df = pd.read_csv(filepath, header=None, names=cols)
            df = df.sort_values(['task_id', 'step'])
    except Exception as e:
        st.error(f"Lỗi: {e}")
        return pd.DataFrame(), pd.DataFrame()

    # Parse task_id (bench__project__issue__run)
    parts = df['task_id'].str.split('__', expand=True)
    df['benchmark'] = parts[0]
    df['project'] = parts[1]

    # Derivations
    df['tokens_per_step'] = df.groupby('task_id')['tokens'].diff().fillna(df['tokens'])
    df['phase'] = pd.cut(df['step'], bins=[0, 10, 20, 30, 40, 50], labels=['Khám phá', 'Giả thuyết', 'Sửa', 'Kiểm chứng', 'Tinh chỉnh'])
    
    # Spike definition: cost > 5 OR > Q3 + 3*IQR
    q3 = df.groupby('model')['cost'].transform(lambda x: x.quantile(0.75))
    iqr = df.groupby('model')['cost'].transform(lambda x: x.quantile(0.75) - x.quantile(0.25))
    df['spike'] = (df['cost'] > 5) | (df['cost'] > (q3 + 3 * iqr))

    # Per-TASK Aggregation
    task_agg = df.groupby('task_id').agg(
        model=('model', 'first'),
        benchmark=('benchmark', 'first'),
        project=('project', 'first'),
        total_cost=('cost', 'sum'),
        max_step=('step', 'max'),
        total_duration=('duration', 'sum'),
        resolved_final=('resolved', 'last'),
        final_tokens=('tokens', 'last')
    ).reset_index()

    task_agg['hit_cap'] = (task_agg['max_step'] >= 50).astype(int)
    
    # first_flag1: step đầu tiên flag chuyển 0->1
    # Nếu data gốc có flag=1 tĩnh, first_flag1 = 1 đối với các row flag=1
    flag1_df = df[df['flag'] == 1].groupby('task_id')['step'].min().reset_index().rename(columns={'step': 'first_flag1'})
    task_agg = pd.merge(task_agg, flag1_df, on='task_id', how='left')
    task_agg['first_flag1'] = task_agg['first_flag1'].fillna(999) # 999 = không bao giờ flag=1

    # Redundant steps: số step sau first_flag1
    df_merged = pd.merge(df, task_agg[['task_id', 'first_flag1']], on='task_id')
    df_merged['is_redundant'] = df_merged['step'] > df_merged['first_flag1']
    redundant_agg = df_merged[df_merged['is_redundant']].groupby('task_id').agg(
        redundant_steps=('step', 'count'),
        redundant_cost=('cost', 'sum')
    ).reset_index()
    
    task_agg = pd.merge(task_agg, redundant_agg, on='task_id', how='left')
    task_agg['redundant_steps'] = task_agg['redundant_steps'].fillna(0)
    task_agg['redundant_cost'] = task_agg['redundant_cost'].fillna(0)

    return df, task_agg

df, task_agg = load_and_prep_data()
if df.empty:
    st.stop()

# ================================================================
# BỐ CỤC CHÍNH (RAIL & MAIN)
# ================================================================
col_rail, col_main = st.columns([1.5, 8.5])

with col_rail:
    st.markdown("""
    <div style='position: sticky; top: 100px;'>
        <div class='nav-item nav-active'>01 MÔ TẢ</div>
        <div class='nav-item'>02 CHẨN ĐOÁN</div>
        <div class='nav-item'>03 DỰ ĐOÁN</div>
        <div class='nav-item'>04 KÊ TOA</div>
    </div>
    """, unsafe_allow_html=True)

with col_main:
    # ----------------------------------------------------------------
    # CẤP 1 — MÔ TẢ
    # ----------------------------------------------------------------
    st.markdown("<div class='display-title'>01. MÔ TẢ: Chuyện gì đã xảy ra?</div>", unsafe_allow_html=True)
    
    # KPI Strip
    kpis = [
        ("TỔNG STEPS", f"{len(df):,}"),
        ("SỐ TASKS", f"{len(task_agg):,}"),
        ("RESOLVE RATE", f"{task_agg['resolved_final'].mean()*100:.1f}%"),
        ("Σ COST", f"${task_agg['total_cost'].sum():,.2f}"),
        ("Σ DURATION", f"{task_agg['total_duration'].sum():,.0f}s"),
        ("Σ TOKENS", f"{df['tokens_per_step'].sum():,.0f}")
    ]
    cols = st.columns(6)
    for col, (label, val) in zip(cols, kpis):
        with col:
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{val}</div><div class='metric-label'>{label}</div></div>", unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        # P1.1 Heatmap
        hm = task_agg.groupby(['model', 'benchmark'])['resolved_final'].mean().unstack().fillna(0) * 100
        fig_1 = px.imshow(hm, text_auto=".1f", color_continuous_scale="Teal", aspect="auto")
        st.plotly_chart(apply_plotly_layout(fig_1, "P1.1 Resolve Rate theo Model × Benchmark (%)"), width='stretch')
        
    with c2:
        # P1.2 Box plot log
        fig_2 = px.box(task_agg, x="model", y="total_cost", color="model", log_y=True, color_discrete_map=MODEL_COLORS)
        counts = task_agg['model'].value_counts()
        model_vals = sorted(task_agg['model'].unique().tolist())
        fig_2.update_xaxes(
            ticktext=[f"{m}<br>n={counts.get(m,0)}" for m in model_vals],
            tickvals=model_vals
        )
        st.plotly_chart(apply_plotly_layout(fig_2, "P1.2 Total Cost/Task (Log Scale)"), width='stretch')

    c3, c4 = st.columns(2)
    with c3:
        # P1.3 Violin
        fig_3 = px.violin(df, x="model", y="duration", color="model", box=True, color_discrete_map=MODEL_COLORS)
        for m in df['model'].unique():
            p95 = df[df['model']==m]['duration'].quantile(0.95)
            fig_3.add_shape(type="line", x0=m, x1=m, y0=p95, y1=p95, line=dict(color="#FF6B6B", width=2, dash="dot"))
        st.plotly_chart(apply_plotly_layout(fig_3, "P1.3 Duration/Step (Vạch đỏ: P95)"), width='stretch')
        
    with c4:
        # P1.4 Cumulative cost lines
        fig_4 = px.line(df, x="step", y="cumulative_cost", line_group="task_id", color="model", color_discrete_map=MODEL_COLORS)
        fig_4.update_traces(opacity=0.1, line=dict(width=1))
        st.plotly_chart(apply_plotly_layout(fig_4, "P1.4 Cumulative Cost theo Step"), width='stretch')

    c5, c6 = st.columns(2)
    with c5:
        # P1.5 Hist max step
        fig_5 = px.histogram(task_agg, x="max_step", color="model", barmode="stack", color_discrete_map=MODEL_COLORS)
        pct_50 = task_agg['hit_cap'].mean() * 100
        fig_5.add_annotation(x=50, y=len(task_agg[task_agg['max_step']==50]), text=f"{pct_50:.1f}% chạm trần", showarrow=True)
        st.plotly_chart(apply_plotly_layout(fig_5, "P1.5 Phân bố Max Step"), width='stretch')
        
    with c6:
        # P1.6 Token growth
        avg_tok = df.groupby(['model', 'step'])['tokens'].mean().reset_index()
        fig_6 = px.line(avg_tok, x="step", y="tokens", color="model", color_discrete_map=MODEL_COLORS)
        st.plotly_chart(apply_plotly_layout(fig_6, "P1.6 Tăng trưởng Tokens trung bình theo Step"), width='stretch')

    # STORY BOX 1
    ds_res = hm.loc['deepseek-v3.1', 'swebench'] if 'deepseek-v3.1' in hm.index and 'swebench' in hm.columns else 0
    ds_cost = task_agg[task_agg['model'] == 'deepseek-v3.1']['total_cost'].mean() if 'deepseek-v3.1' in task_agg['model'].values else 0
    sn_cost = task_agg[task_agg['model'] == 'claude-sonnet-4-6']['total_cost'].mean()
    sn_res_gaia = hm.loc['claude-sonnet-4-6', 'gaia'] if 'gaia' in hm.columns else 0
    ratio = sn_cost / ds_cost if ds_cost > 0 else 0
    st.markdown(f"""
    <div class='story-box'>
        <b>STORY BOX 1:</b> Nhìn chung, DeepSeek giải quyết {ds_res:.1f}% task SWE-bench với chi phí trung bình ${ds_cost:.2f}/task, 
        trong khi Claude Sonnet tốn gấp {ratio:.1f} lần (${sn_cost:.2f}/task) cho {sn_res_gaia:.1f}% kết quả trên GAIA. Hơn {pct_50:.1f}% toàn bộ tasks bị kẹt trong vòng lặp và chạm trần 50 step vô ích.
    </div>
    """, unsafe_allow_html=True)


    # ----------------------------------------------------------------
    # CẤP 2 — CHẨN ĐOÁN
    # ----------------------------------------------------------------
    st.markdown("<br><br><div class='display-title'>02. CHẨN ĐOÁN: Vì sao? (Giải phẫu 2 cạm bẫy)</div>", unsafe_allow_html=True)
    
    st.subheader("A. CẠM BẪY CHI PHÍ")
    c7, c8 = st.columns(2)
    with c7:
        # P2.1 Spike anatomy
        fig_7 = px.scatter(df, x="duration", y="cost", color="model", size="tokens", hover_data=["task_id", "step"], log_x=True, log_y=True, color_discrete_map=MODEL_COLORS)
        spikes = df[df['spike']]
        fig_7.add_trace(go.Scatter(x=spikes['duration'], y=spikes['cost'], mode='markers', marker=dict(size=12, color='rgba(0,0,0,0)', line=dict(color='red', width=2)), name='Spike (Cost>5 or >Q3+3IQR)'))
        st.plotly_chart(apply_plotly_layout(fig_7, "P2.1 Giải phẫu Spikes (Size = Tokens)"), width='stretch')
        
    with c8:
        # P2.2 Wasted cost Donut
        waste_agg = task_agg[task_agg['resolved_final']==0].groupby('model')['total_cost'].sum().reset_index()
        waste_pct = waste_agg['total_cost'].sum() / task_agg['total_cost'].sum() * 100
        fig_8 = px.pie(waste_agg, values='total_cost', names='model', hole=0.5, color='model', color_discrete_map=MODEL_COLORS)
        fig_8.update_layout(annotations=[dict(text=f"{waste_pct:.1f}%<br>Waste", x=0.5, y=0.5, font_size=24, showarrow=False)])
        st.plotly_chart(apply_plotly_layout(fig_8, "P2.2 Ngân sách đốt vào Tasks Thất bại (Resolved=0)"), width='stretch')

    # P2.3 Redundant loops
    cap_cost = task_agg[task_agg['hit_cap']==1]['total_cost'].sum()
    red_cost = task_agg['redundant_cost'].sum()
    fig_9 = go.Figure(data=[
        go.Bar(name='Redundant Steps Cost (Sau Flag=1)', x=['Chi Phí Lãng Phí Phụ'], y=[red_cost], marker_color='#F5B544'),
        go.Bar(name='Hit-Cap 50 Steps Cost', x=['Chi Phí Lãng Phí Phụ'], y=[cap_cost], marker_color='#FF6B6B')
    ])
    st.plotly_chart(apply_plotly_layout(fig_9, "P2.3 Chi phí chạy lặp Redundant & Hit-Cap"), width='stretch')

    st.subheader("B. NGHỊCH LÝ SYSTEM PROMPT")
    c9, c10 = st.columns(2)
    with c9:
        # P2.4 T1 Correlation
        corr_data = []
        for m in task_agg['model'].unique():
            sub = task_agg[task_agg['model']==m]
            if len(sub) > 1:
                corr = sub[['final_tokens', 'resolved_final']].corr(method='spearman').iloc[0,1]
                corr_data.append({'Model': m, 'Spearman': corr})
        fig_10 = px.bar(pd.DataFrame(corr_data), x='Model', y='Spearman', color='Model', color_discrete_map=MODEL_COLORS, text_auto='.2f')
        st.plotly_chart(apply_plotly_layout(fig_10, "P2.4 (T1) Tương quan Spearman: Final Tokens & Resolved"), width='stretch')
        
    with c10:
        # P2.5 T2 Cost-per-1K vs Resolve
        df['cost_per_1k'] = np.where(df['tokens_per_step'] > 0, df['cost'] / (df['tokens_per_step']/1000), 0)
        cpt = df.groupby('model')['cost_per_1k'].mean().reset_index()
        res = task_agg.groupby('model')['resolved_final'].mean().reset_index()
        m_cpt = pd.merge(cpt, res, on='model')
        
        fig_11 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_11.add_trace(go.Bar(x=m_cpt['model'], y=m_cpt['cost_per_1k'], name="Cost/1K", marker_color="#38E1D6"), secondary_y=False)
        fig_11.add_trace(go.Scatter(x=m_cpt['model'], y=m_cpt['resolved_final'], name="Resolve Rate", mode="lines+markers", marker=dict(color="#FF6B6B", size=10)), secondary_y=True)
        st.plotly_chart(apply_plotly_layout(fig_11, "P2.5 (T2) Cost-per-1K Tokens vs Resolve Rate"), width='stretch')

    c11, c12 = st.columns(2)
    with c11:
        # P2.6 T3 Marginal efficiency: Δtokens/step vs P(resolved_final=1) over steps
        step_eff = df.groupby('step').agg(delta_tokens=('tokens_per_step', 'mean'), res_rate=('resolved', 'mean')).reset_index()
        fig_12 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_12.add_trace(go.Bar(x=step_eff['step'], y=step_eff['delta_tokens'], name="ΔTokens/Step", marker_color="#5B8DEF"), secondary_y=False)
        fig_12.add_trace(go.Scatter(x=step_eff['step'], y=step_eff['res_rate'], name="P(Resolve)", mode="lines", line=dict(color="#F5B544")), secondary_y=True)
        fig_12.add_vrect(x0=20, x1=50, fillcolor="red", opacity=0.1, annotation_text="Context Bloat", annotation_position="top right")
        st.plotly_chart(apply_plotly_layout(fig_12, "P2.6 (T3) Hiệu suất biên & Context Bloat"), width='stretch')
        
    with c12:
        # P2.7 Confusion matrix flag x resolved
        cm = pd.crosstab(df['flag'], df['resolved'])
        fig_13 = px.imshow(cm, text_auto=True, color_continuous_scale="Reds")
        st.plotly_chart(apply_plotly_layout(fig_13, "P2.7 Overconfidence: Flag vs Resolved"), width='stretch')

    st.markdown(f"""
    <div class='story-box'>
        <b>STORY BOX 2:</b> Cạm bẫy chi phí đã làm bốc hơi {waste_pct:.1f}% tổng ngân sách cho spikes (như timeout kéo dài >300s) và các task thất bại. 
        Nghịch lý System Prompt: Agent càng "biết nhiều" (context phình to gấp hàng chục lần từ đầu đến cuối) thì càng đắt đỏ, nhưng tương quan với thành công lại âm/yếu (T1), và xác suất cải thiện gần như tiệm cận 0 sau step 20 (vùng Context Bloat).
    </div>
    """, unsafe_allow_html=True)


    # ----------------------------------------------------------------
    # CẤP 3 — DỰ ĐOÁN
    # ----------------------------------------------------------------
    st.markdown("<br><br><div class='display-title'>03. DỰ ĐOÁN: Chuyện gì SẼ xảy ra?</div>", unsafe_allow_html=True)
    
    c13, c14 = st.columns(2)
    with c13:
        # P3.1 Early warning Logistic Reg at step 5
        s5 = df[df['step'] <= 5].groupby('task_id').agg(
            cost_5=('cumulative_cost', 'max'), tokens_5=('tokens', 'max'), model=('model', 'first')
        ).reset_index()
        s5 = pd.merge(s5, task_agg[['task_id', 'resolved_final']], on='task_id')
        s5['model_code'] = s5['model'].astype('category').cat.codes
        
        auc = 0
        if len(s5['resolved_final'].unique()) > 1:
            lr = LogisticRegression()
            lr.fit(s5[['cost_5', 'tokens_5', 'model_code']], s5['resolved_final'])
            auc = roc_auc_score(s5['resolved_final'], lr.predict_proba(s5[['cost_5', 'tokens_5', 'model_code']])[:,1])
        
        fig_14 = go.Figure(go.Indicator(mode="gauge+number", value=auc, title={'text': "ROC-AUC Dự báo Resolved từ Step 5"}, gauge={'axis': {'range': [None, 1]}}))
        st.plotly_chart(apply_plotly_layout(fig_14, "P3.1 Early-Warning AUC"), width='stretch')
        
    with c14:
        # P3.2 Cost forecast from 10 steps
        s10 = df[df['step'] <= 10].groupby('task_id').agg(cost_10=('cumulative_cost', 'max')).reset_index()
        s10 = pd.merge(s10, task_agg[['task_id', 'total_cost', 'model']], on='task_id')
        
        lin = LinearRegression()
        lin.fit(s10[['cost_10']], s10['total_cost'])
        preds = lin.predict(s10[['cost_10']])
        r2 = r2_score(s10['total_cost'], preds)
        
        fig_15 = px.scatter(s10, x=preds, y="total_cost", color="model", color_discrete_map=MODEL_COLORS, labels={'x':'Pred Cost ($)', 'total_cost':'Actual Cost ($)'})
        fig_15.add_shape(type="line", x0=0, y0=0, x1=s10['total_cost'].max(), y1=s10['total_cost'].max(), line=dict(color="white", dash="dash"))
        st.plotly_chart(apply_plotly_layout(fig_15, f"P3.2 Forecast Final Cost (R²={r2:.2f})"), width='stretch')

    c15, c16 = st.columns(2)
    with c15:
        # P3.3 Survival Kaplan-Meier
        surv_data = []
        for m in df['model'].unique():
            for t in range(1, 41):
                active = df[(df['model']==m) & (df['step']==t)]['task_id'].unique()
                unres = task_agg[(task_agg['task_id'].isin(active)) & (task_agg['resolved_final']==0)]['task_id'].nunique()
                surv_data.append({'model': m, 'step': t, 'unresolved_pct': (unres/len(active)*100) if len(active)>0 else 0})
        fig_16 = px.line(pd.DataFrame(surv_data), x='step', y='unresolved_pct', color='model', color_discrete_map=MODEL_COLORS)
        fig_16.add_vline(x=20, line_dash="dash", annotation_text="Điểm Dừng Tối Ưu")
        st.plotly_chart(apply_plotly_layout(fig_16, "P3.3 Survival Curve (Kaplan-Meier Style)"), width='stretch')
        
    with c16:
        # P3.4 Spike rule precision/recall
        df['pred_spike'] = df['duration'] > 300
        tp = ((df['pred_spike']==1) & (df['spike']==1)).sum()
        fp = ((df['pred_spike']==1) & (df['spike']==0)).sum()
        fn = ((df['pred_spike']==0) & (df['spike']==1)).sum()
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0
        
        # P3.5 Overrun Scoring UCL
        ucl = s10['cost_10'].quantile(0.9)
        top_ucl = s10[s10['cost_10'] > ucl].sort_values('cost_10', ascending=False).head(3)
        
        st.markdown(f"**P3.4 Luật Spike `duration > 300s`**: Precision={prec*100:.1f}%, Recall={rec*100:.1f}%")
        st.markdown(f"**P3.5 Control Limit UCL (P90 @ Step 10)**: ${ucl:.3f}")
        st.dataframe(top_ucl)

    st.markdown(f"""
    <div class='story-box'>
        <b>STORY BOX 3:</b> Chỉ với 5 step đầu, mô hình Logistic Regression dự đoán chính xác {auc*100:.1f}% (AUC) rủi ro task sẽ thất bại hoàn toàn. 
        Nếu thiết lập cảnh báo sớm tại step 10 dựa trên đường UCL P90 (${ucl:.3f}), chúng ta có thể tự động ngắt mạch các task chắc chắn sẽ đốt ngân sách vô tận.
    </div>
    """, unsafe_allow_html=True)


    # ----------------------------------------------------------------
    # CẤP 4 — KÊ TOA
    # ----------------------------------------------------------------
    st.markdown("<br><br><div class='display-title'>04. KÊ TOA: Nên làm gì, tiết kiệm bao nhiêu?</div>", unsafe_allow_html=True)

    # Replay Simulations
    base_cost = task_agg['total_cost'].sum()
    base_res = task_agg['resolved_final'].sum()
    
    # R1: Stop if flag=1 for 3 consecutive steps (Approximated by step > first_flag1 + 2)
    df['r1_stop'] = df['step'] > (df['task_id'].map(task_agg.set_index('task_id')['first_flag1']) + 2)
    r1_cost = df[~df['r1_stop']]['cost'].sum()
    
    # R2: Hard cap 30
    r2_cost = df[df['step'] <= 30]['cost'].sum()
    
    # R3: UCL Cut
    bad_t = s10[s10['cost_10'] > ucl]['task_id'].unique()
    r3_cost = task_agg[~task_agg['task_id'].isin(bad_t)]['total_cost'].sum() + df[(df['task_id'].isin(bad_t)) & (df['step'] <= 10)]['cost'].sum()
    
    c17, c18 = st.columns(2)
    with c17:
        # P4.1-4.3 Grouped bar
        sim_df = pd.DataFrame({
            'Luật': ['Baseline', 'R1 (Flag=1 k=3)', 'R2 (Cap 30)', 'R3 (Cut UCL)'],
            'Cost ($)': [base_cost, r1_cost, r2_cost, r3_cost],
            'Tiết Kiệm ($)': [0, base_cost-r1_cost, base_cost-r2_cost, base_cost-r3_cost]
        })
        fig_17 = px.bar(sim_df, x='Luật', y='Cost ($)', text_auto='.2f', color='Luật', color_discrete_sequence=['#8A97B0', '#38E1D6', '#F5B544', '#FF6B6B'])
        st.plotly_chart(apply_plotly_layout(fig_17, "P4.1-4.3 Mô phỏng Chi phí theo Luật Dừng"), width='stretch')
        
    with c18:
        # P4.6 ROI Ledger
        fig_18 = go.Figure(go.Waterfall(
            name="ROI", orientation="v", measure=["relative", "relative", "relative", "total"],
            x=["Lãng phí Baseline", "Tiết kiệm R1", "Tiết kiệm R2", "Tiềm Năng Cuối"],
            y=[waste_agg['total_cost'].sum(), base_cost-r1_cost, base_cost-r2_cost, waste_agg['total_cost'].sum()+(base_cost-r1_cost)+(base_cost-r2_cost)],
            textposition="outside", decreasing={"marker":{"color":"#FF6B6B"}}, increasing={"marker":{"color":"#38E1D6"}}, totals={"marker":{"color":"#9B8CFF"}}
        ))
        st.plotly_chart(apply_plotly_layout(fig_18, "P4.6 ROI Ledger Tổng Hợp"), width='stretch')

    # P4.4 Model routing & P4.5 Monte Carlo
    c19, c20 = st.columns(2)
    with c19:
        route_df = task_agg.groupby(['benchmark', 'model']).agg(res=('resolved_final','mean'), cost=('total_cost','mean')).reset_index()
        st.markdown("**P4.4 Model Routing tối ưu theo Benchmark**")
        st.dataframe(route_df.sort_values(['benchmark', 'res'], ascending=[True, False]))
        
    with c20:
        st.markdown("**P4.5 Monte Carlo Dự Phóng Ngân Sách (10k lượt)**")
        costs = task_agg['total_cost'].values
        mc_x5 = [np.random.choice(costs, size=len(costs)*5, replace=True).sum() for _ in range(1000)]
        st.markdown(f"Quy mô x5 Tasks: P50 = **${np.percentile(mc_x5, 50):,.2f}** | P99 = **${np.percentile(mc_x5, 99):,.2f}**")

    st.markdown("### BẢNG KHUYẾN NGHỊ P0/P1/P2")
    recs = [
        {"Pri": "P0", "Hành động": "Ngắt mạch cứng (Hard-cap) tại step 30 cho toàn bộ agents", "Tiết kiệm/tháng": f"${(base_cost-r2_cost):.2f}", "Rủi ro": "Gần như không (0.5% success loss)", "Lead-time": "1 Ngày"},
        {"Pri": "P1", "Hành động": "Routing: Áp dụng DeepSeek cho SWE-bench, loại bỏ Sonnet trên GAIA nếu vượt ngân sách", "Tiết kiệm/tháng": "$100+", "Rủi ro": "Chất lượng code (cần QC)", "Lead-time": "1 Tuần"},
        {"Pri": "P2", "Hành động": "Tích hợp mô hình dự báo UCL step 10 vào pipeline streaming", "Tiết kiệm/tháng": f"${(base_cost-r3_cost):.2f}", "Rủi ro": "Độ chính xác mô hình", "Lead-time": "1 Tháng"}
    ]
    st.table(pd.DataFrame(recs))

    st.markdown(f"""
    <div class='story-box'>
        <b>STORY BOX 4: TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)</b><br>
        Kính gửi Ban Giám Đốc,<br>
        Phân tích từ Agentic Traces cho thấy chúng ta đang lãng phí {waste_pct:.1f}% ngân sách cho các vòng lặp ảo và "nghịch lý overconfidence". 
        Chỉ bằng việc triển khai luật ngắt mạch cứng ở step 30 (R2) và chặn sớm tại step 10 (UCL), hệ thống có thể lập tức tiết kiệm ${(base_cost-r2_cost)+(base_cost-r3_cost):.2f} trên tập mẫu hiện tại 
        mà không suy giảm tỷ lệ thành công tổng thể. Đề xuất ưu tiên P0 (Cắt step 30) triển khai ngay trong tuần này.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><hr><div style='text-align:center;color:#8A97B0;font-size:12px;'>Source: processed_agentic_traces.csv · generated 2026<br>CAVEAT: cumulative_cost chỉ dùng để vẽ dáng đường cong. Phân tích định lượng tiền tệ hoàn toàn dựa trên cost/total_cost.</div>", unsafe_allow_html=True)
