# ============================================================
# IMPORTS
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CẤU HÌNH TRANG — giữ lại nếu bạn thay cả file
# Nếu đã có st.set_page_config ở đầu file thì KHÔNG cần gọi lại
# ============================================================
# st.set_page_config(
#     page_title="AI Agent Diagnostic Intelligence",
#     layout="wide",
#     page_icon="🧠",
#     initial_sidebar_state="collapsed"
# )

# ============================================================
# DATA LOADING & ENGINEERING
# ============================================================
@st.cache_data
def load_and_engineer_data():
    import os
    csv_candidates = [
        os.path.join(os.path.dirname(__file__), "processed_agentic_traces.csv"),
        "processed_agentic_traces.csv",
        "code/processed_agentic_traces.csv",
        "/home/leducdat/projectDuan/code/processed_agentic_traces.csv"
    ]
    csv_file = next((f for f in csv_candidates if os.path.exists(f)), None)
    if csv_file is None:
        raise FileNotFoundError("Không tìm thấy file processed_agentic_traces.csv")
    df = pd.read_csv(csv_file)

    # Các cột bắt buộc cần có; nếu thiếu thì tự tạo để dashboard không crash
    base_cols = [
        "session_id",
        "model",
        "turn_number",
        "input_tokens",
        "output_length",
        "pre_gap",
        "has_error",
        "turn_cost",
        "is_system_prompt_present",
    ]

    for c in base_cols:
        if c not in df.columns:
            if c in ["session_id", "model"]:
                df[c] = "unknown"
            else:
                df[c] = 0

    num_cols = [
        "output_length",
        "pre_gap",
        "has_error",
        "turn_cost",
        "turn_number",
        "input_tokens",
        "is_system_prompt_present",
    ]

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Derived metrics
    df["latency"] = df["pre_gap"]
    df["success"] = 1 - df["has_error"]
    df["sunk_cost"] = df["has_error"] * df["turn_cost"]
    df["success_cost"] = df["success"] * df["turn_cost"]
    df["token_efficiency"] = np.where(
        df["input_tokens"] > 0,
        df["output_length"] / df["input_tokens"],
        0
    )

    df = df.sort_values(["session_id", "turn_number"])
    df["cum_cost"] = df.groupby("session_id")["turn_cost"].cumsum()

    # Phân loại ngữ cảnh & task theo docx
    df["context_size"] = pd.cut(
        df["input_tokens"],
        bins=[-1, 15000, 35000, np.inf],
        labels=["Thấp (<15K)", "Trung bình (15K-35K)", "Cao (>35K)"]
    )

    df["task_size"] = pd.cut(
        df["output_length"],
        bins=[-1, 150, 600, np.inf],
        labels=["Nhẹ (<150)", "Vừa (150-600)", "Nặng (>600)"]
    )

    # Session-level aggregation
    sess_agg = df.groupby(["session_id", "model"]).agg(
        total_cost=("turn_cost", "sum"),
        total_turns=("turn_number", "max"),
        avg_input_tokens=("input_tokens", "mean"),
        error_rate=("has_error", "mean"),
        sunk_cost=("sunk_cost", "sum"),
        token_efficiency=("token_efficiency", "mean"),
    ).reset_index()

    # Phiên thất bại hoàn toàn = 100% lượt lỗi
    sess_agg["failed_completely"] = (sess_agg["error_rate"] >= 0.999999).astype(int)

    return df, sess_agg


with st.spinner("Đang tải & xử lý dữ liệu..."):
    df, sess_agg = load_and_engineer_data()

# ============================================================
# COLOR PALETTE & PLOT THEME
# ============================================================
COLORS = {
    "claude-opus-4-6": "#10b981",
    "claude-sonnet-4-6": "#38bdf8",
    "deepseek-v3.1": "#f59e0b",
    "minimax-m2.5": "#f43f5e",
}

COLOR_LIST = ["#10b981", "#38bdf8", "#f59e0b", "#f43f5e"]

PLOT_CFG = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(248,250,252,0.6)",
    font=dict(family="Inter", color="#334155", size=12),
    title=dict(font=dict(size=15, color="#0f172a")),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(14,165,233,0.15)",
        borderwidth=1,
        font=dict(color="#334155", size=11)
    ),
    margin=dict(t=60, l=45, r=25, b=45),
    hovermode="closest"
)


def sf(fig, height=380):
    fig.update_layout(**PLOT_CFG, height=height)
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(14,165,233,0.10)",
        zeroline=False,
        tickfont=dict(color="#64748b", size=11)
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(14,165,233,0.10)",
        zeroline=False,
        tickfont=dict(color="#64748b", size=11)
    )
    return fig


def card_open(title=None):
    if title:
        st.markdown(
            f'<div class="bento-card"><div class="card-title">{title}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown('<div class="bento-card">', unsafe_allow_html=True)


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def safe_div(a, b):
    return a / b if b and b != 0 else 0


# ============================================================
# GLOBAL METRICS
# ============================================================
CHEAP_MODELS = ["minimax-m2.5", "deepseek-v3.1"]
SONNET_MODEL = "claude-sonnet-4-6"
OPUS_MODEL = "claude-opus-4-6"

total_turns = len(df)
total_sess = df["session_id"].nunique()
total_cost = df["turn_cost"].sum()
total_sunk = df["sunk_cost"].sum()
avg_error = df["has_error"].mean() * 100
sunk_pct = safe_div(total_sunk, total_cost) * 100

m_list = list(df["model"].unique())

# ------------------------------------------------------------
# Model-level summary
# ------------------------------------------------------------
model_turn = df.groupby("model").agg(
    turns=("session_id", "size"),
    total_cost=("turn_cost", "sum"),
    sunk_cost=("sunk_cost", "sum"),
    turn_error_rate=("has_error", "mean"),
    avg_input_tokens=("input_tokens", "mean"),
    avg_output_length=("output_length", "mean"),
    avg_latency=("latency", "mean"),
    avg_token_efficiency=("token_efficiency", "mean"),
).reset_index()

model_session = sess_agg.groupby("model").agg(
    sessions=("session_id", "nunique"),
    avg_cost_session=("total_cost", "mean"),
    avg_turns_session=("total_turns", "mean"),
    failed_session_pct=("failed_completely", "mean"),
    avg_session_input_tokens=("avg_input_tokens", "mean"),
).reset_index()

model_summary = model_turn.merge(model_session, on="model", how="outer").fillna(0)
model_summary["cost_share_pct"] = model_summary["total_cost"] / max(total_cost, 1e-9) * 100

# ------------------------------------------------------------
# Cheap model trap
# ------------------------------------------------------------
cheap_df = df[df["model"].isin(CHEAP_MODELS)]
cheap_sess = sess_agg[sess_agg["model"].isin(CHEAP_MODELS)]

cheap_cost = cheap_df["turn_cost"].sum() if len(cheap_df) else 0
cheap_sunk = cheap_df["sunk_cost"].sum() if len(cheap_df) else 0
cheap_turns = len(cheap_df)
cheap_sessions = cheap_df["session_id"].nunique() if len(cheap_df) else 0
cheap_error_rate = cheap_df["has_error"].mean() * 100 if len(cheap_df) else 0
cheap_avg_turns_session = cheap_sess["total_turns"].mean() if len(cheap_sess) else 0
cheap_avg_input_tokens = cheap_df["input_tokens"].mean() if len(cheap_df) else 0
cheap_cost_share = safe_div(cheap_cost, total_cost) * 100

# ------------------------------------------------------------
# System Prompt impact on Claude Sonnet
# ------------------------------------------------------------
df_sonnet = df[df["model"] == SONNET_MODEL].copy()

if len(df_sonnet) > 0:
    sonnet_session_sp = df_sonnet.groupby("session_id").agg(
        sp_present=("is_system_prompt_present", "max"),
        total_cost=("turn_cost", "sum"),
        total_turns=("turn_number", "max"),
        avg_input_tokens=("input_tokens", "mean"),
        error_rate=("has_error", "mean"),
    ).reset_index()

    sonnet_session_sp["failed_completely"] = (
        sonnet_session_sp["error_rate"] >= 0.999999
    ).astype(int)

    sp_summary = sonnet_session_sp.groupby("sp_present").agg(
        sessions=("session_id", "nunique"),
        avg_cost_session=("total_cost", "mean"),
        avg_turns_session=("total_turns", "mean"),
        avg_input_tokens=("avg_input_tokens", "mean"),
        failed_session_pct=("failed_completely", "mean"),
    ).reindex([0, 1]).fillna(0)

    sp_turn_error = (
        df_sonnet.groupby("is_system_prompt_present")["has_error"]
        .mean()
        .reindex([0, 1])
        .fillna(0)
    )

else:
    sp_summary = pd.DataFrame(
        {
            "sessions": [0, 0],
            "avg_cost_session": [0, 0],
            "avg_turns_session": [0, 0],
            "avg_input_tokens": [0, 0],
            "failed_session_pct": [0, 0],
        },
        index=[0, 1]
    )
    sp_turn_error = pd.Series([0, 0], index=[0, 1])

sp_no = sp_summary.loc[0]
sp_yes = sp_summary.loc[1]

sp_token_increase_pct = (
    safe_div(
        sp_yes["avg_input_tokens"] - sp_no["avg_input_tokens"],
        sp_no["avg_input_tokens"]
    ) * 100
)

sp_cost_ratio = safe_div(sp_yes["avg_cost_session"], sp_no["avg_cost_session"])
sp_error_no = sp_turn_error.loc[0] * 100
sp_error_yes = sp_turn_error.loc[1] * 100
sp_failed_no = sp_no["failed_session_pct"] * 100
sp_failed_yes = sp_yes["failed_session_pct"] * 100

# ------------------------------------------------------------
# Opus highlight
# ------------------------------------------------------------
opus_row = model_summary[model_summary["model"] == OPUS_MODEL]

if len(opus_row) > 0:
    opus_error_rate = float(opus_row["turn_error_rate"].iloc[0] * 100)
    opus_failed_session_pct = float(opus_row["failed_session_pct"].iloc[0] * 100)
    opus_cost_session = float(opus_row["avg_cost_session"].iloc[0])
    opus_turns_session = float(opus_row["avg_turns_session"].iloc[0])
else:
    opus_error_rate = 0
    opus_failed_session_pct = 0
    opus_cost_session = 0
    opus_turns_session = 0

# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div class="bento-grid">
  <div class="bento-header">
    <div class="header-badge">
      <span>🧠</span> Executive Report • AI Agent Telemetry
    </div>
    <h1 class="header-title">
      Xây dựng báo cáo phân tích chi phí và hiệu năng hoạt động của AI Agent
    </h1>
    <p class="header-subtitle">
      Phân tích {total_turns:,} lượt tương tác agentic qua {total_sess:,} phiên thử nghiệm
      trên benchmark lập trình SWE-bench. Báo cáo tập trung vào hai xung đột chính:
      <strong style="color:#7dd3fc;">chi phí bị lãng phí ở nhóm model giá rẻ</strong>
      và
      <strong style="color:#a78bfa;">tác động tiêu cực của System Prompt cồng kềnh lên Claude Sonnet</strong>.
    </p>
    <div class="header-meta">
      <div class="meta-item"><span class="meta-dot"></span> Processed Telemetry</div>
      <div class="meta-item">📊 {total_turns:,} turns • {total_sess:,} sessions</div>
      <div class="meta-item">🤖 {len(m_list)} models</div>
      <div class="meta-item">💰 Tổng ngân sách: ${total_cost:,.2f}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# KPI ROW
# ============================================================
st.markdown(f"""
<div class="bento-grid">
  <div class="bento-card col-span-3 kpi-card">
    <div>
      <div class="kpi-icon kpi-icon-blue">📡</div>
      <div class="kpi-value">{total_turns:,}</div>
      <div class="kpi-label">Tổng lượt (Turns)</div>
    </div>
    <div class="kpi-trend trend-neutral">↔ {total_sess:,} phiên</div>
  </div>

  <div class="bento-card col-span-3 kpi-card">
    <div>
      <div class="kpi-icon kpi-icon-green">💰</div>
      <div class="kpi-value">${total_cost:,.2f}</div>
      <div class="kpi-label">Tổng ngân sách (USD)</div>
    </div>
    <div class="kpi-trend trend-neutral">↔ 4 mô hình chính</div>
  </div>

  <div class="bento-card col-span-3 kpi-card">
    <div>
      <div class="kpi-icon kpi-icon-rose">🔥</div>
      <div class="kpi-value">${cheap_cost:,.2f}</div>
      <div class="kpi-label">Chi phí nhóm model rẻ</div>
    </div>
    <div class="kpi-trend trend-down">▼ {cheap_cost_share:.1f}% tổng ngân sách</div>
  </div>

  <div class="bento-card col-span-3 kpi-card">
    <div>
      <div class="kpi-icon kpi-icon-purple">🧪</div>
      <div class="kpi-value">{sp_cost_ratio:.2f}x</div>
      <div class="kpi-label">Sonnet: chi phí khi có System Prompt</div>
    </div>
    <div class="kpi-trend trend-down">▼ So với không có System Prompt</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# TABS — ĐÚNG THEO DOCX
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Toàn cảnh dữ liệu",
    "💸 Cạm bẫy giá rẻ",
    "🧪 Nghịch lý System Prompt",
    "✅ Điểm sáng Opus & Khuyến nghị",
])

# ============================================================
# TAB 1 — TOÀN CẢNH DỮ LIỆU
# ============================================================
with tab1:
    card_open("📌 Bối cảnh: Lứa AI Agent đầu tiên đi vào thực tế")
    st.markdown(f"""
    <div class="insight-content">
      Tập dữ liệu ghi nhận <strong>{total_sess:,} phiên làm việc</strong>
      với tổng ngân sách thử nghiệm <strong>${total_cost:,.2f}</strong>,
      phân bổ cho <strong>{len(m_list)} mô hình chính</strong>.
      Mục tiêu ban đầu rất đơn giản: tối ưu chi phí trên mỗi lượt chạy
      và cung cấp prompt chi tiết để AI làm việc chính xác.
      <div class="insight-highlight">
        Tuy nhiên, dữ liệu cho thấy <strong>chi phí rẻ chưa chắc đã tiết kiệm</strong>,
        và <strong>hướng dẫn chi tiết chưa chắc đã hiệu quả</strong>.
      </div>
    </div>
    """, unsafe_allow_html=True)
    card_close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # Bảng tổng hợp theo model
    # ------------------------------------------------------------
    card_open("🧾 Bảng tổng hợp hiệu năng & chi phí theo Model")

    display_summary = model_summary.copy()
    display_summary["turn_error_rate_pct"] = display_summary["turn_error_rate"] * 100
    display_summary["failed_session_pct"] = display_summary["failed_session_pct"] * 100

    display_summary = display_summary[[
        "model",
        "sessions",
        "turns",
        "total_cost",
        "avg_cost_session",
        "avg_turns_session",
        "turn_error_rate_pct",
        "failed_session_pct",
        "avg_input_tokens",
    ]].copy()

    display_summary.columns = [
        "Model",
        "Số phiên",
        "Tổng lượt",
        "Tổng chi phí ($)",
        "Chi phí/phiên ($)",
        "Turns/phiên",
        "Tỷ lệ lỗi/lượt (%)",
        "Phiên lỗi 100% (%)",
        "Input tokens/lượt",
    ]

    styled_table = (
        display_summary.style
        .format({
            "Số phiên": "{:,.0f}",
            "Tổng lượt": "{:,.0f}",
            "Tổng chi phí ($)": "${:,.2f}",
            "Chi phí/phiên ($)": "${:,.4f}",
            "Turns/phiên": "{:.1f}",
            "Tỷ lệ lỗi/lượt (%)": "{:.1f}",
            "Phiên lỗi 100% (%)": "{:.1f}",
            "Input tokens/lượt": "{:,.0f}",
        })
        .hide(axis="index")
    )

    st.dataframe(styled_table, use_container_width=True)
    card_close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # Chi phí theo model + phân bổ ngân sách
    # ------------------------------------------------------------
    c1, c2 = st.columns(2)

    with c1:
        card_open("💰 Tổng chi phí theo Model")
        fig_cost = px.bar(
            model_summary.sort_values("total_cost", ascending=True),
            x="total_cost",
            y="model",
            orientation="h",
            color="model",
            color_discrete_map=COLORS,
            text=model_summary.sort_values("total_cost", ascending=True)["total_cost"]
            .apply(lambda x: f"${x:,.2f}"),
            title="Tổng chi phí thử nghiệm theo Model"
        )
        fig_cost.update_layout(yaxis_title=None, xaxis_title="Tổng chi phí (USD)")
        fig_cost.update_traces(textposition="outside")
        st.plotly_chart(sf(fig_cost, 380), use_container_width=True)
        card_close()

    with c2:
        card_open("🥧 Phân bổ ngân sách theo Model")
        fig_pie = px.pie(
            model_summary,
            names="model",
            values="total_cost",
            color="model",
            color_discrete_map=COLORS,
            hole=0.55,
            title="Tỷ trọng chi phí theo Model"
        )
        fig_pie.update_traces(textinfo="label+percent")
        st.plotly_chart(sf(fig_pie, 380), use_container_width=True)
        card_close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # Chi phí/phiên vs tỷ lệ lỗi
    # ------------------------------------------------------------
    card_open("⚖️ Chi phí mỗi phiên vs Tỷ lệ lỗi theo Model")

    fig_perf = make_subplots(specs=[[{"secondary_y": True}]])

    fig_perf.add_trace(
        go.Bar(
            name="Chi phí/phiên ($)",
            x=model_summary["model"],
            y=model_summary["avg_cost_session"],
            marker_color=[COLORS.get(m, "#94a3b8") for m in model_summary["model"]],
            opacity=0.85,
            text=model_summary["avg_cost_session"].apply(lambda x: f"${x:.4f}"),
            textposition="outside",
            textfont=dict(color="#0f172a", size=11),
        ),
        secondary_y=False
    )

    fig_perf.add_trace(
        go.Scatter(
            name="Tỷ lệ lỗi/lượt (%)",
            x=model_summary["model"],
            y=model_summary["turn_error_rate"] * 100,
            mode="lines+markers+text",
            line=dict(color="#ef4444", width=3),
            marker=dict(size=12, color="#ef4444", symbol="diamond"),
            text=(model_summary["turn_error_rate"] * 100).apply(lambda x: f"{x:.1f}%"),
            textposition="top center",
            textfont=dict(color="#ef4444", size=11),
        ),
        secondary_y=True
    )

    fig_perf.update_layout(
        title="Chi phí trung bình mỗi phiên so với tỷ lệ lỗi",
        barmode="group"
    )
    fig_perf.update_yaxes(title_text="Chi phí/phiên ($)", secondary_y=False)
    fig_perf.update_yaxes(title_text="Tỷ lệ lỗi (%)", secondary_y=True)

    st.plotly_chart(sf(fig_perf, 400), use_container_width=True)
    card_close()

    st.markdown("<br>", unsafe_allow_html=True)

    card_open("🔎 Insight cấp 1")
    st.markdown(f"""
    <div class="insight-content">
      Bức tranh toàn cảnh cho thấy sự phân hóa rất lớn giữa các model:
      <div class="insight-highlight-amber">
        Nhóm model giá rẻ <strong>minimax-m2.5</strong> và <strong>deepseek-v3.1</strong>
        đang chiếm <strong>${cheap_cost:,.2f}</strong> tổng chi phí,
        tương đương <strong>{cheap_cost_share:.1f}%</strong> ngân sách.
      </div>
      <div class="insight-highlight">
        Trong khi đó, <strong>claude-sonnet-4-6</strong> thường là nơi tiêu thụ ngân sách lớn nhất
        và cần được phân tích sâu hơn theo chiều <strong>System Prompt</strong>.
      </div>
      <div class="insight-highlight-green">
        <strong>claude-opus-4-6</strong> nổi lên như một model có độ ổn định tốt hơn
        so với phần còn lại trong tập dữ liệu này.
      </div>
    </div>
    """, unsafe_allow_html=True)
    card_close()

# ============================================================
# TAB 2 — CẠM BẪY GIÁ RẺ
# ============================================================
with tab2:
    card_open("💸 Xung đột #1: Cạm bẫy giá rẻ — Khi chi phí rẻ biến thành lãng phí")
    st.markdown(f"""
    <div class="insight-content">
      Thoạt nhìn, <strong>minimax-m2.5</strong> và <strong>deepseek-v3.1</strong>
      có vẻ là lựa chọn lý tưởng cho tác vụ agentic vì giá mỗi lượt rẻ.
      Tuy nhiên, dữ liệu cho thấy nhóm model giá rẻ có thể tạo ra
      <strong>vòng lặp lỗi kéo dài</strong>, khiến chi phí bị đốt nhiều hơn kỳ vọng.
      <div class="insight-highlight-amber">
        Trong tập dữ liệu này, nhóm model rẻ đã tiêu tốn
        <strong>${cheap_cost:,.2f}</strong> với
        <strong>{cheap_turns:,} lượt</strong> và
        <strong>{cheap_sessions:,} phiên</strong>.
      </div>
    </div>
    """, unsafe_allow_html=True)
    card_close()

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI hàng
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric(
            "Chi phí nhóm model rẻ",
            f"${cheap_cost:,.2f}",
            f"{cheap_cost_share:.1f}% tổng ngân sách",
            delta_color="inverse"
        )

    with k2:
        st.metric(
            "Tỷ lệ lỗi/lượt",
            f"{cheap_error_rate:.1f}%",
            "Rất cao" if cheap_error_rate > 50 else "Cần theo dõi",
            delta_color="inverse"
        )

    with k3:
        st.metric(
            "Turns trung bình/phiên",
            f"{cheap_avg_turns_session:.1f}",
            "Vòng lặp dài" if cheap_avg_turns_session > 20 else "Ổn định hơn"
        )

    with k4:
        st.metric(
            "Input tokens trung bình/lượt",
            f"{cheap_avg_input_tokens:,.0f}",
            "Ngữ cảnh lớn" if cheap_avg_input_tokens > 15000 else "Ngữ cảnh vừa"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # So sánh tỷ lệ lỗi giữa các model
    # ------------------------------------------------------------
    c1, c2 = st.columns(2)

    with c1:
        card_open("❌ Tỷ lệ lỗi theo Model")
        fig_err = px.bar(
            model_summary.sort_values("turn_error_rate", ascending=True),
            x="turn_error_rate",
            y="model",
            orientation="h",
            color="model",
            color_discrete_map=COLORS,
            text=(model_summary.sort_values("turn_error_rate", ascending=True)["turn_error_rate"] * 100)
            .apply(lambda x: f"{x:.1f}%"),
            title="Tỷ lệ lỗi trên tổng lượt theo Model"
        )
        fig_err.update_layout(xaxis_title="Tỷ lệ lỗi (%)", yaxis_title=None)
        fig_err.update_traces(textposition="outside")
        st.plotly_chart(sf(fig_err, 380), use_container_width=True)
        card_close()

    with c2:
        card_open("🔁 Số lượt trung bình mỗi phiên")
        fig_turns = px.bar(
            model_summary.sort_values("avg_turns_session", ascending=True),
            x="avg_turns_session",
            y="model",
            orientation="h",
            color="model",
            color_discrete_map=COLORS,
            text=model_summary.sort_values("avg_turns_session", ascending=True)["avg_turns_session"]
            .apply(lambda x: f"{x:.1f}"),
            title="Số lượt trung bình trên mỗi phiên"
        )
        fig_turns.update_layout(xaxis_title="Turns/phiên", yaxis_title=None)
        fig_turns.update_traces(textposition="outside")
        st.plotly_chart(sf(fig_turns, 380), use_container_width=True)
        card_close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # Chi phí chìm + tốc độ đốt tiền
    # ------------------------------------------------------------
    c3, c4 = st.columns(2)

    with c3:
        card_open("🕳️ Chi phí chìm (Sunk Cost) theo Model")
        fig_sunk = px.bar(
            model_summary.sort_values("sunk_cost", ascending=True),
            x="sunk_cost",
            y="model",
            orientation="h",
            color="model",
            color_discrete_map=COLORS,
            text=model_summary.sort_values("sunk_cost", ascending=True)["sunk_cost"]
            .apply(lambda x: f"${x:,.2f}"),
            title="Chi phí phát sinh ở các lượt lỗi"
        )
        fig_sunk.update_layout(xaxis_title="Chi phí chìm (USD)", yaxis_title=None)
        fig_sunk.update_traces(textposition="outside")
        st.plotly_chart(sf(fig_sunk, 380), use_container_width=True)
        card_close()

    with c4:
        card_open("🔥 Tốc độ đốt tiền theo lượt")

        burn_df = df[df["turn_number"] <= 40].groupby(
            ["model", "turn_number"]
        )["cum_cost"].mean().reset_index()

        fig_burn = px.line(
            burn_df,
            x="turn_number",
            y="cum_cost",
            color="model",
            color_discrete_map=COLORS,
            markers=True,
            title="Chi phí tích lũy trung bình theo lượt"
        )
        fig_burn.update_layout(
            xaxis_title="Turn Number",
            yaxis_title="Chi phí tích lũy trung bình ($)"
        )
        st.plotly_chart(sf(fig_burn, 380), use_container_width=True)
        card_close()

    st.markdown("<br>", unsafe_allow_html=True)

    card_open("🧠 Insight cấp 2")
    st.markdown(f"""
    <div class="insight-content">
      Vấn đề của model giá rẻ không nằm ở đơn giá mỗi token,
      mà nằm ở <strong>hành vi vòng lặp</strong>:
      <div class="insight-highlight-amber">
        Khi model liên tục lỗi nhưng không tự khắc phục được,
        phiên làm việc kéo dài hơn, input tokens phình to hơn,
        và tổng chi phí thực tế tăng lên dù giá mỗi lượt rất thấp.
      </div>
      <div class="insight-highlight">
        Đây chính là <strong>cạm bẫy chi phí</strong>:
        chọn model chỉ vì giá token rẻ có thể dẫn tới tổng chi phí cao hơn
        và hiệu quả thấp hơn.
      </div>
    </div>
    """, unsafe_allow_html=True)
    card_close()

# ============================================================
# TAB 3 — NGHỊCH LÝ SYSTEM PROMPT
# ============================================================
with tab3:
    card_open("🧪 Xung đột #2: Nghịch lý System Prompt ở Claude Sonnet")
    st.markdown(f"""
    <div class="insight-content">
      Phần lớn ngân sách thử nghiệm thường tập trung vào
      <strong>claude-sonnet-4-6</strong>. Khi phân tích sâu hơn về tác động
      của việc đưa System Prompt vào mô hình này, dữ liệu cho thấy:
      <div class="insight-highlight-purple">
        Bổ sung System Prompt không nhất thiết làm AI thông minh hơn,
        mà có thể làm nó bị <strong>ngợp context</strong> và tăng tỷ lệ thất bại.
      </div>
    </div>
    """, unsafe_allow_html=True)
    card_close()

    st.markdown("<br>", unsafe_allow_html=True)

    if len(df_sonnet) == 0:
        st.info("Không tìm thấy dữ liệu của claude-sonnet-4-6 trong file CSV.")
    else:
        # KPI System Prompt
        s1, s2, s3, s4 = st.columns(4)

        with s1:
            st.metric(
                "Input tokens trung bình",
                f"{sp_yes['avg_input_tokens']:,.0f}",
                f"+{sp_token_increase_pct:.1f}% so với không SP",
                delta_color="inverse"
            )

        with s2:
            st.metric(
                "Tỷ lệ lỗi/lượt",
                f"{sp_error_yes:.1f}%",
                f"{sp_error_yes - sp_error_no:+.1f} điểm %",
                delta_color="inverse"
            )

        with s3:
            st.metric(
                "Chi phí/phiên",
                f"${sp_yes['avg_cost_session']:,.4f}",
                f"{sp_cost_ratio:.2f}x so với không SP",
                delta_color="inverse"
            )

        with s4:
            st.metric(
                "Phiên lỗi 100%",
                f"{sp_failed_yes:.1f}%",
                f"{sp_failed_yes - sp_failed_no:+.1f} điểm %",
                delta_color="inverse"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ------------------------------------------------------------
        # Before/After System Prompt
        # ------------------------------------------------------------
        card_open("📈 So sánh Claude Sonnet: Không System Prompt vs Có System Prompt")

        fig_sp = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Input Tokens trung bình/phiên",
                "Tỷ lệ lỗi/lượt (%)",
                "Chi phí trung bình/phiên ($)",
                "Số lượt trung bình/phiên"
            )
        )

        metrics = [
            ("avg_input_tokens", sp_no["avg_input_tokens"], sp_yes["avg_input_tokens"], "{:,.0f}"),
            ("turn_error_rate", sp_error_no, sp_error_yes, "{:.1f}"),
            ("avg_cost_session", sp_no["avg_cost_session"], sp_yes["avg_cost_session"], "${:.4f}"),
            ("avg_turns_session", sp_no["avg_turns_session"], sp_yes["avg_turns_session"], "{:.1f}"),
        ]

        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for metric_name, val_no, val_yes, fmt in metrics:
            row, col = positions[metrics.index((metric_name, val_no, val_yes, fmt))]

            fig_sp.add_trace(
                go.Bar(
                    x=["Không SP", "Có SP"],
                    y=[val_no, val_yes],
                    marker_color=["#94a3b8", "#0ea5e9"],
                    text=[fmt.format(val_no), fmt.format(val_yes)],
                    textposition="outside",
                    textfont=dict(color="#0f172a", size=11),
                    showlegend=False
                ),
                row=row,
                col=col
            )

        fig_sp.update_layout(
            title="Tác động của System Prompt lên Claude Sonnet",
            height=560
        )

        st.plotly_chart(sf(fig_sp, 560), use_container_width=True)
        card_close()

        st.markdown("<br>", unsafe_allow_html=True)

        c_sp1, c_sp2 = st.columns(2)

        with c_sp1:
            card_open("📦 Phân phối chi phí phiên theo System Prompt")

            box_df = sonnet_session_sp.copy()
            box_df["System Prompt"] = box_df["sp_present"].map({
                0: "Không System Prompt",
                1: "Có System Prompt"
            })

            fig_sp_box = px.box(
                box_df,
                x="System Prompt",
                y="total_cost",
                color="System Prompt",
                color_discrete_map={
                    "Không System Prompt": "#94a3b8",
                    "Có System Prompt": "#0ea5e9"
                },
                title="Chi phí phiên của Claude Sonnet"
            )
            fig_sp_box.update_layout(
                xaxis_title=None,
                yaxis_title="Chi phí phiên ($)"
            )
            st.plotly_chart(sf(fig_sp_box, 400), use_container_width=True)
            card_close()

        with c_sp2:
            card_open("📉 Tỷ lệ lỗi theo lượt: Có SP vs Không SP")

            turn_sp = df_sonnet[df_sonnet["turn_number"] <= 30].groupby(
                ["turn_number", "is_system_prompt_present"]
            )["has_error"].mean().reset_index()

            turn_sp["System Prompt"] = turn_sp["is_system_prompt_present"].map({
                0: "Không System Prompt",
                1: "Có System Prompt"
            })

            fig_sp_line = px.line(
                turn_sp,
                x="turn_number",
                y="has_error",
                color="System Prompt",
                color_discrete_map={
                    "Không System Prompt": "#94a3b8",
                    "Có System Prompt": "#0ea5e9"
                },
                markers=True,
                title="Tỷ lệ lỗi theo lượt của Claude Sonnet"
            )
            fig_sp_line.update_layout(
                xaxis_title="Turn Number",
                yaxis_title="Tỷ lệ lỗi"
            )
            fig_sp_line.update_yaxes(tickformat=".0%")
            st.plotly_chart(sf(fig_sp_line, 400), use_container_width=True)
            card_close()

        st.markdown("<br>", unsafe_allow_html=True)

        card_open("🧠 Insight cấp 3")
        st.markdown(f"""
        <div class="insight-content">
          Với Claude Sonnet, System Prompt dài có thể tạo ra chuỗi tác động:
          <div class="insight-highlight-purple">
            [System Prompt bị phình to] → [Model quá tải context] → [Tỷ lệ lỗi tăng]
            → [Vòng lặp sửa lỗi kéo dài] → [Chi phí tăng mạnh].
          </div>
          Trong dữ liệu này:
          <ul>
            <li>Input tokens trung bình tăng <strong>{sp_token_increase_pct:.1f}%</strong>.</li>
            <li>Tỷ lệ lỗi/lượt tăng từ <strong>{sp_error_no:.1f}%</strong> lên <strong>{sp_error_yes:.1f}%</strong>.</li>
            <li>Chi phí/phiên tăng khoảng <strong>{sp_cost_ratio:.2f} lần</strong>.</li>
            <li>Tỷ lệ phiên lỗi hoàn toàn tăng từ <strong>{sp_failed_no:.1f}%</strong> lên <strong>{sp_failed_yes:.1f}%</strong>.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)
        card_close()

# ============================================================
# TAB 4 — ĐIỂM SÁNG OPUS & KHUYẾN NGHỊ
# ============================================================
with tab4:
    card_open("✅ Điểm sáng: Hiệu quả của Claude Opus")
    st.markdown(f"""
    <div class="insight-content">
      Trong khi Sonnet gặp khó khăn với prompt dài,
      <strong>Claude Opus</strong> thể hiện sự ổn định tốt hơn trong tập dữ liệu này.
      <div class="insight-highlight-green">
        Opus cho thấy tỷ lệ lỗi thấp hơn, ít rơi vào vòng lặp thất bại hơn
        và có thể hoàn thành nhiệm vụ với chi phí thực tế hợp lý hơn
        nếu xét trên tổng thể.
      </div>
    </div>
    """, unsafe_allow_html=True)
    card_close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # So sánh Opus vs Sonnet
    # ------------------------------------------------------------
    focus_models = [OPUS_MODEL, SONNET_MODEL]
    focus_summary = model_summary[model_summary["model"].isin(focus_models)].copy()

    if len(focus_summary) == 0:
        st.info("Không có dữ liệu để so sánh Opus và Sonnet.")
    else:
        o1, o2, o3, o4 = st.columns(4)

        with o1:
            st.metric(
                "Opus: Tỷ lệ lỗi/lượt",
                f"{opus_error_rate:.1f}%",
                "Thấp" if opus_error_rate < 25 else "Trung bình"
            )

        with o2:
            st.metric(
                "Opus: Phiên lỗi 100%",
                f"{opus_failed_session_pct:.1f}%",
                "Rất tốt" if opus_failed_session_pct < 5 else "Cần theo dõi"
            )

        with o3:
            st.metric(
                "Opus: Chi phí/phiên",
                f"${opus_cost_session:,.4f}"
            )

        with o4:
            st.metric(
                "Opus: Turns/phiên",
                f"{opus_turns_session:.1f}"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        card_open("🥊 So sánh Claude Opus vs Claude Sonnet")

        fig_focus = make_subplots(
            rows=1,
            cols=4,
            subplot_titles=(
                "Tỷ lệ lỗi/lượt (%)",
                "Phiên lỗi 100% (%)",
                "Chi phí/phiên ($)",
                "Turns/phiên"
            )
        )

        focus_summary["turn_error_rate_pct"] = focus_summary["turn_error_rate"] * 100
        focus_summary["failed_session_pct"] = focus_summary["failed_session_pct"] * 100

        focus_metrics = [
            ("turn_error_rate_pct", 1),
            ("failed_session_pct", 2),
            ("avg_cost_session", 3),
            ("avg_turns_session", 4),
        ]

        for metric, col_idx in focus_metrics:
            fig_focus.add_trace(
                go.Bar(
                    x=focus_summary["model"],
                    y=focus_summary[metric],
                    marker_color=[COLORS.get(m, "#94a3b8") for m in focus_summary["model"]],
                    text=focus_summary[metric].apply(lambda x: f"{x:,.2f}"),
                    textposition="outside",
                    textfont=dict(color="#0f172a", size=11),
                    showlegend=False
                ),
                row=1,
                col=col_idx
            )

        fig_focus.update_layout(
            title="So sánh các chỉ số quan trọng giữa Opus và Sonnet",
            height=420
        )

        st.plotly_chart(sf(fig_focus, 420), use_container_width=True)
        card_close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # Circuit Breaker
    # ------------------------------------------------------------
    card_open("🛑 Khuyến nghị #1: Thiết lập ngắt mạch tự động (Circuit Breakers)")

    cutoffs = list(range(1, 31))
    savings_pct = []

    for c in cutoffs:
        saved = df[df["turn_number"] > c]["sunk_cost"].sum()
        savings_pct.append(safe_div(saved, total_sunk) * 100)

    fig_cb = go.Figure()

    fig_cb.add_trace(
        go.Scatter(
            x=cutoffs,
            y=savings_pct,
            mode="lines+markers",
            line=dict(color="#10b981", width=3),
            marker=dict(size=8, color="#10b981"),
            name="% chi phí chìm có thể tránh"
        )
    )

    fig_cb.add_vline(
        x=10,
        line_dash="dash",
        line_color="#f43f5e",
        annotation_text="Cutoff 10 turns",
        annotation_font_color="#f43f5e"
    )

    fig_cb.add_vline(
        x=15,
        line_dash="dot",
        line_color="#f59e0b",
        annotation_text="Cutoff 15 turns",
        annotation_font_color="#b45309"
    )

    fig_cb.update_layout(
        title="Nếu ngắt phiên sớm hơn, bao nhiêu % chi phí chìm có thể được tránh?",
        xaxis_title="Ngắt sau Turn",
        yaxis_title="% chi phí chìm tránh được"
    )

    st.plotly_chart(sf(fig_cb, 400), use_container_width=True)

    st.markdown("""
    <div class="insight-highlight-green">
      <strong>Khuyến nghị:</strong> Đặt giới hạn cứng tối đa 10–15 turns cho các mô hình agentic.
      Nếu mô hình gặp lỗi liên tiếp 3–5 turns, nên ngắt phiên ngay để tránh tình trạng đốt tiền vô ích.
    </div>
    """, unsafe_allow_html=True)

    card_close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # Prompt pruning & TCO
    # ------------------------------------------------------------
    c_act1, c_act2 = st.columns(2)

    with c_act1:
        card_open("🪓 Khuyến nghị #2: Tinh gọn System Prompt")

        st.markdown(f"""
        <div class="insight-content">
          Tránh nhồi nhét quá nhiều hướng dẫn vào System Prompt
          trong các vòng lặp đa bước. Hãy giữ prompt ngắn gọn,
          tập trung vào trạng thái hiện tại để giảm rủi ro phình context.
          <div class="insight-highlight-purple">
            Với Claude Sonnet, dữ liệu cho thấy khi có System Prompt,
            chi phí/phiên tăng khoảng <strong>{sp_cost_ratio:.2f} lần</strong>
            và tỷ lệ lỗi tăng từ <strong>{sp_error_no:.1f}%</strong>
            lên <strong>{sp_error_yes:.1f}%</strong>.
          </div>
        </div>
        """, unsafe_allow_html=True)

        card_close()

    with c_act2:
        card_open("💰 Khuyến nghị #3: Đánh giá lại Total Cost of Ownership")

        fig_tco = px.scatter(
            model_summary,
            x="avg_cost_session",
            y="failed_session_pct",
            size="sessions",
            color="model",
            color_discrete_map=COLORS,
            hover_name="model",
            labels={
                "avg_cost_session": "Chi phí trung bình/phiên ($)",
                "failed_session_pct": "Tỷ lệ phiên lỗi 100% (%)",
                "model": "Model"
            },
            title="Đánh đổi giữa chi phí/phiên và tỷ lệ phiên thất bại hoàn toàn"
        )

        fig_tco.update_layout(
            xaxis_title="Chi phí trung bình/phiên ($)",
            yaxis_title="Tỷ lệ phiên lỗi 100% (%)"
        )

        st.plotly_chart(sf(fig_tco, 360), use_container_width=True)

        st.markdown("""
        <div class="insight-highlight">
          Đừng chọn model chỉ vì giá token rẻ. Một model đắt hơn nhưng hoàn thành
          công việc trong ít turns hơn và ít lỗi hơn có thể tiết kiệm chi phí tổng thể hơn.
        </div>
        """, unsafe_allow_html=True)

        card_close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------------
    # Action table
    # ------------------------------------------------------------
    card_open("🎯 Action Items theo Docx")

    st.markdown("""
    <div class="insight-content">
      <table style="width:100%; border-collapse: separate; border-spacing: 0 6px; font-size: 0.85rem;">
        <thead>
          <tr>
            <th style="text-align:left; padding: 8px 12px; color:#64748b; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.5px;">Priority</th>
            <th style="text-align:left; padding: 8px 12px; color:#64748b; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.5px;">Action</th>
            <th style="text-align:left; padding: 8px 12px; color:#64748b; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.5px;">Insight từ dữ liệu</th>
            <th style="text-align:left; padding: 8px 12px; color:#64748b; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.5px;">Kỳ vọng</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style="padding:10px 12px; background:rgba(244,63,94,0.08); border-radius:8px 0 0 8px;">
              <span class="status-badge status-critical">P0 Critical</span>
            </td>
            <td style="padding:10px 12px; background:rgba(244,63,94,0.08); color:#0f172a;">
              Thiết lập Circuit Breakers: giới hạn 10–15 turns; ngắt nếu lỗi liên tiếp 3–5 turns
            </td>
            <td style="padding:10px 12px; background:rgba(244,63,94,0.08); color:#0f172a;">
              Chặn vòng lặp vô tận ở nhóm model rẻ
            </td>
            <td style="padding:10px 12px; background:rgba(244,63,94,0.08); border-radius:0 8px 8px 0; color:#0f172a;">
              Giảm chi phí chìm
            </td>
          </tr>
          <tr>
            <td style="padding:10px 12px; background:rgba(245,158,11,0.08); border-radius:8px 0 0 8px;">
              <span class="status-badge status-warning">P1 High</span>
            </td>
            <td style="padding:10px 12px; background:rgba(245,158,11,0.08); color:#0f172a;">
              Tinh gọn System Prompt với Claude Sonnet
            </td>
            <td style="padding:10px 12px; background:rgba(245,158,11,0.08); color:#0f172a;">
              System Prompt dài làm tăng tokens, lỗi và chi phí
            </td>
            <td style="padding:10px 12px; background:rgba(245,158,11,0.08); border-radius:0 8px 8px 0; color:#0f172a;">
              Giảm phình context, giảm chi phí/phiên
            </td>
          </tr>
          <tr>
            <td style="padding:10px 12px; background:rgba(56,189,248,0.08); border-radius:8px 0 0 8px;">
              <span class="status-badge status-optimal">P2 Normal</span>
            </td>
            <td style="padding:10px 12px; background:rgba(56,189,248,0.08); color:#0f172a;">
              Đánh giá model theo Total Cost of Ownership thay vì giá token
            </td>
            <td style="padding:10px 12px; background:rgba(56,189,248,0.08); color:#0f172a;">
              Model rẻ có thể gây lãng phí lớn hơn do vòng lặp lỗi
            </td>
            <td style="padding:10px 12px; background:rgba(56,189,248,0.08); border-radius:0 8px 8px 0; color:#0f172a;">
              Chọn model tối ưu chi phí tổng thể
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    card_close()

# ============================================================
# FOOTER
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)

st.markdown(f"""
<div class="bento-grid">
  <div class="bento-card col-span-12" style="text-align:center; padding: 20px;">
    <span style="font-size: 0.75rem; color: #475569;">
      🧠 AI Agent Diagnostic Intelligence •
      {total_turns:,} Turns • {total_sess:,} Sessions • {len(m_list)} Models •
      Data Source: processed_agentic_traces.csv • Confidential
    </span>
  </div>
</div>
""", unsafe_allow_html=True)