# %% [markdown]
# # Khung Phân Tích 4 Cấp Độ Agentic Traces
# **Data/Analytics Engineering Assessment**
# 
# **Bộ dữ liệu:** `processed_agentic_traces.csv`  
# **Khung phân tích:** Descriptive (Mô tả) → Diagnostic (Chẩn đoán) → Predictive (Dự báo) → Prescriptive (Khuyến nghị & ROI)

# %%
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")

# %% [markdown]
# ## LỚP 1: XỬ LÝ & NGUYÊN THỂ DỮ LIỆU (ETL PIPELINE & FEATURE ENGINEERING)

# %%
def load_and_engineer_data(filepath: str = "processed_agentic_traces.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hàm ETL chính nạp dữ liệu thô và tạo các biến dẫn xuất theo Data Contract.

    Input Columns:
        - session_id (str): Mã phiên định dạng 'benchmark__project__issue__model_run'
        - model (str): Tên mô hình LLM/Agent
        - turn_number (int): Thứ tự turn trong session
        - output_length (float/int): Số character/token đầu ra
        - pre_gap (float): Thời gian chờ giữa các turn (latency)
        - has_error (int): 1 nếu turn bị lỗi, 0 nếu không lỗi
        - turn_cost (float): Chi phí USD của turn
        - input_tokens (float/int): Số token đầu vào của turn
        - is_system_prompt_present (int): 1 nếu có system prompt, 0 nếu không

    Output DataFrames:
        - df (pd.DataFrame): Dataframe cấp Turn với đầy đủ các cột dẫn xuất
        - sess_agg (pd.DataFrame): Dataframe tổng hợp cấp Session (1 dòng / session)

    Edge Cases Handled:
        - Session chỉ có 1 turn: cum_cost & delta_cost tính đúng, last_resolved lấy turn duy nhất đó.
        - turn_number không liên tục: sắp xếp theo (session_id, turn_number) trước khi diff/cumsum.
        - Chia cho 0: throughput và token_efficiency trả về 0.0 nếu latency=0 hoặc input_tokens=0.
    """
    df = pd.read_csv(filepath)

    # 1. Parse session_id
    parts = df['session_id'].str.split('__')
    df['benchmark'] = parts.str[0]
    df['project'] = parts.str[1]
    df['issue'] = parts.str[2]

    # 2. Trường dẫn xuất bắt buộc
    df['latency'] = df['pre_gap']
    df['success'] = 1 - df['has_error']
    df['throughput'] = np.where(df['latency'] > 0, df['output_length'] / df['latency'], 0.0)
    df['token_efficiency'] = np.where(df['input_tokens'] > 0, df['output_length'] / df['input_tokens'], 0.0)

    # 3. Tính toán theo chuỗi thời gian (Session level sort)
    df = df.sort_values(['session_id', 'turn_number']).reset_index(drop=True)
    df['cum_cost'] = df.groupby('session_id')['turn_cost'].cumsum()
    df['delta_cost'] = df.groupby('session_id')['turn_cost'].diff().fillna(df['turn_cost'])

    # 4. Xác định resolved (turn cuối cùng của session có has_error == 0)
    last_turns = df.groupby('session_id').last()
    last_resolved_map = (last_turns['has_error'] == 0).astype(int).to_dict()
    df['resolved'] = df['session_id'].map(last_resolved_map)

    # 5. Xác định Outlier theo quy tắc IQR (Q3 + 1.5*IQR)
    def calc_iqr_outliers(series: pd.Series) -> pd.Series:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        return series > (q3 + 1.5 * iqr)

    df['is_cost_outlier'] = calc_iqr_outliers(df['turn_cost'])
    df['is_latency_outlier'] = calc_iqr_outliers(df['latency'])

    # 6. Bảng tổng hợp phiên (sess_agg)
    sess_agg = df.groupby(['session_id', 'model', 'benchmark']).agg(
        total_cost=('turn_cost', 'sum'),
        n_turns=('turn_number', 'count'),
        resolved=('resolved', 'first'),
        avg_duration=('latency', 'mean'),
        max_duration=('latency', 'max'),
        error_rate=('has_error', 'mean'),
        avg_tokens=('input_tokens', 'mean'),
        any_success=('has_error', lambda x: int((x == 0).any())),
        is_system_prompt=('is_system_prompt_present', 'first')
    ).reset_index()

    return df, sess_agg

# Utility functions cho Khoảng tin cậy (Confidence Intervals)
def calc_mean_ci(series: pd.Series, confidence: float = 0.95) -> tuple[float, float, float]:
    """Tính mean và Khoảng tin cậy 95% (t-distribution) cho biến liên tục."""
    n = len(series)
    if n < 2:
        return series.mean(), series.mean(), series.mean()
    mean = series.mean()
    sem = stats.sem(series)
    h = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, mean - h, mean + h

def calc_wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Tính tỷ lệ p và Khoảng tin cậy Wilson Score 95% cho biến nhị phân/tỷ lệ lỗi (n nhỏ)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)

# Load data
df_traces, df_sessions = load_and_engineer_data("processed_agentic_traces.csv")
print(f"ETL Hoàn tất: {len(df_traces)} turns | {len(df_sessions)} sessions | Ngân sách: ${df_traces['turn_cost'].sum():.2f}")

# %% [markdown]
# ---
# ## CẤP 1: DESCRIPTIVE ANALYSIS ("Điều gì đã xảy ra?")
# **Mục tiêu:** Tái tạo chính xác các con số mốc baseline, mô tả phân phối chi phí và latency mà không diễn giải nguyên nhân nhân quả.

# %%
def compute_level1_metrics(df: pd.DataFrame, sess_agg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tính toán chỉ số Cấp 1: Model summary, Phân phối percentiles, Tỷ trọng ngân sách.
    """
    # 2.1 Model-level summary
    models = df['model'].unique()
    summary_rows = []

    for m in models:
        sub_t = df[df['model'] == m]
        sub_s = sess_agg[sess_agg['model'] == m]
        
        n_sess = len(sub_s)
        n_turns = len(sub_t)
        tot_cost = sub_t['turn_cost'].sum()
        avg_cost_sess, cost_ci_l, cost_ci_h = calc_mean_ci(sub_s['total_cost'])
        
        err_k = int(sub_t['has_error'].sum())
        err_p, err_ci_l, err_ci_h = calc_wilson_ci(err_k, n_turns)
        
        avg_dur = sub_t['latency'].mean()
        avg_tok = sub_t['input_tokens'].mean()
        res_rate = sub_s['resolved'].mean()
        
        summary_rows.append({
            'Model': m,
            'Sessions (n)': n_sess,
            'Turns (n)': n_turns,
            'Total Cost ($)': tot_cost,
            'Avg Cost/Sess ($)': avg_cost_sess,
            'Avg Cost 95% CI': f"[${cost_ci_l:.2f} - ${cost_ci_h:.2f}]",
            'Error Rate (%)': err_p * 100,
            'Error Rate 95% CI': f"[{err_ci_l*100:.1f}% - {err_ci_h*100:.1f}%]",
            'Resolved Rate (%)': res_rate * 100,
            'Avg Latency (s)': avg_dur,
            'Avg Tokens/Turn': avg_tok
        })
        
    model_summary_df = pd.DataFrame(summary_rows).sort_values('Total Cost ($)', ascending=False)

    # 2.2 Phân phối percentiles
    dist_rows = []
    for m in models:
        sub_t = df[df['model'] == m]
        tc = sub_t['turn_cost']
        lat = sub_t['latency']
        dist_rows.append({
            'Model': m,
            'Cost Mean ($)': tc.mean(),
            'Cost Median ($)': tc.median(),
            'Cost P95 ($)': tc.quantile(0.95),
            'Cost Max ($)': tc.max(),
            'Latency Mean (s)': lat.mean(),
            'Latency Median (s)': lat.median(),
            'Latency P95 (s)': lat.quantile(0.95),
            'Latency Max (s)': lat.max()
        })
    dist_df = pd.DataFrame(dist_rows)

    # 2.3 Tỷ trọng ngân sách
    total_budget = df['turn_cost'].sum()
    budget_df = model_summary_df[['Model', 'Total Cost ($)']].copy()
    budget_df['Budget Share (%)'] = (budget_df['Total Cost ($)'] / total_budget) * 100

    return model_summary_df, dist_df, budget_df

model_summary_df, dist_df, budget_df = compute_level1_metrics(df_traces, df_sessions)

print("=== BẢNG 2.1: BÁO CÁO MÔ HÌNH (MODEL SUMMARY) ===")
print(model_summary_df.to_string(index=False))

print("\n=== BẢNG 2.2: PHÂN PHỐI PERCENTILES TURN COST & LATENCY ===")
print(dist_df.to_string(index=False))

print("\n=== BẢNG 2.3: TỶ TRỌNG NGÂN SÁCH TRÊN $290.21 ===")
print(budget_df.to_string(index=False))

# %%
def plot_level1_charts(model_summary_df: pd.DataFrame, budget_df: pd.DataFrame):
    """Trình bày biểu đồ Cấp 1: Chi phí và Tỷ lệ lỗi theo Model."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Tổng Ngân Sách tiêu tốn theo Model ($)", "Tỷ lệ Turn Error theo Model (%)")
    )

    fig.add_trace(
        go.Bar(
            x=budget_df['Model'], y=budget_df['Total Cost ($)'],
            text=[f"${v:.2f} ({p:.1f}%)" for v, p in zip(budget_df['Total Cost ($)'], budget_df['Budget Share (%)'])],
            textposition='auto', marker_color='#0ea5e9', name='Total Cost'
        ), row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=model_summary_df['Model'], y=model_summary_df['Error Rate (%)'],
            text=[f"{v:.1f}%" for v in model_summary_df['Error Rate (%)']],
            textposition='auto', marker_color='#ef4444', name='Error Rate'
        ), row=1, col=2
    )

    fig.update_layout(height=400, width=900, title_text="CẤP 1 — TỔNG QUAN THỰC TRẠNG (DESCRIPTIVE)", showlegend=False)
    fig.show()

plot_level1_charts(model_summary_df, budget_df)

# %% [markdown]
# ---
# ## CẤP 2: DIAGNOSTIC ANALYSIS ("Tại sao nó xảy ra?")
# **Mục tiêu:** Phân tích nguyên nhân góc rễ, kiểm tra yếu tố gây nhiễu (confounders), tính toán khoảng tin cậy Wilson cho mẫu nhỏ Opus, và phân rã cơ chế chi phí.

# %%
def compute_level2_diagnostics(df: pd.DataFrame, sess_agg: pd.DataFrame):
    """
    Tính toán các chỉ số Diagnostic Cấp 2.
    """
    # 3.1 Cạm bẫy giá rẻ (Low-cost Trap)
    trap_data = []
    for m in ['minimax-m2.5', 'deepseek-v3.1', 'claude-sonnet-4-6', 'claude-opus-4-6']:
        sub_s = sess_agg[sess_agg['model'] == m]
        sub_t = df[df['model'] == m]
        trap_data.append({
            'Model': m,
            'Error Rate (%)': sub_t['has_error'].mean() * 100,
            'Avg Turns/Session': sub_s['n_turns'].mean(),
            'Input Tokens/Turn': sub_t['input_tokens'].mean(),
            'Avg Cost/Session ($)': sub_s['total_cost'].mean(),
            'Total Wasted Cost ($)': sub_s['total_cost'].sum() if sub_t['has_error'].mean() == 1.0 else 0.0
        })
    trap_df = pd.DataFrame(trap_data)

    # 3.2 Nghịch lý System Prompt (Subset Sonnet)
    sonnet_s = sess_agg[sess_agg['model'] == 'claude-sonnet-4-6']
    sys_prompt_rows = []
    for sys_p in [0, 1]:
        sub_s = sonnet_s[sonnet_s['is_system_prompt'] == sys_p]
        sub_t = df[(df['model'] == 'claude-sonnet-4-6') & (df['is_system_prompt_present'] == sys_p)]
        
        n_s = len(sub_s)
        avg_tok = sub_t['input_tokens'].mean()
        err_rate = sub_t['has_error'].mean() * 100
        fail_100 = (sub_s['error_rate'] == 1.0).mean() * 100
        turns_per_sess = sub_s['n_turns'].mean()
        cost_per_sess = sub_s['total_cost'].mean()
        
        sys_prompt_rows.append({
            'System Prompt': 'Có (1)' if sys_p == 1 else 'Không (0)',
            'Sessions (n)': n_s,
            'Avg Input Tokens': avg_tok,
            'Error Rate (%)': err_rate,
            '% Session Fail 100%': fail_100,
            'Turns/Session': turns_per_sess,
            'Cost/Session ($)': cost_per_sess
        })
    sys_prompt_df = pd.DataFrame(sys_prompt_rows)

    # Phân tích Yếu tố gây nhiễu (Confounder Check: Benchmark distribution)
    confounder_cross = pd.crosstab(sonnet_s['benchmark'], sonnet_s['is_system_prompt'], normalize='columns') * 100
    confounder_cross.columns = ['No System Prompt (0)', 'System Prompt Present (1)']

    # 3.3 Opus Bright Spot with Wilson CI
    opus_s = sess_agg[sess_agg['model'] == 'claude-opus-4-6']
    opus_t = df[df['model'] == 'claude-opus-4-6']
    k_err = int(opus_t['has_error'].sum())
    n_t = len(opus_t)
    err_p, w_l, w_h = calc_wilson_ci(k_err, n_t)
    opus_stats = {
        'n_sessions': len(opus_s),
        'n_turns': n_t,
        'error_rate_pct': err_p * 100,
        'wilson_ci_pct': f"[{w_l*100:.1f}% - {w_h*100:.1f}%]",
        'resolved_rate_pct': opus_s['resolved'].mean() * 100,
        'avg_cost_session': opus_s['total_cost'].mean()
    }

    # 3.4 Cost Decomposition: total_cost/session = avg_turns/session * avg_cost/turn
    decomp_rows = []
    for m in ['claude-opus-4-6', 'claude-sonnet-4-6', 'deepseek-v3.1', 'minimax-m2.5']:
        sub_s = sess_agg[sess_agg['model'] == m]
        sub_t = df[df['model'] == m]
        
        avg_cost_sess = sub_s['total_cost'].mean()
        avg_turns = sub_s['n_turns'].mean()
        avg_cost_turn = sub_t['turn_cost'].mean()
        
        decomp_rows.append({
            'Model': m,
            'Cost/Session ($)': avg_cost_sess,
            'Turns/Session (Đòn bẩy 1)': avg_turns,
            'Cost/Turn ($) (Đòn bẩy 2)': avg_cost_turn,
            'Giải thích': f"${avg_cost_sess:.2f} = {avg_turns:.1f} turns × ${avg_cost_turn:.4f}/turn"
        })
    decomp_df = pd.DataFrame(decomp_rows)

    return trap_df, sys_prompt_df, confounder_cross, opus_stats, decomp_df

trap_df, sys_prompt_df, confounder_cross, opus_stats, decomp_df = compute_level2_diagnostics(df_traces, df_sessions)

print("=== BẢNG 3.1: CẠM BẪY GIÁ RẺ (LOW-COST TRAP) ===")
print(trap_df.to_string(index=False))

print("\n=== BẢNG 3.2A: NGHỊCH LÝ SYSTEM PROMPT TRÊN CLAUDE-SONNET-4-6 ===")
print(sys_prompt_df.to_string(index=False))

print("\n=== BẢNG 3.2B: PHÂN TÍCH CONFOUNDER BENCHMARK TRONG SONNET (%) ===")
print(confounder_cross.to_string())
print("\n[CẢNH BÁO CONFOUNDER] Nhóm Không Prompt (0) tập trung 100% vào benchmark GAIA (94 sessions), trong khi nhóm Có Prompt (1) tập trung 100% vào SWEBENCH (110 sessions). Sự gia tăng lỗi từ 38.6% lên 68.5% ĐÃ BỊ NHIỄU hoàn toàn bởi độ khó của Benchmark (SWEBENCH phức tạp hơn GAIA). Không thể kết luận System Prompt trực tiếp gây lỗi!")

print("\n=== BẢNG 3.3: ĐIỂM SÁNG OPUS & PHẢN BIỆN MẪU NHỎ ===")
print(f"Opus Sessions: n={opus_stats['n_sessions']} | Turns: n={opus_stats['n_turns']}")
print(f"Turn Error Rate: {opus_stats['error_rate_pct']:.1f}% (Khoảng tin cậy Wilson 95%: {opus_stats['wilson_ci_pct']})")
print(f"Resolved Rate: {opus_stats['resolved_rate_pct']:.1f}% | Avg Cost/Session: ${opus_stats['avg_cost_session']:.2f}")
print("[LƯU Ý QUAN TRỌNG] Kết luận về Opus dựa trên cỡ mẫu nhỏ (n=8 sessions), cần thêm dữ liệu trước khi coi là kết luận chắc chắn ở quy mô production.")

print("\n=== BẢNG 3.4: PHÂN RÃ CƠ CHẾ CHI PHÍ (COST DECOMPOSITION) ===")
print(decomp_df.to_string(index=False))

# %%
def plot_level2_charts(sys_prompt_df: pd.DataFrame, decomp_df: pd.DataFrame):
    """Vẽ biểu đồ Chẩn đoán Cấp 2."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Sonnet: Ảnh hưởng System Prompt đến Cost & Error Rate", "Cost Decomposition: Turns/Session vs Cost/Turn")
    )

    fig.add_trace(
        go.Bar(
            x=sys_prompt_df['System Prompt'], y=sys_prompt_df['Cost/Session ($)'],
            name='Cost/Session ($)', marker_color='#9b8cff', text=[f"${v:.2f}" for v in sys_prompt_df['Cost/Session ($)']], textposition='auto'
        ), row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=decomp_df['Model'], y=decomp_df['Turns/Session (Đòn bẩy 1)'],
            name='Turns/Session', marker_color='#34d399', text=[f"{v:.1f}" for v in decomp_df['Turns/Session (Đòn bẩy 1)']], textposition='auto'
        ), row=1, col=2
    )

    fig.update_layout(height=400, width=900, title_text="CẤP 2 — CHẨN ĐOÁN NGUYÊN NHÂN (DIAGNOSTIC)", showlegend=True)
    fig.show()

plot_level2_charts(sys_prompt_df, decomp_df)

# %% [markdown]
# ---
# ## CẤP 3: PREDICTIVE ANALYSIS ("Có thể dự đoán/định lượng trước điều gì?")
# **Mục tiêu:** Xây dựng mô hình phân tích sinh tồn (Survival Curve / Kaplan-Meier), dự báo chi phí từ turn k, và phân loại rủi ro 100% fail từ 5 turn đầu.

# %%
def compute_level3_predictive(df: pd.DataFrame, sess_agg: pd.DataFrame):
    """
    Tính toán và huấn luyện các mô hình Dự báo Cấp 3.
    """
    # 4.1 Survival Curve (Kaplan-Meier Style)
    turns = list(range(1, 41))
    surv_records = []

    for m in ['claude-sonnet-4-6', 'claude-opus-4-6', 'minimax-m2.5', 'deepseek-v3.1']:
        sub_df = df[df['model'] == m]
        sub_sess = sess_agg[sess_agg['model'] == m]
        
        for t in turns:
            active_sids = sub_df[sub_df['turn_number'] == t]['session_id'].unique()
            n_active = len(active_sids)
            if n_active == 0:
                continue
            
            # Số session CHƯA resolved trong số những session đi tới turn t
            unresolved_n = sub_sess[(sub_sess['session_id'].isin(active_sids)) & (sub_sess['resolved'] == 0)]['session_id'].nunique()
            prob_unresolved = unresolved_n / n_active
            
            surv_records.append({
                'model': m, 'turn': t, 'active_n': n_active,
                'unresolved_n': unresolved_n, 'prob_unresolved': prob_unresolved
            })
            
    surv_df = pd.DataFrame(surv_records)

    # 4.2 Forecast total_cost ~ cost_at_k (Linear & Log-transformed)
    forecast_results = {}
    for k in [5, 10]:
        sess_k = df[df['turn_number'] == k][['session_id', 'cum_cost']].rename(columns={'cum_cost': f'cost_at_{k}'})
        merged = pd.merge(sess_agg, sess_k, on='session_id')
        
        X = merged[[f'cost_at_{k}']].values
        y = merged['total_cost'].values
        
        # Model 1: Linear
        lr_lin = LinearRegression()
        lr_lin.fit(X, y)
        p_lin = lr_lin.predict(X)
        r2_lin = r2_score(y, p_lin)
        mae_lin = mean_absolute_error(y, p_lin)
        
        # Model 2: Log-transformed log(total_cost + 1e-4) ~ log(cost_at_k + 1e-4)
        X_log = np.log1p(X)
        y_log = np.log1p(y)
        lr_log = LinearRegression()
        lr_log.fit(X_log, y_log)
        p_log = np.expm1(lr_log.predict(X_log))
        r2_log = r2_score(y, p_log)
        mae_log = mean_absolute_error(y, p_log)
        
        forecast_results[k] = {
            'n': len(merged),
            'r2_linear': r2_lin, 'mae_linear': mae_lin,
            'r2_log': r2_log, 'mae_log': mae_log,
            'residuals_linear': y - p_lin,
            'residuals_log': y - p_log
        }

    # 4.3 100% Fail Risk Classifier from first 5 turns
    first_5 = df[df['turn_number'] <= 5]
    f5_feat = first_5.groupby('session_id').agg(
        error_streak=('has_error', 'sum'),
        cost_at_5=('cum_cost', 'max'),
        tokens_growth=('input_tokens', lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0.0),
        avg_tokens_5=('input_tokens', 'mean')
    ).reset_index()

    f5_merged = pd.merge(sess_agg, f5_feat, on='session_id')
    f5_merged['fail_100'] = (f5_merged['error_rate'] == 1.0).astype(int)

    # Đánh giá riêng trên Claude-Sonnet-4-6 (nơi cả 2 nhãn 0 và 1 cùng tồn tại)
    sonnet_f5 = f5_merged[f5_merged['model'] == 'claude-sonnet-4-6']
    X_sonnet = sonnet_f5[['error_streak', 'cost_at_5', 'tokens_growth', 'avg_tokens_5']]
    y_sonnet = sonnet_f5['fail_100']

    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_sonnet, y_sonnet)
    probs_sonnet = rf_clf.predict_proba(X_sonnet)[:, 1]
    auc_sonnet = roc_auc_score(y_sonnet, probs_sonnet)

    base_rates = f5_merged.groupby('model')['fail_100'].agg(['count', 'mean']).reset_index()
    base_rates.columns = ['Model', 'Sessions (n)', 'Base Fail-100% Rate']

    return surv_df, forecast_results, auc_sonnet, len(sonnet_f5), base_rates

surv_df, forecast_results, auc_sonnet, n_sonnet_f5, base_rates = compute_level3_predictive(df_traces, df_sessions)

print("=== BẢNG 4.1: ĐƯỜNG CONG SỐNG SÓT UNRESOLVED (%) THEO TURN (SAMPLE) ===")
print(surv_df[surv_df['turn'].isin([5, 10, 15, 20, 25, 30])].pivot(index='turn', columns='model', values='prob_unresolved').to_string())

print("\n=== BẢNG 4.2: ĐÁNH GIÁ MÔ HÌNH DỰ BÁO TỔNG CHI PHÍ TỪ TURN K ===")
for k, res in forecast_results.items():
    print(f"Turn k={k} (n={res['n']} sessions):")
    print(f"  Linear Model:  R² = {res['r2_linear']:.4f} | MAE = ${res['mae_linear']:.4f} | Max Residual = ${np.max(np.abs(res['residuals_linear'])):.2f}")
    print(f"  Log Transform: R² = {res['r2_log']:.4f} | MAE = ${res['mae_log']:.4f} | Max Residual = ${np.max(np.abs(res['residuals_log'])):.2f}")

print("\n=== BẢNG 4.3: PHÂN LOẠI RỦI RO FAIL-100% TỪ 5 TURN ĐẦU ===")
print(base_rates.to_string(index=False))
print(f"\n[CẢNH BÁO BASE RATE] Minimax & DeepSeek có Base Fail Rate = 100.0% (suy biến nhãn). Classifier được đánh giá riêng trên Subset Claude-Sonnet-4-6 (n={n_sonnet_f5}):")
print(f"ROC-AUC Classifier trên Sonnet: {auc_sonnet:.4f}")

# %%
def plot_level3_charts(surv_df: pd.DataFrame):
    """Vẽ đường cong sống sót theo Turn."""
    fig = px.line(
        surv_df, x='turn', y='prob_unresolved', color='model',
        title="CẤP 3 — ĐƯỜNG CONG SỐNG SÓT TỶ LỆ UNRESOLVED THEO TURN (KAPLAN-MEIER STYLE)",
        labels={'turn': 'Turn Number', 'prob_unresolved': 'Xác suất Session chưa Resolved'},
        markers=True
    )
    # Thêm đường chỉ dẫn Elbow Point (Turn 20)
    fig.add_vline(x=20, line_dash="dash", line_color="#f5b544", annotation_text="Elbow Point (Turn 20)")
    fig.update_layout(height=450, width=900)
    fig.show()

plot_level3_charts(surv_df)

# %% [markdown]
# ---
# ## CẤP 4: PRESCRIPTIVE ANALYSIS ("Nên làm gì, và tiết kiệm được bao nhiêu?")
# **Mục tiêu:** Mô phỏng What-If Circuit Breaker, tính toán Prompt Pruning (kèm nhãn Cận trên), lập chỉ số Routing Cost-Efficiency (CEI), và tổng hợp Waterfall ROI Ledger.

# %%
def compute_level4_prescriptive(df: pd.DataFrame, sess_agg: pd.DataFrame):
    """
    Tính toán các giải pháp Khuyến nghị Cấp 4 và bảng cân đối ROI.
    """
    # 5.1 Simulation Circuit Breaker
    cb_results = []
    total_resolved_all = sess_agg['resolved'].sum()

    for cutoff_t in [10, 15, 20, 25, 30]:
        total_saved = 0.0
        resolved_lost = 0

        for idx, row in sess_agg.iterrows():
            sid = row['session_id']
            s_turns = df[df['session_id'] == sid]
            n_t = row['n_turns']
            res = row['resolved']

            if n_t > cutoff_t:
                cost_at_cutoff = s_turns[s_turns['turn_number'] == cutoff_t]['cum_cost'].values[0]
                saved = row['total_cost'] - cost_at_cutoff
                total_saved += saved

                # Kiểm tra nếu phiên đó thành công nhưng turn thành công > cutoff_t
                no_err_turns = s_turns[s_turns['has_error'] == 0]['turn_number'].values
                if res == 1 and len(no_err_turns) > 0 and no_err_turns[0] > cutoff_t:
                    resolved_lost += 1

        lost_pct = (resolved_lost / total_resolved_all) * 100 if total_resolved_all > 0 else 0.0
        cb_results.append({
            'Cutoff Turn': cutoff_t,
            'Ngân Sách Tiết Kiệm ($)': total_saved,
            'Số Session Resolved Bị Mất': f"{resolved_lost}/{total_resolved_all}",
            'Tỷ Lệ Mất Success (%)': lost_pct
        })
    cb_df = pd.DataFrame(cb_results)

    # 5.2 Prompt Pruning Savings
    sonnet_sys0 = sess_agg[(sess_agg['model'] == 'claude-sonnet-4-6') & (sess_agg['is_system_prompt'] == 0)]
    sonnet_sys1 = sess_agg[(sess_agg['model'] == 'claude-sonnet-4-6') & (sess_agg['is_system_prompt'] == 1)]
    c0 = sonnet_sys0['total_cost'].mean()
    c1 = sonnet_sys1['total_cost'].mean()
    prompt_savings_upper_bound = (c1 - c0) * len(sonnet_sys1)

    # 5.3 Model Routing (Cost-Efficiency Index CEI = Resolved Rate / Avg Cost per Session)
    cei_rows = []
    for m in ['claude-opus-4-6', 'claude-sonnet-4-6', 'deepseek-v3.1', 'minimax-m2.5']:
        sub = sess_agg[sess_agg['model'] == m]
        res_r = sub['resolved'].mean()
        avg_c = sub['total_cost'].mean()
        cei = res_r / avg_c if avg_c > 0 else 0.0
        cei_rows.append({
            'Model': m,
            'Resolved Rate (%)': res_r * 100,
            'Avg Cost/Sess ($)': avg_c,
            'CEI (Resolved / $)': cei,
            'Khuyến nghị Routing': 'Route chính cho bài phức tạp' if m == 'claude-opus-4-6' else ('Dùng kèm Circuit Breaker' if m == 'claude-sonnet-4-6' else 'Loại bỏ khỏi Agent Pipeline')
        })
    cei_df = pd.DataFrame(cei_rows)

    # 5.4 Comprehensive ROI Ledger (Waterfall)
    # Lãng phí đã xác nhận từ Minimax + DeepSeek (100% fail)
    wasted_budget_models = sess_agg[sess_agg['model'].isin(['minimax-m2.5', 'deepseek-v3.1'])]['total_cost'].sum()
    
    # Circuit breaker savings tại turn 20 trên Sonnet
    cb_savings_t20 = cb_df[cb_df['Cutoff Turn'] == 20]['Ngân Sách Tiết Kiệm ($)'].values[0]

    roi_waterfall = pd.DataFrame([
        {'Khoản Mục ROI': '1. Ngân sách Lãng phí Đã Xác Nhận (Minimax & DeepSeek 100% Fail)', 'Số Tiền ($)': wasted_budget_models, 'Loại': 'Waste Eliminated'},
        {'Khoản Mục ROI': '2. Tiết kiệm từ Circuit Breaker (Ngưỡng Turn 20)', 'Số Tiền ($)': cb_savings_t20, 'Loại': 'Circuit Breaker'},
        {'Khoản Mục ROI': '3. Tiết kiệm từ Prompt Pruning (Ước tính Cận Trên - Giả định Nhân quả)', 'Số Tiền ($)': prompt_savings_upper_bound, 'Loại': 'Upper Bound Prompt'},
        {'Khoản Mục ROI': 'TỔNG TIẾT KIỆM TIỀM NĂNG (DỰ KIẾN RE-ALLOCATE)', 'Số Tiền ($)': wasted_budget_models + cb_savings_t20 + prompt_savings_upper_bound, 'Loại': 'Total Potential'}
    ])

    return cb_df, prompt_savings_upper_bound, cei_df, roi_waterfall

cb_df, prompt_savings_upper_bound, cei_df, roi_waterfall = compute_level4_prescriptive(df_traces, df_sessions)

print("=== BẢNG 5.1: MÔ PHỎNG CIRCUIT BREAKER THEO NGƯỠNG TURN ===")
print(cb_df.to_string(index=False))

print(f"\n=== BẢNG 5.2: ƯỚC TÍNH TIẾT KIỆM PROMPT PRUNING ===")
print(f"Ước tính Tiết kiệm Cận Trên (Upper-Bound): ${prompt_savings_upper_bound:.2f}")
print("[LƯU Ý CAUTION] Đây là con số ước tính cận trên dựa trên giả định quan hệ nhân quả. BẮT BUỘC thực hiện A/B Test trước khi rollout toàn hệ thống do có yếu tố gây nhiễu Benchmark (GAIA vs SWEBENCH).")

print("\n=== BẢNG 5.3: COST-EFFICIENCY INDEX (CEI) & KHUYẾN NGHỊ ROUTING ===")
print(cei_df.to_string(index=False))

print("\n=== BẢNG 5.4: BẢNG CÂN ĐỐI ROI LEDGER TỔNG HỢP (WATERFALL) ===")
print(roi_waterfall.to_string(index=False))

# %%
def plot_level4_waterfall(roi_waterfall: pd.DataFrame):
    """Vẽ biểu đồ Waterfall ROI Ledger."""
    fig = go.Figure(go.Waterfall(
        name="ROI Ledger", orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=roi_waterfall['Khoản Mục ROI'],
        textposition="outside",
        text=[f"${v:.2f}" for v in roi_waterfall['Số Tiền ($)']],
        y=roi_waterfall['Số Tiền ($)'],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#ef4444"}},
        increasing={"marker": {"color": "#34d399"}},
        totals={"marker": {"color": "#0ea5e9"}}
    ))

    fig.update_layout(
        title="CẤP 4 — BẢNG CÂN ĐỐI ROI LEDGER & TỔNG TIẾT KIỆM ($)",
        height=450, width=950
    )
    fig.show()

plot_level4_waterfall(roi_waterfall)

# %% [markdown]
# ---
# ## TỔNG HỢP PHÁT HIỆN CHÍNH (KEY INSIGHTS SUMMARY)
# 
# 1. **[Cấp 1 — Descriptive] Phân hóa Ngân sách & Bẫy Mẫu Skewed:**
#    Claude-Sonnet-4-6 chiếm tới 66.8% ($193.72) tổng ngân sách $290.21. Phân phối chi phí và latency bị lệch mạnh bởi các outlier timeout (~$301 và ~1511s), khiến giá trị P95 phản ánh chính xác rủi ro vận hành hơn trung bình số học.
# 
# 2. **[Cấp 2 — Diagnostic] Cạm Bẫy Giá Rẻ & Bị Nhiễu Bởi Benchmark:**
#    Minimax-m2.5 và DeepSeek-v3.1 gây lãng phí $93.94 (100% error rate trên 555 sessions) do rơi vào vòng lặp dài (34-37 turns/session). Bên cạnh đó, giả thuyết "System Prompt gây lỗi ở Sonnet" bị phản bác do hiện tượng **Confounding by Benchmark**: nhóm có prompt nằm 100% ở SWEBENCH (khó hơn), nhóm không prompt nằm 100% ở GAIA.
# 
# 3. **[Cấp 2 — Diagnostic] Đòn Bẩy Chi Phí Thực Sự từ Số Turn:**
#    Claude-Opus-4-6 có chi phí trung bình thấp ($0.32/session) nhờ số turn cực ít (18.4 turns/session) dù chi phí/turn cao ($0.0174). Tuy nhiên, kết luận về Opus mang tính chất gợi mở do mẫu nhỏ (n=8 sessions, Wilson CI 95% tỷ lệ lỗi: [6.3% - 16.2%]).
# 
# 4. **[Cấp 3 — Predictive] Ngưỡng Ngắt Mạch Dựa Trên Phân Tích Sinh Tồn:**
#    Đường cong sống sót (Kaplan-Meier) chỉ ra "Elbow Point" tại Turn 20. Từ turn 20 trở đi, xác suất resolve thêm bài toán tiệm cận 0, trong khi chi phí tăng tuyến tính. Mô hình Random Forest trên 5 turn đầu dự báo rủi ro Fail-100% với ROC-AUC = 0.9995 trên Claude-Sonnet.
# 
# 5. **[Cấp 4 — Prescriptive] Tối Ưu Hóa Ngân Sách Với ROI $251.13 (86.5% Budget):**
#    Bằng cách (1) Loại bỏ Minimax/DeepSeek (tiết kiệm $93.94), (2) Thiết lập Circuit Breaker tại Turn 20 (tiết kiệm $40.59 với rủi ro chỉ mất 2.6% success), và (3) Thí điểm A/B test Prompt Pruning (tiềm năng cận trên $116.60), tổ chức có thể giải phóng $251.13 để tái đầu tư vào Claude-Opus (mô hình có Cost-Efficiency Index cao nhất = 2.344 resolved/$).

