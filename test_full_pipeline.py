import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('processed_agentic_traces.csv')

# 1. ETL & Parse
parts = df['session_id'].str.split('__')
df['benchmark'] = parts.str[0]
df['project'] = parts.str[1]
df['issue'] = parts.str[2]

df['latency'] = df['pre_gap']
df['success'] = 1 - df['has_error']
df['throughput'] = np.where(df['latency'] > 0, df['output_length'] / df['latency'], 0.0)
df['token_efficiency'] = np.where(df['input_tokens'] > 0, df['output_length'] / df['input_tokens'], 0.0)

df = df.sort_values(['session_id', 'turn_number']).reset_index(drop=True)
df['cum_cost'] = df.groupby('session_id')['turn_cost'].cumsum()
df['delta_cost'] = df.groupby('session_id')['turn_cost'].diff().fillna(df['turn_cost'])

# Last turn resolved
last_turns = df.groupby('session_id').last()
last_resolved_map = (last_turns['has_error'] == 0).astype(int).to_dict()
df['resolved'] = df['session_id'].map(last_resolved_map)

# Outlier flags
def iqr_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return series > (q3 + 1.5 * iqr)

df['is_cost_outlier'] = iqr_outliers(df['turn_cost'])
df['is_latency_outlier'] = iqr_outliers(df['latency'])

# Session Aggregation
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

print("--- LEVEL 1: DESCRIPTIVE ---")
model_summary = df.groupby('model').agg(
    n_sessions=('session_id', 'nunique'),
    n_turns=('turn_number', 'count'),
    total_cost=('turn_cost', 'sum'),
    avg_cost_session=('turn_cost', lambda x: x.sum() / x.index.to_series().map(df['session_id']).nunique()), # handle grouped total cost
    error_rate=('has_error', 'mean'),
    avg_duration=('latency', 'mean'),
    avg_tokens=('input_tokens', 'mean')
).reset_index()

# Fix avg_cost_session calculation
for idx, row in model_summary.iterrows():
    m = row['model']
    m_sess = sess_agg[sess_agg['model'] == m]
    model_summary.loc[idx, 'avg_cost_session'] = m_sess['total_cost'].mean()
    model_summary.loc[idx, 'resolved_rate'] = m_sess['resolved'].mean()

print(model_summary[['model', 'n_sessions', 'n_turns', 'avg_cost_session', 'total_cost', 'error_rate', 'resolved_rate']])

print("\nCost proportion:")
total_budget = df['turn_cost'].sum()
for m, g in df.groupby('model'):
    c = g['turn_cost'].sum()
    print(f"  {m}: ${c:.2f} ({c/total_budget*100:.2f}%)")

print("\nTurn Cost & Latency percentiles:")
for m, g in df.groupby('model'):
    tc = g['turn_cost']
    lat = g['latency']
    print(f"  {m} turn_cost: mean={tc.mean():.4f}, median={tc.median():.4f}, p95={tc.quantile(0.95):.4f}, max={tc.max():.4f}")
    print(f"  {m} latency  : mean={lat.mean():.2f}, median={lat.median():.2f}, p95={lat.quantile(0.95):.2f}, max={lat.max():.2f}")

print("\n--- LEVEL 2: DIAGNOSTIC ---")
# 3.1 Cheap Trap
print("3.1 Low Cost Trap:")
for m in ['minimax-m2.5', 'deepseek-v3.1', 'claude-sonnet-4-6', 'claude-opus-4-6']:
    s = sess_agg[sess_agg['model'] == m]
    t = df[df['model'] == m]
    print(f"  {m}: error_rate={t['has_error'].mean()*100:.1f}%, turns/sess={s['n_turns'].mean():.1f}, input_tok/turn={t['input_tokens'].mean():.1f}, total_cost=${s['total_cost'].sum():.2f}")

# 3.2 System Prompt Paradox (Sonnet)
print("\n3.2 Sonnet System Prompt Paradox:")
sonnet_s = sess_agg[sess_agg['model'] == 'claude-sonnet-4-6']
for sys_p in [0, 1]:
    sub_s = sonnet_s[sonnet_s['is_system_prompt'] == sys_p]
    sub_t = df[(df['model'] == 'claude-sonnet-4-6') & (df['is_system_prompt_present'] == sys_p)]
    fail100 = (sub_s['error_rate'] == 1.0).mean()
    print(f"  sys_prompt={sys_p}: n={len(sub_s)}, avg_tok={sub_t['input_tokens'].mean():.1f}, err_rate={sub_t['has_error'].mean()*100:.1f}%, fail100={fail100*100:.1f}%, turns/sess={sub_s['n_turns'].mean():.1f}, cost/sess=${sub_s['total_cost'].mean():.2f}")

# Confounder check for system prompt: Benchmark distribution
print("\nBenchmark distribution by system prompt in Sonnet:")
print(pd.crosstab(sonnet_s['benchmark'], sonnet_s['is_system_prompt'], normalize='columns'))

# 3.3 Opus Bright Spot with Wilson CI
print("\n3.3 Opus Bright Spot:")
opus_s = sess_agg[sess_agg['model'] == 'claude-opus-4-6']
opus_t = df[df['model'] == 'claude-opus-4-6']

# Wilson interval function
def wilson_ci(k, n, alpha=0.05):
    if n == 0: return 0.0, 0.0
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)

opus_err_count = int(opus_t['has_error'].sum())
opus_turn_n = len(opus_t)
w_low, w_high = wilson_ci(opus_err_count, opus_turn_n)
print(f"  Opus turn error rate: {opus_err_count}/{opus_turn_n} = {opus_t['has_error'].mean()*100:.1f}% [Wilson 95% CI: {w_low*100:.1f}% - {w_high*100:.1f}%]")

# 3.4 Cost Decomposition
print("\n3.4 Cost Decomposition (total_cost/sess = avg_turns/sess * avg_cost/turn):")
for m in ['claude-opus-4-6', 'claude-sonnet-4-6', 'deepseek-v3.1', 'minimax-m2.5']:
    s = sess_agg[sess_agg['model'] == m]
    t = df[df['model'] == m]
    avg_turns = s['n_turns'].mean()
    avg_cost_turn = t['turn_cost'].mean()
    cost_sess = avg_turns * avg_cost_turn
    print(f"  {m}: ${cost_sess:.3f}/sess = {avg_turns:.1f} turns/sess * ${avg_cost_turn:.4f}/turn")

