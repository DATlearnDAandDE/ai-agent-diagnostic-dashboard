import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np

df = pd.read_csv('processed_agentic_traces.csv')
print('Shape:', df.shape)
print()
print('Columns:', list(df.columns))
print()
print('dtypes:')
print(df.dtypes)
print()
print('=== Describe ===')
print(df.describe())
print()
print('=== Unique models ===')
print(df['model'].unique())
print()
print('=== Value counts model ===')
print(df['model'].value_counts())
print()
print('=== is_system_prompt_present ===')
print(df['is_system_prompt_present'].value_counts())
print()
print('=== has_error ===')
print(df['has_error'].value_counts())
print()
print('=== turn_number range ===')
print('min:', df['turn_number'].min(), 'max:', df['turn_number'].max())
print()
print('=== turn_cost stats ===')
print(df['turn_cost'].describe())
print()
print('=== pre_gap stats ===')
print(df['pre_gap'].describe())
print()
print('=== input_tokens stats ===')
print(df['input_tokens'].describe())
print()
print('=== output_length stats ===')
print(df['output_length'].describe())
print()
print('=== session_id nunique ===')
print(df['session_id'].nunique())
print()
print('=== Sample session_ids ===')
print(df['session_id'].unique()[:5])
print()

# Per-model detailed stats
print('=== Per Model Error Rate ===')
print(df.groupby('model')['has_error'].mean())
print()
print('=== Per Model Cost Sum ===')
print(df.groupby('model')['turn_cost'].sum())
print()
print('=== Per Model Cost Mean ===')
print(df.groupby('model')['turn_cost'].mean())
print()
print('=== Per Model Latency Mean ===')
print(df.groupby('model')['pre_gap'].mean())
print()
print('=== Per Model Latency Median ===')
print(df.groupby('model')['pre_gap'].median())
print()
print('=== Per Model Turn Count ===')
print(df.groupby('model')['turn_number'].max().describe())
print()

# System prompt impact analysis
print('=== Error Rate by Model x System Prompt ===')
pivot = df.groupby(['model', 'is_system_prompt_present'])['has_error'].agg(['mean', 'count']).reset_index()
print(pivot.to_string())
print()

# Session-level analysis
session_stats = df.groupby('session_id').agg(
    model=('model', 'first'),
    total_turns=('turn_number', 'max'),
    total_cost=('turn_cost', 'sum'),
    error_rate=('has_error', 'mean'),
    total_errors=('has_error', 'sum'),
    avg_latency=('pre_gap', 'mean'),
    max_latency=('pre_gap', 'max'),
    total_output=('output_length', 'sum'),
    avg_input_tokens=('input_tokens', 'mean'),
).reset_index()

print('=== Session Stats Describe ===')
print(session_stats.describe())
print()
print('=== Sessions per model ===')
print(session_stats.groupby('model').size())
print()
print('=== Avg turns per session per model ===')
print(session_stats.groupby('model')['total_turns'].mean())
print()
print('=== Avg cost per session per model ===')
print(session_stats.groupby('model')['total_cost'].mean())
print()
print('=== Sessions with 100% error rate ===')
full_error_sessions = session_stats[session_stats['error_rate'] == 1.0]
print(f"Count: {len(full_error_sessions)} / {len(session_stats)}")
print(full_error_sessions.groupby('model').size())
print()

# Cost efficiency
print('=== Cost Efficiency (cost per successful turn) ===')
df['success'] = 1 - df['has_error']
for model in df['model'].unique():
    mdf = df[df['model'] == model]
    success_count = mdf['success'].sum()
    total_cost = mdf['turn_cost'].sum()
    if success_count > 0:
        print(f"{model}: ${total_cost/success_count:.4f}/successful turn")
    else:
        print(f"{model}: No successful turns!")
print()

# Token efficiency
print('=== Output per Input Token Ratio ===')
for model in df['model'].unique():
    mdf = df[df['model'] == model]
    ratio = mdf['output_length'].sum() / mdf['input_tokens'].sum() if mdf['input_tokens'].sum() > 0 else 0
    print(f"{model}: {ratio:.4f}")
print()

# Correlation analysis
print('=== Correlation Matrix ===')
numeric_cols = ['output_length', 'pre_gap', 'has_error', 'input_tokens', 'turn_number', 'turn_cost']
print(df[numeric_cols].corr().round(3))
print()

# Turn progression analysis
print('=== Error rate by turn number (first 20 turns) ===')
turn_err = df[df['turn_number'] <= 20].groupby('turn_number')['has_error'].mean()
print(turn_err)
print()

# Cost percentile analysis
print('=== Cost Percentiles ===')
for p in [50, 75, 90, 95, 99]:
    print(f"P{p}: ${df['turn_cost'].quantile(p/100):.4f}")
print()

# Latency percentile analysis
print('=== Latency Percentiles ===')
for p in [50, 75, 90, 95, 99]:
    print(f"P{p}: {df['pre_gap'].quantile(p/100):.2f}s")
